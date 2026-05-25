"""
diplomacy_belief_lifecycle.py — outcome-driven evolution of beliefs.

The lifecycle runs after each phase resolves. It does four things:

  1. promote_proto_beliefs:
       PROTO beliefs become ACTIVE when their predictions confirm at high
       enough rate. This is what gives the system "earned" beliefs rather
       than "proposed" beliefs.

  2. retire_active_beliefs:
       ACTIVE beliefs retire under three conditions:
         (a) predictive failure — recent prediction confirm rate dropped
             below threshold
         (b) belief too broad — head gets foveated frequently but its
             predictions are all trivially confirmed, suggesting the
             belief is too vague to fail (the diplomacy analog of
             magic_go's `precondition_too_broad_unused`)
         (c) stale idle — un-foveated past the HP decay threshold

  3. invite_revisions:
       For each retired-by-predictive-failure belief, request the LLM to
       propose a narrower successor. The function signature only — the
       actual LLM call lives in the engine integration layer. We return
       the list of beliefs eligible for revision.

  4. decay_idle:
       HP decay on every un-foveated node. Below threshold → archive.
       Cross-game persistent types (DISPOSITION, CREDIBILITY) decay slower.

DESIGN PRINCIPLES (carried from magic_go):
  - The LLM proposes; deterministic code grades and decides.
  - Hand-rolled thresholds; every threshold has a single source of truth here.
  - Lifecycle is conservative — we'd rather keep a slightly-wrong belief
    around than retire prematurely. Diplomacy belief is softer than Go
    pattern, and the cost of being wrong about a person is higher than
    the cost of being wrong about a board pattern.
"""

from __future__ import annotations

from typing import Optional

from diplomacy_kg_schema import (
    AgentMind, BeliefNode, BeliefStatus, BeliefType,
    PredictionNode, PredictionStatus,
    PhaseKey, new_id,
)
from diplomacy_fovea import _phase_index


# ============================================================================
# Lifecycle thresholds — all hand-tuned, all in one place
# ============================================================================

# --- Proto promotion ---
PROMOTE_MIN_PREDICTIONS = 2          # need at least N predictions before promoting
PROMOTE_MIN_CONFIRMED   = 1          # at least M confirmed
PROMOTE_MIN_CONFIRM_RATE = 0.5       # confirmed / (confirmed + refuted) >= this
                                     # PARTIAL counts as 0.5 in this calc

# --- Active retirement: predictive failure ---
RETIRE_MIN_RECENT_PREDICTIONS = 3    # need at least N recent predictions to judge
RETIRE_MAX_REFUTE_RATE = 0.5         # refute_rate exceeds this → retire
RETIRE_RECENT_WINDOW_PHASES = 6      # "recent" = last N phases

# --- Active retirement: belief too broad ---
BROAD_MIN_FOVEATED = 8               # foveated at least this many times
BROAD_MIN_PREDICTIONS_ALL_CONFIRMED = 4  # AND all 4+ predictions came back CONFIRMED
                                     # AND none of those predictions were REFUTED or PARTIAL
                                     # (i.e., the belief is unfalsifiable)

# --- Stale idle ---
HP_DECAY_PER_PHASE = 0.05
HP_DECAY_PER_PHASE_PERSISTENT = 0.02   # for DISPOSITION + CREDIBILITY
HP_ARCHIVE_THRESHOLD = 0.10            # below this: archive (set status RETIRED, retire_reason=stale_idle)
HP_RECHARGE_ON_FOVEATE = 0.10          # used by the fovea side after a successful pull

# --- Revision proposal ---
REVISION_TRIGGER_MIN_REFUTED = 2     # at least N refuted predictions to invite revision
REVISION_TRIGGER_REFUTE_RATE = 0.4   # AND refute_rate >= this


# ============================================================================
# Helpers
# ============================================================================

def _belief_predictions(mind: AgentMind, belief: BeliefNode) -> list[PredictionNode]:
    """Return predictions that name `belief` as a source."""
    return [
        p for p in mind.predictions.values()
        if belief.id in p.source_belief_ids
    ]


def _recent_predictions(mind: AgentMind, belief: BeliefNode,
                        current_phase: PhaseKey,
                        window_phases: int = RETIRE_RECENT_WINDOW_PHASES,
                        ) -> list[PredictionNode]:
    """Predictions whose formed_at_phase is within `window_phases` of current."""
    cur_idx = _phase_index(current_phase)
    return [
        p for p in _belief_predictions(mind, belief)
        if cur_idx - _phase_index(p.formed_at_phase) <= window_phases
    ]


def _resolved_only(predictions: list[PredictionNode]) -> list[PredictionNode]:
    return [p for p in predictions if p.status in (
        PredictionStatus.CONFIRMED, PredictionStatus.REFUTED,
        PredictionStatus.PARTIAL,
    )]


def _confirm_rate(predictions: list[PredictionNode]) -> tuple[int, int, float]:
    """Returns (confirmed_count, total_resolved, rate).

    PARTIAL counts as 0.5 in the rate calc. Returns rate=0.0 if no predictions resolved.
    """
    resolved = _resolved_only(predictions)
    if not resolved:
        return 0, 0, 0.0
    score = sum(1.0 if p.status == PredictionStatus.CONFIRMED else
                0.5 if p.status == PredictionStatus.PARTIAL else
                0.0
                for p in resolved)
    return (sum(1 for p in resolved if p.status == PredictionStatus.CONFIRMED),
            len(resolved), score / len(resolved))


def _refute_rate(predictions: list[PredictionNode]) -> tuple[int, int, float]:
    """Mirror of _confirm_rate but for REFUTED."""
    resolved = _resolved_only(predictions)
    if not resolved:
        return 0, 0, 0.0
    refuted = sum(1 for p in resolved if p.status == PredictionStatus.REFUTED)
    return refuted, len(resolved), refuted / len(resolved)


def _decay_rate_for(belief: BeliefNode) -> float:
    """How fast a belief's HP decays each phase. Persistent beliefs decay slower."""
    if belief.belief_type in (BeliefType.DISPOSITION, BeliefType.CREDIBILITY):
        return HP_DECAY_PER_PHASE_PERSISTENT
    return HP_DECAY_PER_PHASE


# ============================================================================
# 1. Promote proto-beliefs to active
# ============================================================================

def lifecycle_promote_proto_beliefs(
    mind: AgentMind,
    current_phase: PhaseKey,
) -> list[BeliefNode]:
    """Walk all PROTO beliefs. Promote any whose predictions have:
      - at least PROMOTE_MIN_PREDICTIONS resolved
      - at least PROMOTE_MIN_CONFIRMED confirmed
      - confirm_rate >= PROMOTE_MIN_CONFIRM_RATE

    Returns the list of beliefs promoted in this pass.
    """
    promoted = []
    for belief in mind.beliefs.values():
        if belief.status != BeliefStatus.PROTO:
            continue
        preds = _belief_predictions(mind, belief)
        confirmed, total, rate = _confirm_rate(preds)
        if total < PROMOTE_MIN_PREDICTIONS:
            continue
        if confirmed < PROMOTE_MIN_CONFIRMED:
            continue
        if rate < PROMOTE_MIN_CONFIRM_RATE:
            continue
        belief.status = BeliefStatus.ACTIVE
        belief.last_updated_phase = current_phase
        promoted.append(belief)
    return promoted


# ============================================================================
# 2. Retire active beliefs
# ============================================================================

def lifecycle_retire_active_beliefs(
    mind: AgentMind,
    current_phase: PhaseKey,
) -> list[BeliefNode]:
    """Walk all ACTIVE beliefs. Retire those that fail the discipline tests.

    Three retirement reasons (set on belief.retire_reason):
      - predictive_failure: recent confirm rate too low
      - belief_too_broad: foveated frequently but predictions never fail
                          (the diplomacy immune response)
      - stale_idle: HP decayed below threshold (handled by lifecycle_decay_idle
                    below; included here only when the active belief itself
                    has fallen below threshold)

    Returns the list of beliefs retired in this pass.
    """
    retired = []
    for belief in mind.beliefs.values():
        if belief.status != BeliefStatus.ACTIVE:
            continue

        # Test 1: predictive failure
        recent = _recent_predictions(mind, belief, current_phase)
        recent_resolved = _resolved_only(recent)
        if len(recent_resolved) >= RETIRE_MIN_RECENT_PREDICTIONS:
            refuted, total, rate = _refute_rate(recent)
            if rate > RETIRE_MAX_REFUTE_RATE:
                belief.status = BeliefStatus.RETIRED
                belief.retire_reason = "predictive_failure"
                belief.last_updated_phase = current_phase
                retired.append(belief)
                continue

        # Test 2: belief too broad — diplomacy immune response
        all_preds = _belief_predictions(mind, belief)
        all_resolved = _resolved_only(all_preds)
        if (belief.times_foveated >= BROAD_MIN_FOVEATED
                and len(all_resolved) >= BROAD_MIN_PREDICTIONS_ALL_CONFIRMED):
            all_confirmed = all(
                p.status == PredictionStatus.CONFIRMED for p in all_resolved
            )
            if all_confirmed:
                # Belief is foveated heavily, makes many predictions, but
                # NONE fail. It's likely too vague to fail.
                belief.status = BeliefStatus.RETIRED
                belief.retire_reason = "belief_too_broad"
                belief.last_updated_phase = current_phase
                retired.append(belief)
                continue

        # Test 3 (HP-based) is handled in lifecycle_decay_idle below — that
        # function adjusts hp first, then archives anything that drops below
        # threshold.

    return retired


# ============================================================================
# 3. Invite revisions
# ============================================================================

def lifecycle_invite_revisions(
    mind: AgentMind,
    current_phase: PhaseKey,
) -> list[BeliefNode]:
    """Identify ACTIVE beliefs whose recent track record has decayed enough
    to warrant a revision invitation, but which haven't yet been retired.

    This function returns the *candidates* — the LLM call to actually
    propose a successor lives in the engine integration layer. We just
    surface which beliefs need attention.

    NOTE: in the current setup, predictive_failure retires the belief
    in lifecycle_retire_active_beliefs, then we'd want to invite revision
    for those (now-retired) beliefs. But a "soft" warning state — invite
    revision BEFORE retirement — is also valuable: it means a candidate
    successor can be on probation when the parent retires, smoothing the
    transition.

    This function returns the soft-warning candidates: ACTIVE beliefs
    whose refute rate has crossed REVISION_TRIGGER_REFUTE_RATE but not
    yet RETIRE_MAX_REFUTE_RATE. The caller can ALSO invite revisions for
    just-retired beliefs.
    """
    candidates = []
    for belief in mind.beliefs.values():
        if belief.status != BeliefStatus.ACTIVE:
            continue
        recent = _recent_predictions(mind, belief, current_phase)
        recent_resolved = _resolved_only(recent)
        if len(recent_resolved) < REVISION_TRIGGER_MIN_REFUTED:
            continue
        refuted, total, rate = _refute_rate(recent)
        if rate >= REVISION_TRIGGER_REFUTE_RATE and rate < RETIRE_MAX_REFUTE_RATE:
            candidates.append(belief)
    return candidates


# ============================================================================
# 4. HP decay and stale-idle archival
# ============================================================================

def lifecycle_decay_idle(
    mind: AgentMind,
    current_phase: PhaseKey,
    foveated_belief_ids_this_phase: set[str],
) -> list[BeliefNode]:
    """Decay HP on every belief that wasn't foveated this phase. Recharge HP
    for beliefs that WERE foveated (a soft 'use it or lose it' incentive).

    Beliefs whose HP drops below HP_ARCHIVE_THRESHOLD get retired with
    reason=stale_idle.

    Returns list of beliefs archived in this pass.
    """
    archived = []
    for belief in mind.beliefs.values():
        if belief.status not in (BeliefStatus.ACTIVE, BeliefStatus.PROTO):
            continue
        if belief.id in foveated_belief_ids_this_phase:
            belief.hp = min(belief.hp + HP_RECHARGE_ON_FOVEATE, 1.5)
        else:
            belief.hp = max(0.0, belief.hp - _decay_rate_for(belief))
            if belief.hp < HP_ARCHIVE_THRESHOLD:
                # Cross-game persistent beliefs need a higher bar before archival —
                # we want them to persist even when not actively in play.
                if belief.belief_type in (BeliefType.DISPOSITION, BeliefType.CREDIBILITY):
                    if belief.hp >= HP_ARCHIVE_THRESHOLD * 0.5:
                        continue
                belief.status = BeliefStatus.RETIRED
                belief.retire_reason = "stale_idle"
                belief.last_updated_phase = current_phase
                archived.append(belief)
    return archived


# ============================================================================
# 5. Cross-game reset
# ============================================================================

def lifecycle_reset_game_specific_beliefs(mind: AgentMind) -> list[BeliefNode]:
    """At new-game boundary: retire all TACTICAL_PATTERN, RELATIONSHIP, and
    RISK_ASSESSMENT beliefs. DISPOSITION and CREDIBILITY persist.
    Also retire all StrategicIntentNodes (intent is always game-specific).
    """
    reset = []
    game_specific = (
        BeliefType.TACTICAL_PATTERN,
        BeliefType.RELATIONSHIP,
        BeliefType.RISK_ASSESSMENT,
    )
    for belief in mind.beliefs.values():
        if belief.status not in (BeliefStatus.ACTIVE, BeliefStatus.PROTO):
            continue
        if belief.belief_type in game_specific:
            belief.status = BeliefStatus.RETIRED
            belief.retire_reason = "game_ended_resettable"
            reset.append(belief)
    return reset


# ============================================================================
# 6. The combined lifecycle pass
# ============================================================================

def run_belief_lifecycle(
    mind: AgentMind,
    current_phase: PhaseKey,
    foveated_belief_ids_this_phase: Optional[set[str]] = None,
) -> dict[str, list[BeliefNode]]:
    """Run the full belief lifecycle in dependency order.

    Order matters:
      1. promote first (so a proto-belief whose predictions just confirmed
         can become active before we evaluate "fires too broadly" on it)
      2. retire next (active beliefs that fail the discipline tests)
      3. invite_revisions returns candidates for the LLM call
      4. decay_idle runs last, after all retirements (otherwise a belief
         could be retired twice with conflicting reasons)

    Returns a dict of { phase: list[BeliefNode] } summarizing what happened.
    """
    if foveated_belief_ids_this_phase is None:
        foveated_belief_ids_this_phase = set()

    promoted = lifecycle_promote_proto_beliefs(mind, current_phase)
    retired = lifecycle_retire_active_beliefs(mind, current_phase)
    revision_candidates = lifecycle_invite_revisions(mind, current_phase)
    archived = lifecycle_decay_idle(
        mind, current_phase, foveated_belief_ids_this_phase,
    )
    return {
        "promoted": promoted,
        "retired": retired,
        "revision_candidates": revision_candidates,
        "archived": archived,
    }


# ============================================================================
# Sanity check + worked examples
# ============================================================================

if __name__ == "__main__":
    import time as _t
    from diplomacy_kg_schema import (
        AgentMind, BeliefNode, BeliefType, BeliefStatus,
        PredictionNode, PredictionStatus, PredictionWindowKind,
    )

    print("=" * 72)
    print("BELIEF LIFECYCLE TESTS")
    print("=" * 72)

    # ---- Test 1: proto belief promotes when predictions confirm ----
    print()
    print("Test 1: proto belief promotes when predictions confirm")
    print("-" * 72)
    m1 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    b = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.TACTICAL_PATTERN,
        head="Russia opens A WAR -> GAL when paired with Austria.",
        body="(elided)",
        formed_at_phase="1901-WINTER-ADJUSTMENTS", formed_in_game=1,
        last_updated_phase="1901-WINTER-ADJUSTMENTS",
        status=BeliefStatus.PROTO,
    )
    m1.beliefs[b.id] = b
    # Spawn 3 predictions, 2 confirmed, 1 refuted — confirm rate 2/3 = 0.67 ≥ 0.5
    for i, status in enumerate([PredictionStatus.CONFIRMED, PredictionStatus.CONFIRMED,
                                PredictionStatus.REFUTED]):
        p = PredictionNode(
            id=new_id("pred"), about_power="RUSSIA",
            formed_at_phase="1902-SPRING-MOVES",
            predicted_event_type="move_to",
            predicted_target="GAL", predicted_subject_power=None,
            prediction_window="1902-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.6,
            source_belief_ids=[b.id],
            status=status,
        )
        m1.predictions[p.id] = p

    promoted = lifecycle_promote_proto_beliefs(m1, "1902-FALL-MOVES")
    print(f"  before promote: status={b.status.value}")
    print(f"  promoted: {len(promoted)}")
    print(f"  after promote: status={b.status.value}")
    assert len(promoted) == 1, f"Expected 1 promotion, got {len(promoted)}"
    assert b.status == BeliefStatus.ACTIVE

    # ---- Test 2: proto belief stays proto when predictions fail ----
    print()
    print("Test 2: proto belief stays proto when predictions fail")
    print("-" * 72)
    m2 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    b2 = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.TACTICAL_PATTERN,
        head="Russia commits to alliances early.", body="(elided)",
        formed_at_phase="1901-WINTER-ADJUSTMENTS", formed_in_game=1,
        last_updated_phase="1901-WINTER-ADJUSTMENTS",
        status=BeliefStatus.PROTO,
    )
    m2.beliefs[b2.id] = b2
    # 3 predictions, 1 confirmed, 2 refuted — 1/3 = 0.33 < 0.5
    for status in [PredictionStatus.CONFIRMED, PredictionStatus.REFUTED, PredictionStatus.REFUTED]:
        p = PredictionNode(
            id=new_id("pred"), about_power="RUSSIA",
            formed_at_phase="1902-SPRING-MOVES",
            predicted_event_type="alliance",
            predicted_target=None, predicted_subject_power="GERMANY",
            prediction_window="1902-FALL-MOVES",
            window_kind=PredictionWindowKind.LONG_HORIZON,
            confidence=0.6,
            source_belief_ids=[b2.id],
            status=status,
        )
        m2.predictions[p.id] = p

    promoted = lifecycle_promote_proto_beliefs(m2, "1902-FALL-MOVES")
    print(f"  predictions: 1 confirmed, 2 refuted (rate 0.33)")
    print(f"  promoted: {len(promoted)}")
    print(f"  status remains: {b2.status.value}")
    assert len(promoted) == 0
    assert b2.status == BeliefStatus.PROTO

    # ---- Test 3: active belief retires after enough refutations ----
    print()
    print("Test 3: active belief retires after refutations exceed threshold")
    print("-" * 72)
    m3 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    b3 = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.RELATIONSHIP,
        head="Russia is allied with Austria this game.", body="(elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1902-SPRING-MOVES",
        status=BeliefStatus.ACTIVE,
    )
    m3.beliefs[b3.id] = b3
    # Recent predictions: 1 confirmed, 3 refuted — refute rate 0.75 > 0.5
    for status in [PredictionStatus.CONFIRMED, PredictionStatus.REFUTED,
                   PredictionStatus.REFUTED, PredictionStatus.REFUTED]:
        p = PredictionNode(
            id=new_id("pred"), about_power="RUSSIA",
            formed_at_phase="1902-FALL-MOVES",   # recent
            predicted_event_type="non_action",
            predicted_target="GAL", predicted_subject_power=None,
            prediction_window="1903-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.6,
            source_belief_ids=[b3.id],
            status=status,
        )
        m3.predictions[p.id] = p

    retired = lifecycle_retire_active_beliefs(m3, "1903-FALL-MOVES")
    print(f"  predictions: 1 confirmed, 3 refuted (refute rate 0.75)")
    print(f"  retired: {len(retired)}")
    print(f"  retire_reason: {b3.retire_reason}")
    assert len(retired) == 1
    assert b3.status == BeliefStatus.RETIRED
    assert b3.retire_reason == "predictive_failure"

    # ---- Test 4: belief retires for being too broad (immune response) ----
    print()
    print("Test 4: belief retires for being too broad (diplomacy immune response)")
    print("-" * 72)
    m4 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    b4 = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.DISPOSITION,
        head="Russia plays Diplomacy.", body="(elided)",   # vacuous
        formed_at_phase="1901-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1903-FALL-MOVES",
        status=BeliefStatus.ACTIVE,
        times_foveated=12,    # ≥ BROAD_MIN_FOVEATED (8)
    )
    m4.beliefs[b4.id] = b4
    # 5 predictions, all confirmed — never failed, suspiciously perfect
    for _ in range(5):
        p = PredictionNode(
            id=new_id("pred"), about_power="RUSSIA",
            formed_at_phase="1902-SPRING-MOVES",
            predicted_event_type="move_to",
            predicted_target="GAL", predicted_subject_power=None,
            prediction_window="1902-FALL-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.6,
            source_belief_ids=[b4.id],
            status=PredictionStatus.CONFIRMED,
        )
        m4.predictions[p.id] = p

    retired = lifecycle_retire_active_beliefs(m4, "1903-FALL-MOVES")
    print(f"  foveated: {b4.times_foveated} times")
    print(f"  predictions: 5 confirmed, 0 refuted (suspiciously clean)")
    print(f"  retired: {len(retired)}, reason: {b4.retire_reason}")
    assert len(retired) == 1
    assert b4.retire_reason == "belief_too_broad"

    # ---- Test 5: revision invitation for slipping belief ----
    print()
    print("Test 5: revision candidate identified for slipping belief")
    print("-" * 72)
    m5 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    b5 = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.RELATIONSHIP,
        head="Russia is reliable in tactical commitments.", body="(elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1902-FALL-MOVES",
        status=BeliefStatus.ACTIVE,
    )
    m5.beliefs[b5.id] = b5
    # 4 recent predictions: 2 confirmed, 1 refuted, 1 partial — refute rate 0.25
    # That's between REVISION_TRIGGER_REFUTE_RATE (0.4) and
    # RETIRE_MAX_REFUTE_RATE (0.5). Won't qualify for revision yet.
    # Bump to 2 refutes / 4 = 0.5 → that's >= 0.5, so retires instead.
    # Use 2/5 = 0.4 to land in the revision band.
    statuses = [PredictionStatus.CONFIRMED, PredictionStatus.CONFIRMED,
                PredictionStatus.CONFIRMED, PredictionStatus.REFUTED,
                PredictionStatus.REFUTED]
    for s in statuses:
        p = PredictionNode(
            id=new_id("pred"), about_power="RUSSIA",
            formed_at_phase="1902-FALL-MOVES",
            predicted_event_type="move_to",
            predicted_target="GAL", predicted_subject_power=None,
            prediction_window="1903-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.6,
            source_belief_ids=[b5.id],
            status=s,
        )
        m5.predictions[p.id] = p

    candidates = lifecycle_invite_revisions(m5, "1903-FALL-MOVES")
    print(f"  predictions: 3 confirmed, 2 refuted (refute rate 0.4)")
    print(f"  revision candidates: {len(candidates)}")
    assert len(candidates) == 1, f"Expected 1 revision candidate, got {len(candidates)}"

    # ---- Test 6: HP decay on idle belief, archival below threshold ----
    print()
    print("Test 6: HP decay archives a long-idle belief")
    print("-" * 72)
    m6 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    b6 = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.TACTICAL_PATTERN,    # not persistent
        head="(stale tactical observation)", body="(elided)",
        formed_at_phase="1901-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1901-SPRING-MOVES",
        status=BeliefStatus.ACTIVE,
        hp=0.12,        # already low
    )
    m6.beliefs[b6.id] = b6
    # No fovea hit this phase
    archived = lifecycle_decay_idle(m6, "1903-FALL-MOVES",
                                    foveated_belief_ids_this_phase=set())
    print(f"  hp before: 0.12, decay 0.05 → hp after: {b6.hp:.3f}")
    print(f"  archived: {len(archived)}, reason: {b6.retire_reason}")
    assert len(archived) == 1
    assert b6.retire_reason == "stale_idle"

    # ---- Test 7: HP decay slower for persistent belief types ----
    print()
    print("Test 7: persistent belief decays slower")
    print("-" * 72)
    m7 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    b7 = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.DISPOSITION,
        head="Russia plays defensively.", body="(elided)",
        formed_at_phase="1901-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1901-SPRING-MOVES",
        status=BeliefStatus.ACTIVE,
        hp=0.12,
        persists_across_games=True,
    )
    m7.beliefs[b7.id] = b7
    archived = lifecycle_decay_idle(m7, "1903-FALL-MOVES",
                                    foveated_belief_ids_this_phase=set())
    print(f"  hp before: 0.12, persistent-decay 0.02 → hp after: {b7.hp:.3f}")
    print(f"  archived: {len(archived)} (expected 0 — protected by persistence)")
    assert len(archived) == 0
    assert b7.status == BeliefStatus.ACTIVE

    # ---- Test 8: cross-game reset retires game-specific belief types ----
    print()
    print("Test 8: cross-game reset retires game-specific beliefs only")
    print("-" * 72)
    m8 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    persistent = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.DISPOSITION,
        head="Russia plays defensively.", body="",
        formed_at_phase="1901-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1903-WINTER-ADJUSTMENTS",
        status=BeliefStatus.ACTIVE, persists_across_games=True,
    )
    relational = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.RELATIONSHIP,
        head="Russia and Austria are allied this game.", body="",
        formed_at_phase="1902-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1903-WINTER-ADJUSTMENTS",
        status=BeliefStatus.ACTIVE,
    )
    m8.beliefs[persistent.id] = persistent
    m8.beliefs[relational.id] = relational
    reset = lifecycle_reset_game_specific_beliefs(m8)
    print(f"  reset: {len(reset)}")
    print(f"  persistent ({persistent.belief_type.value}): {persistent.status.value}")
    print(f"  relational ({relational.belief_type.value}): {relational.status.value}")
    assert len(reset) == 1
    assert persistent.status == BeliefStatus.ACTIVE
    assert relational.status == BeliefStatus.RETIRED

    # ---- Test 9: full lifecycle pass ----
    print()
    print("Test 9: full lifecycle pass — promote + retire + decay together")
    print("-" * 72)
    m9 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    # Reuse the proto-promote candidate
    proto = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.TACTICAL_PATTERN,
        head="Russia opens via Galicia.", body="",
        formed_at_phase="1901-WINTER-ADJUSTMENTS", formed_in_game=1,
        last_updated_phase="1901-WINTER-ADJUSTMENTS",
        status=BeliefStatus.PROTO,
    )
    m9.beliefs[proto.id] = proto
    for s in [PredictionStatus.CONFIRMED, PredictionStatus.CONFIRMED]:
        p = PredictionNode(
            id=new_id("pred"), about_power="RUSSIA",
            formed_at_phase="1902-SPRING-MOVES",
            predicted_event_type="move_to",
            predicted_target="GAL", predicted_subject_power=None,
            prediction_window="1902-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.7,
            source_belief_ids=[proto.id],
            status=s,
        )
        m9.predictions[p.id] = p

    summary = run_belief_lifecycle(m9, "1902-FALL-MOVES",
                                   foveated_belief_ids_this_phase={proto.id})
    print(f"  summary:")
    for k, v in summary.items():
        print(f"    {k}: {len(v)}")
    assert len(summary["promoted"]) == 1
    assert proto.status == BeliefStatus.ACTIVE

    print()
    print("All belief lifecycle tests passed.")
