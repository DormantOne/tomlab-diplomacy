"""
diplomacy_strategy_lifecycle.py — outcome-driven evolution of strategic intent.

This is the diplomacy translation of magic_go's strategy machinery:
  - proto_strategy → active_strategy → retired/revised
  - strategy_commitment carrying an active strategy across multiple moves
  - strategy_revision proposals when an active strategy fails

The diplomacy translation:
  - StrategicIntentNode replaces strategy_node
  - PlanNode replaces plan_node (per-phase tactical plan; supporting evidence)
  - IntentCommitmentNode replaces strategy_commitment (carries an active
    intent across multiple PHASES with divergence/failure thresholds)
  - StrategicIntentRevisionProposal replaces strategy_revision

The whole module mirrors diplomacy_belief_lifecycle.py in shape; the
parallel is intentional. Functions:

  1. lifecycle_review_plans
       After a phase resolves, grade every PlanNode formed at or before
       that phase. Roll outcome stats up into the parent intent.

  2. lifecycle_promote_proto_intents
       Proto-intents earn ACTIVE when they accumulate supporting plans +
       confirmed predictions + non-negative SC delta.

  3. lifecycle_retire_active_intents
       Three retirement conditions:
       (a) predictive_failure: recent prediction confirm rate too low
       (b) intent_too_broad: ACTIVE for >=4 phases with >=3 supporting plans
           but sc_delta_under_intent <= 0 (the diplomacy immune response)
       (c) horizon_passed_failed: declared horizon reached without success

  4. lifecycle_invite_intent_revisions
       Surfaces failing intents as revision candidates. (LLM call elsewhere.)

  5. lifecycle_review_intent_commitments
       The magic_go strategy_commitment analog. Window reached, divergence
       threshold breached, or predictive failure mid-commitment.
"""

from __future__ import annotations

from typing import Optional

from diplomacy_kg_schema import (
    AgentMind,
    StrategicIntentNode, StrategicIntentStatus,
    PlanNode, IntentCommitmentNode, StrategicIntentRevisionProposal,
    PredictionNode, PredictionStatus,
    PhaseState,
    PhaseKey, new_id,
)
from diplomacy_fovea import _phase_index


# ============================================================================
# Lifecycle thresholds — all hand-tuned, all in one place
# ============================================================================

# --- Plan review ---
PLAN_ADVANCED_MIN_SC_DELTA = 1       # SC gained this phase = advanced
PLAN_SETBACK_MIN_SC_LOSS   = 1       # SC lost this phase = setback (negative direction)

# --- Proto promotion ---
PROMOTE_INTENT_MIN_PLANS         = 2     # at least N plans support it
PROMOTE_INTENT_MIN_CONFIRMED_PREDS = 2   # at least M predictions confirmed
PROMOTE_INTENT_MIN_CONFIRM_RATE   = 0.5  # confirmed / (confirmed + refuted)
PROMOTE_INTENT_MIN_SC_DELTA       = 0    # non-negative SC delta required

# --- Active retirement ---
RETIRE_INTENT_MIN_RECENT_PREDS  = 3      # need this many recent preds to judge
RETIRE_INTENT_MAX_REFUTE_RATE   = 0.5    # refute_rate above this → retire
RETIRE_INTENT_RECENT_WINDOW     = 6      # phases counted as "recent"

# --- Immune response: intent too broad ---
BROAD_INTENT_MIN_ACTIVE_PHASES   = 4     # been ACTIVE for at least N phases
BROAD_INTENT_MIN_PLANS_SUPPORTED = 3     # at least N plans named it as parent
BROAD_INTENT_MAX_SC_DELTA        = 0     # but produced ≤ 0 net SC change

# --- Revision ---
REVISION_INTENT_MIN_REFUTED   = 2
REVISION_INTENT_REFUTE_RATE   = 0.4

# --- Intent commitment ---
COMMITMENT_DEFAULT_WINDOW_PHASES = 3      # near-term intents committed for 3 phases
COMMITMENT_MAX_DIVERGENCES       = 2      # plans naming a different parent_intent
COMMITMENT_MAX_REFUTED_PREDICTIONS = 2    # predictions during commitment refuted


# ============================================================================
# Helpers (parallels to diplomacy_belief_lifecycle helpers)
# ============================================================================

def _intent_predictions(mind: AgentMind, intent: StrategicIntentNode
                        ) -> list[PredictionNode]:
    """Predictions linked to this intent (parent_intent_id matches)."""
    return [
        p for p in mind.predictions.values()
        if p.parent_intent_id == intent.id
    ]


def _recent_intent_predictions(
    mind: AgentMind, intent: StrategicIntentNode,
    current_phase: PhaseKey,
    window: int = RETIRE_INTENT_RECENT_WINDOW,
) -> list[PredictionNode]:
    cur_idx = _phase_index(current_phase)
    return [
        p for p in _intent_predictions(mind, intent)
        if cur_idx - _phase_index(p.formed_at_phase) <= window
    ]


def _resolved_only(predictions: list[PredictionNode]) -> list[PredictionNode]:
    return [p for p in predictions if p.status in (
        PredictionStatus.CONFIRMED, PredictionStatus.REFUTED,
        PredictionStatus.PARTIAL,
    )]


def _confirm_rate(predictions: list[PredictionNode]) -> tuple[int, int, float]:
    resolved = _resolved_only(predictions)
    if not resolved:
        return 0, 0, 0.0
    confirmed = sum(1 for p in resolved if p.status == PredictionStatus.CONFIRMED)
    score = sum(1.0 if p.status == PredictionStatus.CONFIRMED else
                0.5 if p.status == PredictionStatus.PARTIAL else
                0.0
                for p in resolved)
    return confirmed, len(resolved), score / len(resolved)


def _refute_rate(predictions: list[PredictionNode]) -> tuple[int, int, float]:
    resolved = _resolved_only(predictions)
    if not resolved:
        return 0, 0, 0.0
    refuted = sum(1 for p in resolved if p.status == PredictionStatus.REFUTED)
    return refuted, len(resolved), refuted / len(resolved)


def _intent_plans(mind: AgentMind, intent: StrategicIntentNode) -> list[PlanNode]:
    return [
        p for p in mind.plan_nodes.values()
        if p.parent_intent_id == intent.id
    ]


def _phases_active(intent: StrategicIntentNode, current_phase: PhaseKey) -> int:
    """How many phases this intent has been ACTIVE.

    Uses active_since_phase (set when promoted to ACTIVE), not formed_at_phase.
    A freshly-promoted intent reports 0 phases active even if it was formed
    several phases ago — protects newly-active intents from the broad-intent
    immune response firing on phases spent in PROTO.
    """
    if intent.status != StrategicIntentStatus.ACTIVE:
        return 0
    if intent.active_since_phase is None:
        # Edge case: intent is ACTIVE but never had active_since_phase set
        # (e.g. test fixture). Fall back to formed_at_phase but log this
        # is a defensive fallback.
        return _phase_index(current_phase) - _phase_index(intent.formed_at_phase)
    return _phase_index(current_phase) - _phase_index(intent.active_since_phase)


# ============================================================================
# 1. Plan review — runs every phase
# ============================================================================

def lifecycle_review_plans(
    mind: AgentMind,
    just_resolved_phase: PhaseKey,
) -> list[PlanNode]:
    """For every pending PlanNode whose phase has resolved:
       - grade outcome from emitted predictions + sc_delta_this_phase
       - mark plan_outcome ∈ {advanced, neutral, setback}
       - roll stats up into parent intent
       - mark plan reviewed

    Returns the list of plans newly reviewed.

    NOTE: sc_delta_this_phase is expected to have been set by the caller
    (the engine integration layer) when the phase resolved. We don't compute
    it here because that would require the phase_state diff and we want to
    keep this function pure over the data already in the mind.
    """
    reviewed = []
    just_resolved_idx = _phase_index(just_resolved_phase)
    for plan in mind.plan_nodes.values():
        if plan.status != "pending":
            continue
        if _phase_index(plan.formed_at_phase) > just_resolved_idx:
            continue

        # Determine outcome from sc_delta + emitted prediction grades
        emitted_preds = [
            p for p in mind.predictions.values()
            if p.id in plan.emitted_prediction_ids
        ]
        confirmed_count, total, _ = _confirm_rate(emitted_preds)

        if plan.sc_delta_this_phase >= PLAN_ADVANCED_MIN_SC_DELTA:
            plan.plan_outcome = "advanced"
        elif plan.sc_delta_this_phase <= -PLAN_SETBACK_MIN_SC_LOSS:
            plan.plan_outcome = "setback"
        elif total > 0 and confirmed_count == total:
            plan.plan_outcome = "advanced"     # predictions all hit, neutral SC
        elif total > 0 and confirmed_count == 0:
            plan.plan_outcome = "setback"      # predictions all missed
        else:
            plan.plan_outcome = "neutral"

        # Roll stats up into parent intent
        if plan.parent_intent_id and plan.parent_intent_id in mind.strategic_intents:
            intent = mind.strategic_intents[plan.parent_intent_id]
            intent.sc_delta_under_intent += plan.sc_delta_this_phase
            intent.predictions_confirmed += confirmed_count
            intent.predictions_refuted += sum(
                1 for p in emitted_preds if p.status == PredictionStatus.REFUTED
            )
            if plan.id not in intent.supporting_plan_ids:
                intent.supporting_plan_ids.append(plan.id)

        plan.status = "reviewed"
        reviewed.append(plan)

    return reviewed


# ============================================================================
# 2. Promote proto-intents
# ============================================================================

def lifecycle_promote_proto_intents(
    mind: AgentMind,
    current_phase: PhaseKey,
) -> list[StrategicIntentNode]:
    """Walk all PROTO intents. Promote any that have accumulated:
       - at least PROMOTE_INTENT_MIN_PLANS supporting plans
       - at least PROMOTE_INTENT_MIN_CONFIRMED_PREDS confirmed predictions
       - confirm_rate >= PROMOTE_INTENT_MIN_CONFIRM_RATE
       - sc_delta_under_intent >= PROMOTE_INTENT_MIN_SC_DELTA

    Returns the list of intents promoted in this pass.
    """
    promoted = []
    for intent in mind.strategic_intents.values():
        if intent.status != StrategicIntentStatus.PROTO:
            continue

        if len(intent.supporting_plan_ids) < PROMOTE_INTENT_MIN_PLANS:
            continue

        preds = _intent_predictions(mind, intent)
        confirmed_count, total, rate = _confirm_rate(preds)

        if confirmed_count < PROMOTE_INTENT_MIN_CONFIRMED_PREDS:
            continue
        if rate < PROMOTE_INTENT_MIN_CONFIRM_RATE:
            continue
        if intent.sc_delta_under_intent < PROMOTE_INTENT_MIN_SC_DELTA:
            continue

        intent.status = StrategicIntentStatus.ACTIVE
        intent.active_since_phase = current_phase
        promoted.append(intent)

    return promoted


# ============================================================================
# 3. Retire active intents
# ============================================================================

def lifecycle_retire_active_intents(
    mind: AgentMind,
    current_phase: PhaseKey,
) -> list[StrategicIntentNode]:
    """Walk all ACTIVE intents. Retire those that fail the discipline tests:
       (a) predictive_failure: recent prediction refute rate too high
       (b) intent_too_broad: long-active without producing SC gain
                              (the diplomacy immune response)
       (c) horizon_passed_failed: horizon reached, success criteria unmet

    Returns the list of intents retired in this pass.
    """
    retired = []
    for intent in mind.strategic_intents.values():
        if intent.status != StrategicIntentStatus.ACTIVE:
            continue

        # Test 1: predictive failure
        recent = _recent_intent_predictions(mind, intent, current_phase)
        recent_resolved = _resolved_only(recent)
        if len(recent_resolved) >= RETIRE_INTENT_MIN_RECENT_PREDS:
            refuted, total, rate = _refute_rate(recent)
            if rate > RETIRE_INTENT_MAX_REFUTE_RATE:
                intent.status = StrategicIntentStatus.RETIRED
                intent.retire_reason = "predictive_failure"
                retired.append(intent)
                continue

        # Test 2: intent too broad (the immune response)
        phases_active = _phases_active(intent, current_phase)
        if (phases_active >= BROAD_INTENT_MIN_ACTIVE_PHASES
                and len(intent.supporting_plan_ids) >= BROAD_INTENT_MIN_PLANS_SUPPORTED
                and intent.sc_delta_under_intent <= BROAD_INTENT_MAX_SC_DELTA):
            intent.status = StrategicIntentStatus.RETIRED
            intent.retire_reason = "intent_too_broad"
            retired.append(intent)
            continue

        # Test 3: horizon passed without success
        if _phase_index(current_phase) >= _phase_index(intent.horizon):
            # The horizon has been reached. Check whether success criteria
            # are met. For now: any positive sc_delta is success;
            # zero or negative is failure.
            if intent.sc_delta_under_intent > 0:
                intent.status = StrategicIntentStatus.SUCCEEDED
            else:
                intent.status = StrategicIntentStatus.FAILED
                intent.retire_reason = "horizon_passed_failed"
            retired.append(intent)
            continue

    return retired


# ============================================================================
# 4. Invite intent revisions
# ============================================================================

def lifecycle_invite_intent_revisions(
    mind: AgentMind,
    current_phase: PhaseKey,
) -> list[StrategicIntentNode]:
    """Identify ACTIVE intents whose track record has slipped enough to
    warrant a revision invitation, but which haven't yet been retired.

    Returns the list of candidate intents. The actual LLM call to propose
    a successor lives in the engine integration layer.
    """
    candidates = []
    for intent in mind.strategic_intents.values():
        if intent.status != StrategicIntentStatus.ACTIVE:
            continue

        recent = _recent_intent_predictions(mind, intent, current_phase)
        recent_resolved = _resolved_only(recent)
        if len(recent_resolved) < REVISION_INTENT_MIN_REFUTED:
            continue
        refuted, total, rate = _refute_rate(recent)
        if rate >= REVISION_INTENT_REFUTE_RATE and rate < RETIRE_INTENT_MAX_REFUTE_RATE:
            candidates.append(intent)

    return candidates


# ============================================================================
# 5. Intent commitment review
# ============================================================================

def lifecycle_review_intent_commitments(
    mind: AgentMind,
    just_resolved_phase: PhaseKey,
) -> list[IntentCommitmentNode]:
    """Walk active IntentCommitmentNodes. End each that has:
       - reached its window (window_reached_completed → terminal status)
       - accumulated too many divergent plans (abandoned)
       - accumulated too many refuted predictions during commitment
         (abandoned with predictive_failure)

    For window-completed commitments, mark the parent intent SUCCEEDED if
    predictions confirmed at high rate and sc_delta met expectations,
    otherwise FAILED.

    Returns the list of commitments whose status changed in this pass.
    """
    just_resolved_idx = _phase_index(just_resolved_phase)
    changed = []

    for commitment in mind.intent_commitments.values():
        if commitment.status != "active":
            continue

        # Tally divergent plans and predictions during the commitment window
        # for this commitment's intent.
        intent = mind.strategic_intents.get(commitment.intent_id)
        if intent is None:
            # Parent intent missing — orphan commitment. Close it.
            commitment.status = "abandoned"
            commitment.end_reason = "superseded"
            commitment.ended_at_phase = just_resolved_phase
            changed.append(commitment)
            continue

        # Update the running window
        if just_resolved_phase not in commitment.phases_under_commitment:
            commitment.phases_under_commitment.append(just_resolved_phase)

        # Plans formed during the commitment window
        for plan in mind.plan_nodes.values():
            if plan.formed_at_phase not in commitment.phases_under_commitment:
                continue
            if plan.parent_intent_id == intent.id:
                # Count plans whose parent is this intent
                if plan.id not in intent.supporting_plan_ids:
                    pass  # already rolled up by lifecycle_review_plans
                # plans_followed is incremented per-phase; we set absolute count
            else:
                # divergence
                pass

        # Recount from authoritative source: walk plan_nodes once
        followed = 0
        diverged = 0
        for plan in mind.plan_nodes.values():
            if plan.formed_at_phase not in commitment.phases_under_commitment:
                continue
            if plan.parent_intent_id == intent.id:
                followed += 1
            else:
                diverged += 1
        commitment.plans_followed = followed
        commitment.plans_diverged = diverged

        # Refuted predictions during the commitment window
        refuted_during = 0
        confirmed_during = 0
        for pred in mind.predictions.values():
            if pred.parent_intent_id != intent.id:
                continue
            if pred.formed_at_phase not in commitment.phases_under_commitment:
                continue
            if pred.status == PredictionStatus.REFUTED:
                refuted_during += 1
            elif pred.status == PredictionStatus.CONFIRMED:
                confirmed_during += 1
        commitment.predictions_refuted_during = refuted_during
        commitment.predictions_confirmed_during = confirmed_during

        # Test for early-end conditions
        phases_done = len(commitment.phases_under_commitment)

        if refuted_during > COMMITMENT_MAX_REFUTED_PREDICTIONS:
            commitment.status = "abandoned"
            commitment.end_reason = "abandoned_predictive_failure"
            commitment.ended_at_phase = just_resolved_phase
            changed.append(commitment)
            continue

        if diverged > COMMITMENT_MAX_DIVERGENCES:
            commitment.status = "abandoned"
            commitment.end_reason = "abandoned_too_many_divergences"
            commitment.ended_at_phase = just_resolved_phase
            changed.append(commitment)
            continue

        # Window reached?
        if phases_done >= commitment.window_phases:
            commitment.status = "completed"
            commitment.end_reason = "window_reached_completed"
            commitment.ended_at_phase = just_resolved_phase

            # Update parent intent status based on commitment outcome —
            # but ONLY if the intent is still ACTIVE. If the intent was
            # retired or revised mid-window, leave its status alone; the
            # commitment can still close cleanly.
            if intent.status == StrategicIntentStatus.ACTIVE:
                total_resolved = confirmed_during + refuted_during
                confirm_rate = (confirmed_during / total_resolved
                                if total_resolved > 0 else 0.0)
                if (confirm_rate >= 0.6
                        and intent.sc_delta_under_intent > 0):
                    intent.status = StrategicIntentStatus.SUCCEEDED
                elif intent.sc_delta_under_intent <= 0:
                    intent.status = StrategicIntentStatus.FAILED
                    intent.retire_reason = "horizon_passed_failed"
                # else: intent remains ACTIVE and may be re-committed later
            changed.append(commitment)
            continue

    return changed


# ============================================================================
# 6. Combined lifecycle pass
# ============================================================================

def run_strategy_lifecycle(
    mind: AgentMind,
    current_phase: PhaseKey,
    just_resolved_phase: PhaseKey,
) -> dict[str, list]:
    """Run the full strategy lifecycle in dependency order.

    Order matters:
      1. Review plans first (they roll stats up to intents)
      2. Review intent commitments (they may change intent status)
      3. Promote proto-intents (using freshly-rolled-up stats)
      4. Retire active intents (whose stats now reflect this phase)
      5. Invite revisions (surface candidates for the LLM call)

    Returns a dict summarizing what happened.
    """
    plans_reviewed = lifecycle_review_plans(mind, just_resolved_phase)
    commitments_changed = lifecycle_review_intent_commitments(
        mind, just_resolved_phase,
    )
    promoted = lifecycle_promote_proto_intents(mind, current_phase)
    retired = lifecycle_retire_active_intents(mind, current_phase)
    revision_candidates = lifecycle_invite_intent_revisions(mind, current_phase)

    return {
        "plans_reviewed": plans_reviewed,
        "commitments_changed": commitments_changed,
        "promoted": promoted,
        "retired": retired,
        "revision_candidates": revision_candidates,
    }


# ============================================================================
# 7. Helper for starting a new commitment
# ============================================================================

def start_intent_commitment(
    mind: AgentMind,
    intent_id: str,
    started_at_phase: PhaseKey,
    window_phases: int = COMMITMENT_DEFAULT_WINDOW_PHASES,
) -> Optional[IntentCommitmentNode]:
    """Open a new IntentCommitmentNode for an active intent.

    Called by the engine integration layer when the agent decides to
    commit to an active intent (typically right after promotion, or after
    a previous commitment ended and the intent is still active).

    Returns None if the intent doesn't exist or isn't active.
    """
    intent = mind.strategic_intents.get(intent_id)
    if intent is None:
        return None
    if intent.status != StrategicIntentStatus.ACTIVE:
        return None

    commitment = IntentCommitmentNode(
        id=new_id("ic"),
        intent_id=intent_id,
        started_at_phase=started_at_phase,
        window_phases=window_phases,
    )
    mind.intent_commitments[commitment.id] = commitment
    intent.times_committed += 1
    return commitment


# ============================================================================
# Sanity check + worked examples
# ============================================================================

if __name__ == "__main__":
    import time as _t
    from diplomacy_kg_schema import (
        AgentMind, StrategicIntentNode, StrategicIntentStatus,
        PlanNode, IntentCommitmentNode,
        PredictionNode, PredictionStatus, PredictionWindowKind,
    )

    print("=" * 72)
    print("STRATEGY LIFECYCLE TESTS")
    print("=" * 72)

    # ---- Test 1: plan review rolls outcome up to parent intent ----
    print()
    print("Test 1: plan review — outcome rolls into parent intent")
    print("-" * 72)
    m1 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    intent = StrategicIntentNode(
        id=new_id("intent"),
        head="Take Belgium by 1903.",
        body="(elided)",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        target_powers=["GERMANY"], target_provinces=["BEL"],
        horizon="1903-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.PROTO,
    )
    m1.strategic_intents[intent.id] = intent

    plan = PlanNode(
        id=new_id("plan"),
        formed_at_phase="1902-SPRING-MOVES",
        head="Move A PAR to BUR; pressure GER on Belgium next.",
        body="",
        parent_intent_id=intent.id,
        sc_delta_this_phase=1,    # gained 1 SC
    )
    m1.plan_nodes[plan.id] = plan

    reviewed = lifecycle_review_plans(m1, "1902-SPRING-MOVES")
    print(f"  plans reviewed: {len(reviewed)}")
    print(f"  plan outcome: {plan.plan_outcome}")
    print(f"  intent.sc_delta_under_intent: {intent.sc_delta_under_intent}")
    print(f"  intent.supporting_plan_ids: {len(intent.supporting_plan_ids)}")
    assert plan.plan_outcome == "advanced"
    assert intent.sc_delta_under_intent == 1
    assert plan.id in intent.supporting_plan_ids

    # ---- Test 2: proto intent promotes when criteria met ----
    print()
    print("Test 2: proto intent promotes when criteria met")
    print("-" * 72)
    m2 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    intent2 = StrategicIntentNode(
        id=new_id("intent"),
        head="Take Belgium by 1903.", body="",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        target_powers=["GERMANY"], target_provinces=["BEL"],
        horizon="1903-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.PROTO,
        sc_delta_under_intent=1,
        supporting_plan_ids=[new_id("plan"), new_id("plan")],   # 2 plans
    )
    m2.strategic_intents[intent2.id] = intent2
    # 2 confirmed predictions
    for _ in range(2):
        p = PredictionNode(
            id=new_id("pred"), about_power="GERMANY",
            formed_at_phase="1902-SPRING-MOVES",
            predicted_event_type="non_action",
            predicted_target="BEL", predicted_subject_power=None,
            prediction_window="1902-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.6,
            parent_intent_id=intent2.id,
            status=PredictionStatus.CONFIRMED,
        )
        m2.predictions[p.id] = p

    promoted = lifecycle_promote_proto_intents(m2, "1902-FALL-MOVES")
    print(f"  promoted: {len(promoted)}")
    print(f"  intent status: {intent2.status.value}")
    assert len(promoted) == 1
    assert intent2.status == StrategicIntentStatus.ACTIVE

    # ---- Test 3: active intent retires for predictive failure ----
    print()
    print("Test 3: active intent retires for predictive failure")
    print("-" * 72)
    m3 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    intent3 = StrategicIntentNode(
        id=new_id("intent"),
        head="Coordinate with England against Germany.", body="",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        target_powers=["ENGLAND", "GERMANY"], target_provinces=[],
        horizon="1903-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.ACTIVE,
    )
    m3.strategic_intents[intent3.id] = intent3
    # Recent: 1 confirmed, 3 refuted
    for s in [PredictionStatus.CONFIRMED, PredictionStatus.REFUTED,
              PredictionStatus.REFUTED, PredictionStatus.REFUTED]:
        p = PredictionNode(
            id=new_id("pred"), about_power="ENGLAND",
            formed_at_phase="1902-FALL-MOVES",
            predicted_event_type="alliance",
            predicted_target=None, predicted_subject_power="FRANCE",
            prediction_window="1903-SPRING-MOVES",
            window_kind=PredictionWindowKind.LONG_HORIZON,
            confidence=0.6,
            parent_intent_id=intent3.id,
            status=s,
        )
        m3.predictions[p.id] = p

    retired = lifecycle_retire_active_intents(m3, "1903-FALL-MOVES")
    print(f"  predictions: 1 confirmed, 3 refuted (refute rate 0.75)")
    print(f"  retired: {len(retired)}, reason: {intent3.retire_reason}")
    assert len(retired) == 1
    assert intent3.status == StrategicIntentStatus.RETIRED
    assert intent3.retire_reason == "predictive_failure"

    # ---- Test 4: intent_too_broad — diplomacy immune response ----
    print()
    print("Test 4: intent_too_broad — diplomacy immune response")
    print("-" * 72)
    m4 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    # Active for many phases, 3 plans support it, but no SC gain
    intent4 = StrategicIntentNode(
        id=new_id("intent"),
        head="Maintain a flexible posture across the continent.",
        body="(vague — no concrete target)",
        formed_at_phase="1901-SPRING-MOVES",        # 5+ phases ago
        target_powers=["FRANCE"], target_provinces=[],
        horizon="1905-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.ACTIVE,
        sc_delta_under_intent=0,     # zero net gain
        supporting_plan_ids=[new_id("plan") for _ in range(4)],  # 4 plans
    )
    m4.strategic_intents[intent4.id] = intent4

    retired = lifecycle_retire_active_intents(m4, "1903-SPRING-MOVES")
    print(f"  active phases: ~{_phases_active(intent4, '1903-SPRING-MOVES') if intent4.status == StrategicIntentStatus.ACTIVE else 'retired'}")
    print(f"  supporting plans: {len(intent4.supporting_plan_ids)}")
    print(f"  sc_delta: {intent4.sc_delta_under_intent}")
    print(f"  retired: {len(retired)}, reason: {intent4.retire_reason}")
    assert len(retired) == 1
    assert intent4.retire_reason == "intent_too_broad"

    # ---- Test 5: revision candidate identified for slipping intent ----
    print()
    print("Test 5: revision candidate for slipping intent")
    print("-" * 72)
    m5 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    intent5 = StrategicIntentNode(
        id=new_id("intent"),
        head="Western alliance vs Germany.", body="",
        formed_at_phase="1901-FALL-MOVES",
        target_powers=["ENGLAND", "GERMANY"], target_provinces=[],
        horizon="1903-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.ACTIVE,
    )
    m5.strategic_intents[intent5.id] = intent5
    # 5 recent: 3 confirmed, 2 refuted → rate 0.4 (between trigger and retire)
    statuses = [PredictionStatus.CONFIRMED, PredictionStatus.CONFIRMED,
                PredictionStatus.CONFIRMED, PredictionStatus.REFUTED,
                PredictionStatus.REFUTED]
    for s in statuses:
        p = PredictionNode(
            id=new_id("pred"), about_power="ENGLAND",
            formed_at_phase="1902-FALL-MOVES",
            predicted_event_type="non_action",
            predicted_target="BEL", predicted_subject_power=None,
            prediction_window="1903-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.55,
            parent_intent_id=intent5.id,
            status=s,
        )
        m5.predictions[p.id] = p

    candidates = lifecycle_invite_intent_revisions(m5, "1903-FALL-MOVES")
    print(f"  predictions: 3 confirmed, 2 refuted (refute rate 0.40)")
    print(f"  revision candidates: {len(candidates)}")
    assert len(candidates) == 1

    # ---- Test 6: intent commitment lifecycle — window completion ----
    print()
    print("Test 6: intent commitment — window completion")
    print("-" * 72)
    m6 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    intent6 = StrategicIntentNode(
        id=new_id("intent"),
        head="Take Belgium.", body="",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        target_powers=["GERMANY"], target_provinces=["BEL"],
        horizon="1903-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.ACTIVE,
        sc_delta_under_intent=1,
    )
    m6.strategic_intents[intent6.id] = intent6
    commitment = start_intent_commitment(
        m6, intent6.id, started_at_phase="1902-SPRING-MOVES",
        window_phases=3,
    )
    print(f"  commitment opened, window={commitment.window_phases}")
    # Add 3 plans naming this intent (followed) over 3 phases
    for phase in ["1902-SPRING-MOVES", "1902-FALL-MOVES", "1903-SPRING-MOVES"]:
        plan = PlanNode(
            id=new_id("plan"), formed_at_phase=phase,
            head=f"Tactical plan at {phase}", body="",
            parent_intent_id=intent6.id,
            sc_delta_this_phase=0,
        )
        m6.plan_nodes[plan.id] = plan

    # Add 2 confirmed predictions
    for phase in ["1902-SPRING-MOVES", "1902-FALL-MOVES"]:
        p = PredictionNode(
            id=new_id("pred"), about_power="GERMANY",
            formed_at_phase=phase,
            predicted_event_type="non_action",
            predicted_target="BEL", predicted_subject_power=None,
            prediction_window=phase,
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.6,
            parent_intent_id=intent6.id,
            status=PredictionStatus.CONFIRMED,
        )
        m6.predictions[p.id] = p

    # Walk through three phase reviews
    for phase in ["1902-SPRING-MOVES", "1902-FALL-MOVES", "1903-SPRING-MOVES"]:
        changed = lifecycle_review_intent_commitments(m6, phase)
        if changed:
            print(f"    after {phase}: commitment status = {commitment.status}, "
                  f"end_reason = {commitment.end_reason}")
    print(f"  plans_followed: {commitment.plans_followed}, "
          f"diverged: {commitment.plans_diverged}")
    print(f"  predictions during: {commitment.predictions_confirmed_during} confirmed, "
          f"{commitment.predictions_refuted_during} refuted")
    assert commitment.status == "completed"
    assert commitment.end_reason == "window_reached_completed"
    assert intent6.status == StrategicIntentStatus.SUCCEEDED

    # ---- Test 7: intent commitment — abandoned for divergences ----
    print()
    print("Test 7: intent commitment — abandoned for too many divergences")
    print("-" * 72)
    m7 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    intent_a = StrategicIntentNode(
        id=new_id("intent"),
        head="Plan A: take Belgium.", body="",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        target_powers=["GERMANY"], target_provinces=["BEL"],
        horizon="1903-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.ACTIVE,
    )
    intent_b = StrategicIntentNode(
        id=new_id("intent"),
        head="Plan B: head south.", body="",
        formed_at_phase="1902-SPRING-MOVES",
        target_powers=["ITALY"], target_provinces=["MAR"],
        horizon="1904-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.ACTIVE,
    )
    m7.strategic_intents[intent_a.id] = intent_a
    m7.strategic_intents[intent_b.id] = intent_b
    commit = start_intent_commitment(m7, intent_a.id,
                                     started_at_phase="1902-SPRING-MOVES",
                                     window_phases=4)

    # Plans named DIFFERENT intent (intent_b) — divergent from commitment
    for phase in ["1902-SPRING-MOVES", "1902-FALL-MOVES", "1903-SPRING-MOVES"]:
        plan = PlanNode(
            id=new_id("plan"), formed_at_phase=phase,
            head=f"Diverging plan at {phase}", body="",
            parent_intent_id=intent_b.id,    # WRONG — under commitment to intent_a
            sc_delta_this_phase=0,
        )
        m7.plan_nodes[plan.id] = plan

    for phase in ["1902-SPRING-MOVES", "1902-FALL-MOVES", "1903-SPRING-MOVES"]:
        lifecycle_review_intent_commitments(m7, phase)
    print(f"  divergences: {commit.plans_diverged}")
    print(f"  commitment status: {commit.status}, end_reason: {commit.end_reason}")
    assert commit.status == "abandoned"
    assert commit.end_reason == "abandoned_too_many_divergences"

    # ---- Test 8: full lifecycle pass ----
    print()
    print("Test 8: full strategy lifecycle pass — review/promote/retire/revisions")
    print("-" * 72)
    m8 = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    proto = StrategicIntentNode(
        id=new_id("intent"),
        head="Solo Belgium by 1903.", body="",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        target_powers=["GERMANY"], target_provinces=["BEL"],
        horizon="1903-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.PROTO,
    )
    m8.strategic_intents[proto.id] = proto
    plan = PlanNode(
        id=new_id("plan"), formed_at_phase="1902-SPRING-MOVES",
        head="Move A PAR -> BUR.", body="",
        parent_intent_id=proto.id, sc_delta_this_phase=1,
    )
    m8.plan_nodes[plan.id] = plan
    plan2 = PlanNode(
        id=new_id("plan"), formed_at_phase="1902-FALL-MOVES",
        head="Hold pressure.", body="",
        parent_intent_id=proto.id, sc_delta_this_phase=0,
    )
    m8.plan_nodes[plan2.id] = plan2
    for _ in range(2):
        p = PredictionNode(
            id=new_id("pred"), about_power="GERMANY",
            formed_at_phase="1902-SPRING-MOVES",
            predicted_event_type="non_action",
            predicted_target="BEL", predicted_subject_power=None,
            prediction_window="1902-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.6,
            parent_intent_id=proto.id,
            status=PredictionStatus.CONFIRMED,
        )
        m8.predictions[p.id] = p

    summary = run_strategy_lifecycle(
        m8, current_phase="1902-FALL-MOVES",
        just_resolved_phase="1902-FALL-MOVES",
    )
    print(f"  plans_reviewed: {len(summary['plans_reviewed'])}")
    print(f"  commitments_changed: {len(summary['commitments_changed'])}")
    print(f"  promoted: {len(summary['promoted'])}")
    print(f"  retired: {len(summary['retired'])}")
    print(f"  revision_candidates: {len(summary['revision_candidates'])}")
    print(f"  proto intent now: {proto.status.value}")
    assert proto.status == StrategicIntentStatus.ACTIVE

    print()
    print("All strategy lifecycle tests passed.")
