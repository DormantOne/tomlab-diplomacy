"""
diplomacy_fovea.py — fovea selection for the diplomacy KG.

Given an AgentMind and a call context (negotiation or orders, with addressees),
produce a TurnFovea: the narrow slice of mind state that goes into the prompt.

Design constraints (from magic_kg_v4_6 + this design's commitments):
  - Broad storage, narrow display.
  - At most ONE belief per addressee.
  - At most ONE open commitment in / one self-commitment out per addressee.
  - Empty slots are legitimate — silent default is valid.
  - Scoring is structural (typed lookups) before it is similarity-based.
    This bounds cost to O(beliefs-about-target-power), not O(all-beliefs).
  - No LLM calls. The fovea is a pure function over the mind state.

Public entry point: build_fovea(...).

The selection logic in this module is intentionally separate from rendering.
build_fovea() returns a TurnFovea data object; render() is a method on
TurnFovea (defined in the schema). This separation lets us inspect what
*would* be shown without paying for the prompt assembly, and lets us swap
rendering styles without touching selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable

# Type imports from the schema. In a real package these'd be a single import.
from diplomacy_kg_schema import (
    AgentMind, BeliefNode, BeliefStatus, BeliefType,
    CommitmentNode, SelfCommitmentNode, CommitmentStatus,
    StrategicIntentNode, StrategicIntentStatus,
    FoveaSlot, TurnFovea,
    PowerName, ProvinceCode, PhaseKey,
)


# ============================================================================
# Call context — what the fovea is being built for
# ============================================================================
# The fovea scopes itself to a call. Two call kinds matter:
#   - NEGOTIATION: writing a message to specific addressees. Fovea slice
#     is per-addressee, very narrow — what I currently believe about THIS
#     person, what we owe each other, what I can't lie about that they'd see.
#   - ORDERS: deciding orders. Fovea slice is per-power-whose-units-touch-me,
#     because those are the powers whose intentions matter for my move.

@dataclass
class CallContext:
    """Why we're building a fovea right now."""
    kind: str                                       # "negotiation" | "orders"
    phase: PhaseKey
    addressees: list[PowerName]                     # one for nego per recipient,
                                                    # multiple for orders (powers I'm planning against)
    current_phase_for_relevance: PhaseKey           # usually = phase, but during reflection
                                                    # may differ from when fovea is consumed


# ============================================================================
# Scoring weights — tunable, all in one place
# ============================================================================
# Hand-rolled, like magic_go. Tuned by intuition, not learned. Every score
# additive so we can attribute "why was THIS belief selected over THAT one"
# at debug time.

# --- Belief selection ---
W_BELIEF_TYPE_FOR_NEGOTIATION = {
    BeliefType.CREDIBILITY:      1.0,
    BeliefType.RELATIONSHIP:     0.85,
    BeliefType.DISPOSITION:      0.6,
    BeliefType.RISK_ASSESSMENT:  0.7,
    BeliefType.TACTICAL_PATTERN: 0.4,
}
W_BELIEF_TYPE_FOR_ORDERS = {
    BeliefType.RISK_ASSESSMENT:  1.0,
    BeliefType.TACTICAL_PATTERN: 0.9,
    BeliefType.RELATIONSHIP:     0.7,
    BeliefType.CREDIBILITY:      0.6,
    BeliefType.DISPOSITION:      0.5,
}
W_BELIEF_HP_MAX_BOOST    = 0.25         # full hp adds up to 0.25
W_BELIEF_CRITIC_BOOST    = 0.30         # critic 1.0 adds 0.30, 0.0 subtracts 0.0
W_BELIEF_RECENCY_PHASES  = 0.40         # most-recent +0.40, decays each phase since last_updated
W_BELIEF_RECENCY_DECAY   = 0.10         # per-phase falloff
W_BELIEF_INTENT_ALIGNED  = 0.35         # if belief.about_power overlaps active intent's targets
W_BELIEF_FOVEATED_RECENT = -0.15        # selection_cap: belief picked recently scores lower
                                        # (to surface new perspectives — Patch 7 of magic_go)

# --- Commitment selection ---
W_COMMITMENT_DEADLINE_NEAR = 1.0        # full credit if deadline IS this phase
W_COMMITMENT_DEADLINE_FALLOFF = 0.20    # per-phase falloff for commitments due later
W_COMMITMENT_CONDITIONAL_PENALTY = -0.20  # conditional commitments rank slightly lower
                                          # (less actionable because their grading is fuzzier)

# --- Per-call caps ---
PER_TURN_FOVEA_BELIEF_CAP = 1                # one belief shown per game per addressee per call
                                              # (not per game-globally — that's the magic_go cap)
                                              # rather: per call, per slot

# Minimum belief score to include at all — below this, we show nothing
# (silent-default principle from magic_go Patch 3)
MIN_BELIEF_SCORE_TO_SHOW = 0.35


# ============================================================================
# Scoring functions
# ============================================================================

def _score_belief(
    belief: BeliefNode,
    target_power: PowerName,
    context: CallContext,
    active_intent: Optional[StrategicIntentNode],
) -> float:
    """Score a belief for inclusion in the fovea slot for `target_power`.

    The score breakdown is deliberately additive so that a debug log of
    "why did this belief win" reads cleanly.
    """
    if belief.about_power != target_power:
        return float("-inf")    # never select cross-power
    if belief.status != BeliefStatus.ACTIVE:
        return float("-inf")    # protos and revisions live elsewhere

    weights = (W_BELIEF_TYPE_FOR_NEGOTIATION
               if context.kind == "negotiation"
               else W_BELIEF_TYPE_FOR_ORDERS)
    score = weights.get(belief.belief_type, 0.5)

    # HP and critic
    score += min(belief.hp, 1.0) * W_BELIEF_HP_MAX_BOOST
    score += belief.critic_score * W_BELIEF_CRITIC_BOOST

    # Recency
    phases_since = _phases_between(belief.last_updated_phase, context.current_phase_for_relevance)
    if phases_since == 0:
        score += W_BELIEF_RECENCY_PHASES
    else:
        score += max(0.0, W_BELIEF_RECENCY_PHASES - W_BELIEF_RECENCY_DECAY * phases_since)

    # Intent alignment
    if active_intent and target_power in active_intent.target_powers:
        score += W_BELIEF_INTENT_ALIGNED

    # Anti-anchor: a belief that was foveated very recently scores slightly
    # lower so we don't keep showing the same one. Cheap version: based on
    # times_foveated relative to peers. For now a flat penalty if hit > 5x.
    if belief.times_foveated > 5:
        score += W_BELIEF_FOVEATED_RECENT

    return score


def _score_commitment(
    commitment: CommitmentNode,
    context: CallContext,
) -> float:
    """Score an open commitment for the fovea. Soonest-due wins."""
    if commitment.status != CommitmentStatus.PENDING:
        return float("-inf")

    phases_until = _phases_between(context.current_phase_for_relevance, commitment.deadline_phase)
    if phases_until < 0:
        return float("-inf")    # already overdue but not yet graded — skip
    if phases_until == 0:
        score = W_COMMITMENT_DEADLINE_NEAR
    else:
        score = max(0.0, W_COMMITMENT_DEADLINE_NEAR - W_COMMITMENT_DEADLINE_FALLOFF * phases_until)

    if commitment.conditional_on:
        score += W_COMMITMENT_CONDITIONAL_PENALTY

    return score


# ============================================================================
# Helpers
# ============================================================================

def _phases_between(earlier: PhaseKey, later: PhaseKey) -> int:
    """Count phases from `earlier` to `later`. Returns negative if earlier
    is in fact after later. The PhaseKey format is YYYY-SEASON-PHASE.

    Diplomacy ordering per game-year:
      SPRING-MOVES -> SPRING-RETREATS -> FALL-MOVES -> FALL-RETREATS
      -> WINTER-ADJUSTMENTS

    Five phases per year. So phase index = year * 5 + season_phase_offset.
    """
    return _phase_index(later) - _phase_index(earlier)


_PHASE_OFFSET = {
    ("SPRING",  "MOVES"):       0,
    ("SPRING",  "RETREATS"):    1,
    ("FALL",    "MOVES"):       2,
    ("FALL",    "RETREATS"):    3,
    ("WINTER",  "ADJUSTMENTS"): 4,
}


def _phase_index(pk: PhaseKey) -> int:
    """Convert "1902-FALL-MOVES" → integer index for ordinal arithmetic."""
    parts = pk.split("-")
    if len(parts) != 3:
        # malformed — treat as far-future to avoid surprising selections
        return 10_000_000
    year, season, phase = parts
    try:
        y = int(year)
    except ValueError:
        return 10_000_000
    offset = _PHASE_OFFSET.get((season, phase), 0)
    return y * 5 + offset


def _phases_until(now_phase: PhaseKey, target_phase: PhaseKey) -> int:
    """Convenience: same as _phases_between(now, target)."""
    return _phases_between(now_phase, target_phase)


# ============================================================================
# Per-slot selection
# ============================================================================

def _select_belief_for_power(
    mind: AgentMind,
    target_power: PowerName,
    context: CallContext,
    active_intent: Optional[StrategicIntentNode],
) -> Optional[BeliefNode]:
    """Pick at most one active belief about `target_power`.

    Returns None if no belief clears MIN_BELIEF_SCORE_TO_SHOW. Returning None
    is a legitimate outcome — we'd rather show nothing than show noise.
    """
    candidates: list[tuple[float, BeliefNode]] = []
    for belief in mind.beliefs.values():
        if belief.about_power != target_power:
            continue
        if belief.status != BeliefStatus.ACTIVE:
            continue
        score = _score_belief(belief, target_power, context, active_intent)
        if score >= MIN_BELIEF_SCORE_TO_SHOW:
            candidates.append((score, belief))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _select_open_commitment_from(
    mind: AgentMind,
    speaker: PowerName,
    context: CallContext,
) -> Optional[CommitmentNode]:
    """Pick the soonest-due pending commitment that `speaker` made to me.

    Returns None if there is no pending, ungraded commitment from this power.
    """
    candidates: list[tuple[float, CommitmentNode]] = []
    for c in mind.incoming_commitments.values():
        if c.speaker != speaker:
            continue
        if mind.owner_power not in c.addressees and not c.addressees:
            # addressees=[] means public commitment, which we still count
            pass
        elif mind.owner_power not in c.addressees:
            continue
        score = _score_commitment(c, context)
        if score > float("-inf"):
            candidates.append((score, c))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _select_open_commitment_to(
    mind: AgentMind,
    target_power: PowerName,
    context: CallContext,
) -> Optional[SelfCommitmentNode]:
    """Pick the soonest-due pending self-commitment we made to `target_power`."""
    candidates: list[tuple[float, SelfCommitmentNode]] = []
    for sc in mind.self_commitments.values():
        if sc.target_power != target_power:
            continue
        score = _score_commitment(sc, context)
        if score > float("-inf"):
            candidates.append((score, sc))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _select_active_intent(
    mind: AgentMind,
    context: CallContext,
) -> Optional[StrategicIntentNode]:
    """Pick the strategic intent currently under commitment (or, if none,
    the highest-scoring ACTIVE intent). Only one intent is ever foveated."""
    # First, prefer an actively-committed intent.
    for ic in mind.intent_commitments.values():
        if ic.status == "active":
            intent = mind.strategic_intents.get(ic.intent_id)
            if intent and intent.status == StrategicIntentStatus.ACTIVE:
                return intent

    # Fallback: highest hp + critic ACTIVE intent, if any.
    actives = [
        i for i in mind.strategic_intents.values()
        if i.status == StrategicIntentStatus.ACTIVE
    ]
    if not actives:
        return None
    actives.sort(key=lambda i: (i.hp + i.critic_score, i.success), reverse=True)
    return actives[0]


def _credibility_summary_line(
    mind: AgentMind,
    addressees: Iterable[PowerName],
) -> Optional[str]:
    """Single-line summary of kept/broken commitments per addressee.

    Returns None if no addressee has any graded commitment yet — early game,
    we'd rather show nothing than misleadingly show "RUS: 0/0 kept".
    """
    bits = []
    for power in addressees:
        kept = 0; broken = 0
        for c in mind.incoming_commitments.values():
            if c.speaker != power:
                continue
            if c.status == CommitmentStatus.KEPT:
                kept += 1
            elif c.status == CommitmentStatus.BROKEN:
                broken += 1
        total = kept + broken
        if total > 0:
            bits.append(f"{power[:3]}:{kept}/{total}")
    return ("kept-record: " + " ".join(bits)) if bits else None


# ============================================================================
# The public entry point
# ============================================================================

def build_fovea(mind: AgentMind, context: CallContext) -> TurnFovea:
    """Construct the narrow slice of `mind` that should go into this call's
    prompt.

    Returns a TurnFovea data object whose .render() method produces the
    final prompt text. Empty slots are simply omitted from the rendering.

    DESIGN NOTE: this function is pure. It does not mutate `mind`. All
    side effects (incrementing times_foveated on selected beliefs, etc.)
    are handled by the caller AFTER the LLM call succeeds, so a re-run of
    the same fovea against the same mind state is reproducible.
    """
    slots: list[FoveaSlot] = []

    # 1. The currently committed strategic intent (if any).
    active_intent = _select_active_intent(mind, context)
    intent_head = active_intent.head if active_intent else None

    # 2. Per-addressee slots.
    for power in context.addressees:
        if power == mind.owner_power:
            continue    # don't foveate ourselves into our own prompt

        # Belief about this power
        belief = _select_belief_for_power(mind, power, context, active_intent)
        if belief is not None:
            slots.append(FoveaSlot(
                role="belief_about",
                target_power=power,
                head_text=belief.head,
                node_id=belief.id,
                inspectable=True,
            ))

        # Open commitment from this power to us
        cmt_in = _select_open_commitment_from(mind, power, context)
        if cmt_in is not None:
            slots.append(FoveaSlot(
                role="open_commitment_from",
                target_power=power,
                head_text=_format_commitment_head(cmt_in),
                node_id=cmt_in.id,
                inspectable=True,
            ))

        # Open self-commitment from us to this power
        cmt_out = _select_open_commitment_to(mind, power, context)
        if cmt_out is not None:
            slots.append(FoveaSlot(
                role="open_commitment_to",
                target_power=power,
                head_text=_format_commitment_head(cmt_out),
                node_id=cmt_out.id,
                inspectable=True,
            ))

    # 3. Credibility summary across addressees (one compact line)
    cred_line = _credibility_summary_line(mind, context.addressees)
    if cred_line:
        slots.append(FoveaSlot(
            role="credibility_summary",
            target_power=None,
            head_text=cred_line,
            node_id="<derived>",
            inspectable=False,
        ))

    return TurnFovea(
        phase=context.phase,
        addressees=list(context.addressees),
        slots=slots,
        strategic_intent_head=intent_head,
        character_brief_text=(
            mind.character_brief.text if mind.character_brief else ""
        ),
    )


def _format_commitment_head(c: CommitmentNode) -> str:
    """One-line rendering of a commitment for the fovea.

    Shape: "<type> <subject> by <deadline>[ if <conditional>]"
    """
    bits = [c.type.value]
    subject = c.subject_unit or c.subject_province or c.counterparty or ""
    if subject:
        bits.append(subject)
    if c.target_province:
        bits.append(f"-> {c.target_province}")
    bits.append(f"by {c.deadline_phase}")
    if c.conditional_on:
        bits.append(f"if {c.conditional_on}")
    return " ".join(bits)


# ============================================================================
# Side-effect helpers (called AFTER the LLM call resolves)
# ============================================================================

def mark_fovea_used(mind: AgentMind, fovea: TurnFovea) -> None:
    """Increment times_foveated counters on selected beliefs.

    Call this only AFTER the LLM call that consumed `fovea` returns
    successfully. Calling it inside build_fovea() would corrupt the
    "fovea is a pure function" property and make replays nondeterministic.
    """
    for slot in fovea.slots:
        if slot.role == "belief_about" and slot.node_id in mind.beliefs:
            mind.beliefs[slot.node_id].times_foveated += 1


def mark_belief_inspected(mind: AgentMind, belief_id: str) -> None:
    """Increment times_inspected when the LLM calls inspect() on a belief."""
    if belief_id in mind.beliefs:
        mind.beliefs[belief_id].times_inspected += 1


# ============================================================================
# Sanity check + worked example
# ============================================================================

if __name__ == "__main__":
    import time as _time
    from diplomacy_kg_schema import (
        new_id, CharacterBrief, BeliefNode, BeliefType, BeliefStatus,
        CommitmentNode, SelfCommitmentNode, CommitmentType, CommitmentStatus,
        StrategicIntentNode, StrategicIntentStatus,
    )

    # Stand up a worked example: AUSTRIA, mid-game, turn 1902-SPRING-MOVES,
    # negotiating with FRANCE.

    mind = AgentMind(owner_power="AUSTRIA", archetype="MARSHAL_VEIL")
    mind.character_brief = CharacterBrief(
        id=new_id("brief"), archetype="MARSHAL_VEIL",
        text=("I am Marshal Veil. I keep my word when watched, and I expect the same. "
              "I do not improvise. Every promise I make is one I have measured."),
        generated_at=_time.time(),
    )

    # A few beliefs about FRANCE
    b1 = BeliefNode(
        id=new_id("belief"), about_power="FRANCE", belief_type=BeliefType.RELATIONSHIP,
        head="France is currently coordinating with England against Germany.",
        body="(full evidence trail elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1902-SPRING-MOVES",
        hp=1.2, critic_score=0.7, status=BeliefStatus.ACTIVE,
    )
    b2 = BeliefNode(
        id=new_id("belief"), about_power="FRANCE", belief_type=BeliefType.CREDIBILITY,
        head="France has kept 4/5 of the commitments they've made me.",
        body="(full evidence trail elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1902-SPRING-MOVES",
        hp=1.0, critic_score=0.9, status=BeliefStatus.ACTIVE,
        persists_across_games=True,
    )
    b3 = BeliefNode(
        id=new_id("belief"), about_power="FRANCE", belief_type=BeliefType.DISPOSITION,
        head="France favors slow consolidation over aggressive expansion.",
        body="(full evidence trail elided)",
        formed_at_phase="1901-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1901-FALL-MOVES",        # older
        hp=0.9, critic_score=0.6, status=BeliefStatus.ACTIVE,
        persists_across_games=True,
    )
    # And a stale proto-belief that should NOT be selected
    b_proto = BeliefNode(
        id=new_id("belief"), about_power="FRANCE", belief_type=BeliefType.RISK_ASSESSMENT,
        head="France will move on Belgium by 1902.",
        body="(elided)",
        formed_at_phase="1902-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1902-SPRING-MOVES",
        hp=1.0, critic_score=0.5, status=BeliefStatus.PROTO,    # not active
    )
    for b in (b1, b2, b3, b_proto):
        mind.beliefs[b.id] = b

    # An incoming commitment from France (pending, due this phase)
    cmt_in = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:fra1",
        speaker="FRANCE", addressees=["AUSTRIA"],
        type=CommitmentType.NOT_MOVE_TO,
        subject_unit=None, subject_province="MUN", target_province=None,
        counterparty=None, deadline_phase="1902-SPRING-MOVES",
        conditional_on=None,
        status=CommitmentStatus.PENDING,
        raw_commitspeak_line="not_move_to: MUN by 1902-SPRING-MOVES",
    )
    mind.incoming_commitments[cmt_in.id] = cmt_in

    # A self-commitment we made to France
    cmt_out = SelfCommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:aus1",
        speaker="AUSTRIA", addressees=["FRANCE"],
        type=CommitmentType.SUPPORT,
        subject_unit="A VIE", subject_province="VIE",
        target_province="TYR", counterparty=None,
        deadline_phase="1902-SPRING-MOVES",
        conditional_on=None,
        status=CommitmentStatus.PENDING,
        raw_commitspeak_line="support: A VIE S A MAR -> TYR by 1902-SPRING-MOVES",
        target_power="FRANCE",
    )
    mind.self_commitments[cmt_out.id] = cmt_out

    # Mark a few past commitments graded so credibility summary has content
    for i in range(4):
        c = CommitmentNode(
            id=new_id("cmt"), source_msg_id=f"msg:past{i}",
            speaker="FRANCE", addressees=["AUSTRIA"],
            type=CommitmentType.NOT_MOVE_TO,
            subject_unit=None, subject_province="MUN", target_province=None,
            counterparty=None, deadline_phase="1901-FALL-MOVES",
            conditional_on=None,
            status=CommitmentStatus.KEPT,
        )
        mind.incoming_commitments[c.id] = c
    c_broken = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:past_b",
        speaker="FRANCE", addressees=["AUSTRIA"],
        type=CommitmentType.NOT_MOVE_TO,
        subject_unit=None, subject_province="BUR", target_province=None,
        counterparty=None, deadline_phase="1901-FALL-MOVES",
        conditional_on=None,
        status=CommitmentStatus.BROKEN,
    )
    mind.incoming_commitments[c_broken.id] = c_broken

    # An active strategic intent that involves FRANCE
    intent = StrategicIntentNode(
        id=new_id("intent"),
        head="Hold the western front via alliance with France through 1903.",
        body="(full plan elided)",
        formed_at_phase="1901-FALL-MOVES",
        target_powers=["FRANCE"],
        target_provinces=["MUN", "TYR"],
        horizon="1903-WINTER-ADJUSTMENTS",
        hp=1.4, critic_score=0.7,
        status=StrategicIntentStatus.ACTIVE,
    )
    mind.strategic_intents[intent.id] = intent

    # ---- Build the fovea for negotiation with FRANCE ----
    ctx = CallContext(
        kind="negotiation",
        phase="1902-SPRING-MOVES",
        addressees=["FRANCE"],
        current_phase_for_relevance="1902-SPRING-MOVES",
    )
    fovea = build_fovea(mind, ctx)

    print("=" * 72)
    print("FOVEA RENDERING (for AUSTRIA negotiating with FRANCE, 1902-SPRING-MOVES)")
    print("=" * 72)
    print(fovea.render())
    print()
    print(f"Slots: {len(fovea.slots)}")
    for slot in fovea.slots:
        print(f"  [{slot.role}] target={slot.target_power} "
              f"id={slot.node_id} :: {slot.head_text[:80]}")
