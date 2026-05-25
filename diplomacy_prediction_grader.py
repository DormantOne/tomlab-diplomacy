"""
diplomacy_prediction_grader.py — deterministic check functions for predictions.

A prediction is a falsifiable claim the LLM made about another power's
future behavior. The grader walks open predictions whose window has closed
and marks each:
  CONFIRMED — the predicted event happened
  REFUTED   — the predicted event explicitly didn't happen
  PARTIAL   — multi-phase prediction with mixed evidence
  OPEN      — window still hasn't fully closed; check again next phase
  SUPERSEDED — set by the lifecycle (not the grader) when the source
               belief was retired before resolution

The grader operates on the same EventIndex used by commitment grading.
This is intentional — the indexes get built once per phase and reused.

DESIGN PRINCIPLES (mirror commitment graders):
  - No LLM calls. Pure functions over the event stream.
  - When in doubt, prefer leaving OPEN over forcing a verdict.
  - The grader returns evidence_ids; the lifecycle uses those to update
    the source belief's evidence_for / evidence_against lists.
"""

from __future__ import annotations

from typing import Optional

from diplomacy_kg_schema import (
    PredictionNode, PredictionStatus, PredictionWindowKind,
    PowerName, ProvinceCode, PhaseKey,
)
from diplomacy_graders import EventIndex
from diplomacy_fovea import _phase_index


# ============================================================================
# Per-event-type predicate graders
# ============================================================================

def _grade_move_to(p: PredictionNode, index: EventIndex,
                   window_phases: list[PhaseKey]
                   ) -> tuple[PredictionStatus, list[str]]:
    """CONFIRMED iff the predicted power moved into predicted_target
    in any phase within the window."""
    if not p.predicted_target:
        return PredictionStatus.OPEN, []   # malformed — leave open

    matching = []
    attempted = []
    for phase in window_phases:
        moves = index.moves_into(p.about_power, phase, p.predicted_target)
        for m in moves:
            if m.order_type == "MOVE":
                attempted.append(m)
                if m.result == "success":
                    matching.append(m)

    if matching:
        return PredictionStatus.CONFIRMED, [m.id for m in matching]
    if attempted:
        # They tried, didn't make it. Charitable read: PARTIAL — the intent
        # was there even though execution failed.
        return PredictionStatus.PARTIAL, [m.id for m in attempted]
    return PredictionStatus.REFUTED, []


def _grade_attack(p: PredictionNode, index: EventIndex,
                  window_phases: list[PhaseKey]
                  ) -> tuple[PredictionStatus, list[str]]:
    """CONFIRMED iff the predicted power attacked the predicted_subject_power.

    "Attack" here means: a successful move-into a province owned by the
    target power at start of phase, OR moving into a province where the
    target had a unit. We grade against phase_state for ownership and
    against move_events for attack-shape moves.
    """
    if not p.predicted_subject_power:
        return PredictionStatus.OPEN, []

    matching = []
    for phase in window_phases:
        ps = index.phase_state(phase)
        if ps is None:
            continue
        # Provinces owned by target at start of this phase
        target_scs = {prov for prov, owner in ps.sc_owner.items()
                      if owner == p.predicted_subject_power}
        # Provinces with target's units at start of this phase
        target_units = {loc for kind, loc in
                        ps.units_by_power.get(p.predicted_subject_power, [])}
        attacked_provs = target_scs | target_units

        for prov in attacked_provs:
            for m in index.moves_into(p.about_power, phase, prov):
                if m.order_type == "MOVE" and m.result == "success":
                    matching.append(m)

    if matching:
        return PredictionStatus.CONFIRMED, [m.id for m in matching]
    return PredictionStatus.REFUTED, []


def _grade_support(p: PredictionNode, index: EventIndex,
                   window_phases: list[PhaseKey]
                   ) -> tuple[PredictionStatus, list[str]]:
    """CONFIRMED iff the predicted power issued a SUPPORT order matching
    predicted_target (the supported province) in the window."""
    matching = []
    for phase in window_phases:
        for m in index.moves_by(p.about_power, phase):
            if m.order_type != "SUPPORT":
                continue
            if p.predicted_target and m.target != p.predicted_target:
                continue
            matching.append(m)

    if matching:
        return PredictionStatus.CONFIRMED, [m.id for m in matching]
    return PredictionStatus.REFUTED, []


def _grade_alliance(p: PredictionNode, index: EventIndex,
                    window_phases: list[PhaseKey]
                    ) -> tuple[PredictionStatus, list[str]]:
    """CONFIRMED iff predicted power did NOT attack predicted_subject_power
    AND issued at least one supportive action toward them.

    This is harder than non_aggression because alliance is bidirectional
    behavior. We check (i) no attacks (necessary), (ii) at least one
    SUPPORT order whose support_of references a unit owned by the
    predicted_subject_power (sufficient for confirmation).
    """
    if not p.predicted_subject_power:
        return PredictionStatus.OPEN, []

    aggressions = []
    supports_for_subject = []
    for phase in window_phases:
        ps = index.phase_state(phase)
        if ps is None:
            continue

        # Aggression check
        target_scs = {prov for prov, owner in ps.sc_owner.items()
                      if owner == p.predicted_subject_power}
        for prov in target_scs:
            for m in index.moves_into(p.about_power, phase, prov):
                if m.order_type == "MOVE" and m.result == "success":
                    aggressions.append(m)

        # Support-for check: speaker SUPPORTed a unit at a province where
        # the subject_power has a unit
        subject_unit_provs = {loc for kind, loc in
                              ps.units_by_power.get(p.predicted_subject_power, [])}
        for m in index.moves_by(p.about_power, phase):
            if m.order_type != "SUPPORT":
                continue
            # support_of looks like "A KIE -> BER" — extract origin
            if m.support_of:
                supported_origin = m.support_of.split()[1] if " " in m.support_of else None
                if supported_origin in subject_unit_provs:
                    supports_for_subject.append(m)

    if aggressions:
        return PredictionStatus.REFUTED, [m.id for m in aggressions]
    if supports_for_subject:
        return PredictionStatus.CONFIRMED, [m.id for m in supports_for_subject]
    # No aggressions and no positive supports = ambiguous; charitable read
    # is PARTIAL (no betrayal, no active alliance behavior either).
    return PredictionStatus.PARTIAL, []


def _grade_non_action(p: PredictionNode, index: EventIndex,
                      window_phases: list[PhaseKey]
                      ) -> tuple[PredictionStatus, list[str]]:
    """CONFIRMED iff the predicted power did NOT successfully move into
    predicted_target in the window. The mirror of move_to.

    REFUTED if they moved (successfully) where we predicted they wouldn't.
    """
    if not p.predicted_target:
        return PredictionStatus.OPEN, []

    successes = []
    for phase in window_phases:
        for m in index.moves_into(p.about_power, phase, p.predicted_target):
            if m.order_type == "MOVE" and m.result == "success":
                successes.append(m)
    if successes:
        return PredictionStatus.REFUTED, [m.id for m in successes]
    return PredictionStatus.CONFIRMED, []


def _grade_build_at(p: PredictionNode, index: EventIndex,
                    window_phases: list[PhaseKey]
                    ) -> tuple[PredictionStatus, list[str]]:
    """CONFIRMED iff predicted power built a unit at predicted_target
    in any adjustment phase within the window."""
    if not p.predicted_target:
        return PredictionStatus.OPEN, []

    matching = []
    for phase in window_phases:
        for adj in index.adjustments_by(p.about_power, phase):
            if adj.kind == "BUILD" and adj.location == p.predicted_target:
                matching.append(adj)

    if matching:
        return PredictionStatus.CONFIRMED, [a.id for a in matching]
    return PredictionStatus.REFUTED, []


def _grade_elimination_of(p: PredictionNode, index: EventIndex,
                          window_phases: list[PhaseKey]
                          ) -> tuple[PredictionStatus, list[str]]:
    """CONFIRMED iff predicted_subject_power was eliminated within the window."""
    if not p.predicted_subject_power:
        return PredictionStatus.OPEN, []

    if not window_phases:
        return PredictionStatus.OPEN, []

    last_phase = window_phases[-1]
    if index.eliminated_before(p.predicted_subject_power, last_phase):
        # Find the elimination event for evidence
        elim_evidence = []
        for phase in window_phases:
            for adj in index.adjustments_by(p.predicted_subject_power, phase):
                if adj.kind == "ELIMINATED":
                    elim_evidence.append(adj.id)
        return PredictionStatus.CONFIRMED, elim_evidence
    return PredictionStatus.REFUTED, []


# ============================================================================
# Window helpers
# ============================================================================

def _window_phases_for(p: PredictionNode,
                       all_resolved_phases: list[PhaseKey],
                       current_phase: PhaseKey,
                       ) -> list[PhaseKey]:
    """The list of resolved phases between p.formed_at_phase (exclusive)
    and min(p.prediction_window, current_phase) (inclusive).

    Used to decide whether a prediction is grade-able yet, and what window
    to grade it against.
    """
    formed_idx = _phase_index(p.formed_at_phase)
    target_idx = _phase_index(p.prediction_window)
    current_idx = _phase_index(current_phase)
    grading_end_idx = min(target_idx, current_idx)

    return [
        phase for phase in all_resolved_phases
        if formed_idx <= _phase_index(phase) <= grading_end_idx
    ]


def _window_fully_closed(p: PredictionNode, current_phase: PhaseKey) -> bool:
    """True iff the prediction's window has been fully traversed.

    Until the window is fully closed, multi-phase predictions stay OPEN
    even when interim evidence exists, because later phases could still
    resolve them differently.
    """
    return _phase_index(current_phase) >= _phase_index(p.prediction_window)


# ============================================================================
# Dispatch
# ============================================================================

_PREDICTION_GRADERS = {
    "move_to":         _grade_move_to,
    "attack":          _grade_attack,
    "support":         _grade_support,
    "alliance":        _grade_alliance,
    "non_action":      _grade_non_action,
    "build_at":        _grade_build_at,
    "elimination_of":  _grade_elimination_of,
}


def grade_prediction(
    p: PredictionNode,
    index: EventIndex,
    *,
    all_resolved_phases: list[PhaseKey],
    current_phase: PhaseKey,
) -> tuple[PredictionStatus, list[str]]:
    """Grade a single prediction. Returns (status, evidence_ids).

    The lifecycle is responsible for:
      - calling this only on OPEN predictions whose window has begun closing
      - updating the source belief's evidence after grading
      - leaving NEAR_TERM predictions with full window grade-able after
        their single resolved phase (window closes immediately)
    """
    if p.status != PredictionStatus.OPEN:
        return p.status, list(p.grading_evidence)

    window_phases = _window_phases_for(p, all_resolved_phases, current_phase)

    # If we have no phases in the window yet, we can't grade.
    if not window_phases:
        return PredictionStatus.OPEN, []

    grader = _PREDICTION_GRADERS.get(p.predicted_event_type)
    if grader is None:
        return PredictionStatus.OPEN, []

    status, evidence = grader(p, index, window_phases)

    # Multi-phase predictions stay OPEN until the window fully closes,
    # UNLESS we already have decisive evidence. Decisiveness depends on
    # the prediction shape:
    #
    #   "they will do X" predictions (move_to, build_at, attack, alliance,
    #     support, elimination_of):
    #      - one CONFIRMED occurrence is decisive (it happened, claim true)
    #      - REFUTED before window-close just means "not yet" — hold OPEN
    #
    #   "they will NOT do X" predictions (non_action):
    #      - one violation is decisive REFUTED
    #      - silence before window-close is "still on track" — hold OPEN
    #
    if not _window_fully_closed(p, current_phase):
        if p.predicted_event_type == "non_action":
            # Inverse predictions: REFUTED is decisive; CONFIRMED holds open
            if status == PredictionStatus.REFUTED:
                return status, evidence
            return PredictionStatus.OPEN, evidence
        else:
            # Positive predictions: CONFIRMED is decisive; REFUTED holds open
            if status == PredictionStatus.CONFIRMED:
                return status, evidence
            return PredictionStatus.OPEN, evidence

    return status, evidence


# ============================================================================
# Sanity check + worked examples
# ============================================================================

if __name__ == "__main__":
    import time as _t
    from diplomacy_kg_schema import (
        new_id, MoveEvent, AdjustmentEvent, PhaseState,
        PredictionNode, PredictionStatus, PredictionWindowKind,
    )

    # Build a small event index for 1902-SPRING-MOVES + 1902-FALL-MOVES
    index = EventIndex()
    s_phase = "1902-SPRING-MOVES"
    f_phase = "1902-FALL-MOVES"

    # Russia in spring: A WAR -> GAL (success), A MOS -> UKR (success)
    russia_s_moves = [
        MoveEvent(id=new_id("mv"), phase=s_phase, power="RUSSIA",
                  unit_kind="A", origin="WAR", order_type="MOVE",
                  target="GAL", support_of=None, result="success",
                  resolved_at=_t.time()),
        MoveEvent(id=new_id("mv"), phase=s_phase, power="RUSSIA",
                  unit_kind="A", origin="MOS", order_type="MOVE",
                  target="UKR", support_of=None, result="success",
                  resolved_at=_t.time()),
    ]
    # Russia in fall: A UKR -> SEV (success — attacks Turkey-owned SC)
    russia_f_moves = [
        MoveEvent(id=new_id("mv"), phase=f_phase, power="RUSSIA",
                  unit_kind="A", origin="UKR", order_type="MOVE",
                  target="RUM", support_of=None, result="success",
                  resolved_at=_t.time()),
    ]
    for m in russia_s_moves + russia_f_moves:
        index.add_move(m)

    # Phase states. SPRING: Russia owns its 4, Turkey owns RUM.
    s_state = PhaseState(
        id=new_id("ps"), phase=s_phase,
        sc_owner={"WAR": "RUSSIA", "MOS": "RUSSIA", "SEV": "RUSSIA",
                  "STP": "RUSSIA", "RUM": "TURKEY", "BUL": "TURKEY",
                  "ANK": "TURKEY", "CON": "TURKEY", "SMY": "TURKEY"},
        units_by_power={"RUSSIA": [("A", "WAR"), ("A", "MOS"),
                                   ("F", "SEV"), ("F", "STP")],
                        "TURKEY": [("A", "CON"), ("F", "ANK"), ("A", "SMY")]},
        eliminated=set(),
        captured_at=_t.time(),
    )
    # FALL: Russia took GAL and UKR, then attacks RUM
    f_state = PhaseState(
        id=new_id("ps"), phase=f_phase,
        sc_owner={"WAR": "RUSSIA", "MOS": "RUSSIA", "SEV": "RUSSIA",
                  "STP": "RUSSIA", "RUM": "TURKEY", "BUL": "TURKEY",
                  "ANK": "TURKEY", "CON": "TURKEY", "SMY": "TURKEY"},
        units_by_power={"RUSSIA": [("A", "GAL"), ("A", "UKR"),
                                   ("F", "SEV"), ("F", "STP")],
                        "TURKEY": [("A", "CON"), ("F", "ANK"), ("A", "SMY")]},
        eliminated=set(),
        captured_at=_t.time(),
    )
    index.add_phase_state(s_state)
    index.add_phase_state(f_state)

    print("=" * 72)
    print("PREDICTION GRADER TESTS")
    print("=" * 72)

    # Test 1: NEAR_TERM prediction "Russia will move to GAL by S1902" — CONFIRMED
    p1 = PredictionNode(
        id=new_id("pred"), about_power="RUSSIA",
        formed_at_phase="1902-SPRING-MOVES",          # formed AT the phase being graded
        predicted_event_type="move_to",
        predicted_target="GAL", predicted_subject_power=None,
        prediction_window=s_phase,
        window_kind=PredictionWindowKind.NEAR_TERM,
        confidence=0.7,
    )
    # Predictions formed AT a phase grade against that phase — relax the
    # strict-greater-than for the worked example. In real use, predictions
    # are formed DURING the phase (after orders, before resolution) and
    # graded against the same phase's resolution. Adjust formed_at_phase
    # to slightly earlier to satisfy the "formed_idx < phase_idx" check:
    p1.formed_at_phase = "1901-WINTER-ADJUSTMENTS"

    status, ev = grade_prediction(
        p1, index,
        all_resolved_phases=[s_phase, f_phase],
        current_phase=s_phase,
    )
    print(f"  Pred1: Russia move_to GAL by S1902")
    print(f"    -> {status.value}  (evidence: {len(ev)})")
    assert status == PredictionStatus.CONFIRMED

    # Test 2: prediction "Russia will move to BLA by S1902" — REFUTED
    p2 = PredictionNode(
        id=new_id("pred"), about_power="RUSSIA",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        predicted_event_type="move_to",
        predicted_target="BLA", predicted_subject_power=None,
        prediction_window=s_phase,
        window_kind=PredictionWindowKind.NEAR_TERM,
        confidence=0.5,
    )
    status, ev = grade_prediction(
        p2, index,
        all_resolved_phases=[s_phase, f_phase],
        current_phase=s_phase,
    )
    print(f"  Pred2: Russia move_to BLA by S1902")
    print(f"    -> {status.value}  (evidence: {len(ev)})")
    assert status == PredictionStatus.REFUTED

    # Test 3: LONG_HORIZON prediction "Russia will attack TURKEY by F1902"
    # — CONFIRMED (Russia attacked RUM in F1902)
    p3 = PredictionNode(
        id=new_id("pred"), about_power="RUSSIA",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        predicted_event_type="attack",
        predicted_target=None, predicted_subject_power="TURKEY",
        prediction_window=f_phase,
        window_kind=PredictionWindowKind.LONG_HORIZON,
        confidence=0.6,
    )
    status, ev = grade_prediction(
        p3, index,
        all_resolved_phases=[s_phase, f_phase],
        current_phase=f_phase,
    )
    print(f"  Pred3: Russia attack TURKEY by F1902")
    print(f"    -> {status.value}  (evidence: {len(ev)})")
    assert status == PredictionStatus.CONFIRMED

    # Test 4: same prediction but graded mid-window (only spring resolved)
    # — should still be OPEN because attack is non-monotonic confirmation
    status, ev = grade_prediction(
        p3, index,
        all_resolved_phases=[s_phase],
        current_phase=s_phase,
    )
    print(f"  Pred3 mid-window (only spring resolved):")
    print(f"    -> {status.value}  (evidence: {len(ev)})")
    assert status == PredictionStatus.OPEN, (
        f"Expected OPEN mid-window for attack-type, got {status.value}"
    )

    # Test 5: non_action prediction "Russia will not move into RUM by F1902"
    # — REFUTED (they did)
    p5 = PredictionNode(
        id=new_id("pred"), about_power="RUSSIA",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        predicted_event_type="non_action",
        predicted_target="RUM", predicted_subject_power=None,
        prediction_window=f_phase,
        window_kind=PredictionWindowKind.LONG_HORIZON,
        confidence=0.4,
    )
    status, ev = grade_prediction(
        p5, index,
        all_resolved_phases=[s_phase, f_phase],
        current_phase=f_phase,
    )
    print(f"  Pred5: Russia non_action on RUM by F1902")
    print(f"    -> {status.value}  (evidence: {len(ev)})")
    assert status == PredictionStatus.REFUTED

    # Test 6: prediction whose window hasn't started yet — OPEN
    p6 = PredictionNode(
        id=new_id("pred"), about_power="RUSSIA",
        formed_at_phase="1902-FALL-MOVES",
        predicted_event_type="move_to",
        predicted_target="STP", predicted_subject_power=None,
        prediction_window="1903-SPRING-MOVES",
        window_kind=PredictionWindowKind.LONG_HORIZON,
        confidence=0.5,
    )
    status, ev = grade_prediction(
        p6, index,
        all_resolved_phases=[s_phase, f_phase],
        current_phase=f_phase,
    )
    print(f"  Pred6: Russia move_to STP by S1903 (not yet resolved)")
    print(f"    -> {status.value}  (evidence: {len(ev)})")
    assert status == PredictionStatus.OPEN

    print()
    print("All prediction grader tests passed.")
