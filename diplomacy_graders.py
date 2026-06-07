"""
diplomacy_graders.py — deterministic check functions for commitments.

For each CommitmentType, write a grade function that takes the commitment
plus an event index and returns:
  (status, grading_evidence) — where status ∈ KEPT | BROKEN | IRRELEVANT
                                and grading_evidence is a list of event ids
                                that justify the verdict.

DESIGN PRINCIPLES:
  1. The graders never call the LLM. They are pure functions over the
     event stream.
  2. KEPT means "the commitment was honored; here are the events that prove it."
  3. BROKEN means "the commitment was violated; here are the events that prove it."
  4. IRRELEVANT means "the situation changed such that the commitment can't
     be tested" — e.g. the speaker was eliminated before the deadline.
     IRRELEVANT does NOT count as broken when computing credibility.
  5. When in doubt, prefer KEPT over BROKEN. The principle here is that
     a system that punishes ambiguous outcomes will turn every player into
     a paranoid; a system that rewards ambiguity will be exploited; a
     system that prefers the charitable reading is closer to how humans
     actually update credibility from gameplay. We err charitable.
  6. Graders are typed by CommitmentType — a separate function per kind.
"""

from __future__ import annotations

from typing import Optional

from diplomacy_kg_schema import (
    CommitmentNode, CommitmentStatus, CommitmentType,
    MoveEvent, AdjustmentEvent, PhaseState,
    PhaseKey, PowerName, ProvinceCode,
)


# ============================================================================
# Event index — the lookup structure graders use
# ============================================================================
# Built once per phase by the lifecycle, passed to graders. Indexes events
# by (power, phase) for cheap retrieval.

class EventIndex:
    """O(1)-lookup index over move/adjustment events and phase states.

    The lifecycle calls `index.add_*` as events resolve, then passes the
    same index to every grader for that phase. Graders never scan the
    full event list.
    """

    def __init__(self):
        # MoveEvents indexed by (power, phase).
        # value: list of MoveEvent (one power can have multiple units acting)
        self._moves_by_power_phase: dict[tuple[PowerName, PhaseKey],
                                         list[MoveEvent]] = {}
        # MoveEvents indexed by (power, phase, target_province) for
        # fast "did X attack Y?" queries.
        self._moves_by_target: dict[tuple[PowerName, PhaseKey, ProvinceCode],
                                    list[MoveEvent]] = {}
        # AdjustmentEvents (builds/disbands) by (power, phase)
        self._adj_by_power_phase: dict[tuple[PowerName, PhaseKey],
                                       list[AdjustmentEvent]] = {}
        # PhaseStates by phase
        self._states_by_phase: dict[PhaseKey, PhaseState] = {}
        # Eliminated powers, by phase they were eliminated
        self._eliminated_at: dict[PowerName, PhaseKey] = {}

    def add_move(self, m: MoveEvent) -> None:
        self._moves_by_power_phase.setdefault((m.power, m.phase), []).append(m)
        if m.target:
            self._moves_by_target.setdefault(
                (m.power, m.phase, m.target), []
            ).append(m)

    def add_adjustment(self, a: AdjustmentEvent) -> None:
        self._adj_by_power_phase.setdefault((a.power, a.phase), []).append(a)
        if a.kind == "ELIMINATED" and a.power not in self._eliminated_at:
            self._eliminated_at[a.power] = a.phase

    def add_phase_state(self, s: PhaseState) -> None:
        self._states_by_phase[s.phase] = s

    # --- lookups graders use ---
    def moves_by(self, power: PowerName, phase: PhaseKey) -> list[MoveEvent]:
        return list(self._moves_by_power_phase.get((power, phase), []))

    def moves_into(self, power: PowerName, phase: PhaseKey,
                   target: ProvinceCode) -> list[MoveEvent]:
        return list(self._moves_by_target.get((power, phase, target), []))

    def adjustments_by(self, power: PowerName,
                       phase: PhaseKey) -> list[AdjustmentEvent]:
        return list(self._adj_by_power_phase.get((power, phase), []))

    def phase_state(self, phase: PhaseKey) -> Optional[PhaseState]:
        return self._states_by_phase.get(phase)

    def eliminated_before(self, power: PowerName, phase: PhaseKey) -> bool:
        elim_phase = self._eliminated_at.get(power)
        if elim_phase is None:
            return False
        # use the phase-index helper from fovea module
        from diplomacy_fovea import _phase_index
        return _phase_index(elim_phase) < _phase_index(phase)


# ============================================================================
# Per-type graders
# ============================================================================
# Each takes (commitment, index) and returns (status, evidence_ids).

def _grade_move_to(c: CommitmentNode, index: EventIndex
                   ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff the speaker successfully moved subject_unit (or any unit at
    subject_province) to target_province in deadline_phase.

    BROKEN if the speaker had a unit at the origin and didn't try, or tried
    and bounced — bounced still counts as broken (you said you'd do it; you
    didn't). The exception: if the speaker was eliminated before the
    deadline, IRRELEVANT.
    """
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []

    # Find this speaker's moves at the deadline phase
    moves = index.moves_by(c.speaker, c.deadline_phase)
    if not moves:
        # Speaker had no units acting? Strange but possible. IRRELEVANT.
        return CommitmentStatus.IRRELEVANT, []

    # Look for a successful move-into-target by this speaker.
    matching = [
        m for m in moves
        if m.order_type == "MOVE"
        and m.target == c.target_province
        and m.result == "success"
        and (c.subject_unit is None or
             f"{m.unit_kind} {m.origin}" == c.subject_unit)
    ]
    if matching:
        return CommitmentStatus.KEPT, [m.id for m in matching]

    # No successful matching move. Check if they tried-but-bounced for evidence.
    attempted = [
        m for m in moves
        if m.order_type == "MOVE"
        and m.target == c.target_province
        and (c.subject_unit is None or
             f"{m.unit_kind} {m.origin}" == c.subject_unit)
    ]
    if attempted:
        # They tried, didn't make it. Still broken — the promise was about
        # the move succeeding, not about attempting.
        return CommitmentStatus.BROKEN, [m.id for m in attempted]

    # They didn't even try.
    return CommitmentStatus.BROKEN, []


def _grade_not_move_to(c: CommitmentNode, index: EventIndex
                       ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff no successful move-into target_province by speaker in any
    phase up through deadline_phase. (We check the whole window because
    'not_move_to BLA by F1902' means 'not in S1902 and not in F1902'.)

    For now the grader checks the deadline phase only; multi-phase
    windows can be handled by callers passing all phases between the
    commitment's formation and its deadline.
    """
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []

    moves_into_target = index.moves_into(
        c.speaker, c.deadline_phase, c.target_province or "",
    )
    successful_breaches = [
        m for m in moves_into_target
        if m.order_type == "MOVE" and m.result == "success"
    ]
    if successful_breaches:
        return CommitmentStatus.BROKEN, [m.id for m in successful_breaches]

    # Even an attempted (but bounced) move can be considered a violation —
    # you said you wouldn't, you tried. Charitable reading: count attempted
    # moves to the forbidden province as BROKEN evidence too, but flag them.
    attempts = [m for m in moves_into_target if m.order_type == "MOVE"]
    if attempts:
        return CommitmentStatus.BROKEN, [m.id for m in attempts]

    return CommitmentStatus.KEPT, []


def _grade_hold_at(c: CommitmentNode, index: EventIndex
                   ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff the unit at subject_province held (or no MOVE order issued
    away from it that succeeded). BROKEN if the unit moved away successfully.
    """
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []

    moves = index.moves_by(c.speaker, c.deadline_phase)
    relevant = [
        m for m in moves
        if m.origin == c.subject_province
        and (c.subject_unit is None or
             f"{m.unit_kind} {m.origin}" == c.subject_unit)
    ]
    if not relevant:
        # No order tracked for the unit at this province — IRRELEVANT
        # (perhaps the unit was no longer there).
        return CommitmentStatus.IRRELEVANT, []

    moved_away = [
        m for m in relevant
        if m.order_type == "MOVE" and m.result == "success"
    ]
    if moved_away:
        return CommitmentStatus.BROKEN, [m.id for m in moved_away]

    # Unit stayed (HOLD, SUPPORT-in-place, CONVOY, or attempted-but-failed move).
    return CommitmentStatus.KEPT, [m.id for m in relevant]


def _grade_support(c: CommitmentNode, index: EventIndex
                   ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff speaker issued a SUPPORT order matching the promised target.

    The check is more permissive than move/not_move because supports get cut
    by attackers, which isn't the speaker's fault. We grade on whether the
    SUPPORT order was issued, not whether it was uncut.
    """
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []

    moves = index.moves_by(c.speaker, c.deadline_phase)
    supports = [m for m in moves if m.order_type == "SUPPORT"]
    if not supports:
        return CommitmentStatus.BROKEN, []

    # Match by subject_unit (the supporter) and target_province.
    matching = []
    for s in supports:
        if c.subject_unit and f"{s.unit_kind} {s.origin}" != c.subject_unit:
            continue
        if c.target_province and s.target != c.target_province:
            continue
        matching.append(s)

    if matching:
        # Honored — even if cut. Note in evidence whether it was cut.
        return CommitmentStatus.KEPT, [m.id for m in matching]

    # Speaker issued SUPPORTs but none matched the promise.
    return CommitmentStatus.BROKEN, [m.id for m in supports]


def _grade_non_aggression(c: CommitmentNode, index: EventIndex,
                          phases_in_window: list[PhaseKey]
                          ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff speaker did not successfully move into any of counterparty's
    home centers (or owned SCs at start of the window) across all phases
    in the commitment window.

    This grader needs more than a single phase — non-aggression spans time.
    """
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []
    if c.counterparty is None:
        return CommitmentStatus.IRRELEVANT, []

    # What "counterparty's territory" means at the time the commitment formed:
    # the SCs that counterparty owned at the most recent phase_state preceding
    # window start. We approximate via the first phase_state in the window.
    target_scs: set[ProvinceCode] = set()
    for phase in phases_in_window:
        ps = index.phase_state(phase)
        if ps is None:
            continue
        for prov, owner in ps.sc_owner.items():
            if owner == c.counterparty:
                target_scs.add(prov)
        break  # first phase_state is enough for the snapshot

    if not target_scs:
        # No phase_state in window? Can't grade. IRRELEVANT.
        return CommitmentStatus.IRRELEVANT, []

    aggressions = []
    for phase in phases_in_window:
        for prov in target_scs:
            for m in index.moves_into(c.speaker, phase, prov):
                if m.order_type == "MOVE" and m.result == "success":
                    aggressions.append(m)
    if aggressions:
        return CommitmentStatus.BROKEN, [m.id for m in aggressions]

    return CommitmentStatus.KEPT, []


def _grade_demilitarize(c: CommitmentNode, index: EventIndex,
                        phases_in_window: list[PhaseKey]
                        ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff speaker had no unit in subject_province across the window."""
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []

    occupations = []
    for phase in phases_in_window:
        ps = index.phase_state(phase)
        if ps is None:
            continue
        units = ps.units_by_power.get(c.speaker, [])
        for kind, loc in units:
            if loc == c.subject_province:
                occupations.append(f"{phase}:{kind}{loc}")

    if occupations:
        return CommitmentStatus.BROKEN, occupations
    return CommitmentStatus.KEPT, []


def _grade_alliance_for(c: CommitmentNode, index: EventIndex,
                        phases_in_window: list[PhaseKey]
                        ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff speaker did not move successfully against counterparty's SCs
    across the window. (Same shape as non_aggression for now; alliance might
    add positive-action checks later — 'did you support an ally's move?' —
    but that's a more demanding grader.)
    """
    return _grade_non_aggression(c, index, phases_in_window)


def _grade_build(c: CommitmentNode, index: EventIndex
                 ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff a matching BUILD adjustment event exists at deadline_phase."""
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []

    adjs = index.adjustments_by(c.speaker, c.deadline_phase)
    matching = [
        a for a in adjs
        if a.kind == "BUILD"
        and a.location == c.subject_province
        and (c.subject_unit is None or
             (a.unit_kind and f"{a.unit_kind} {a.location}" == c.subject_unit))
    ]
    if matching:
        return CommitmentStatus.KEPT, [a.id for a in matching]
    return CommitmentStatus.BROKEN, []


def _grade_disband(c: CommitmentNode, index: EventIndex
                   ) -> tuple[CommitmentStatus, list[str]]:
    """KEPT iff a matching DISBAND adjustment event exists at deadline_phase."""
    if index.eliminated_before(c.speaker, c.deadline_phase):
        return CommitmentStatus.IRRELEVANT, []
    adjs = index.adjustments_by(c.speaker, c.deadline_phase)
    matching = [
        a for a in adjs
        if a.kind == "DISBAND"
        and (c.subject_unit is None or
             (a.unit_kind and f"{a.unit_kind} {a.location}" == c.subject_unit))
    ]
    if matching:
        return CommitmentStatus.KEPT, [a.id for a in matching]
    return CommitmentStatus.BROKEN, []


# ============================================================================
# Dispatch
# ============================================================================

def grade(c: CommitmentNode, index: EventIndex,
          phases_in_window: Optional[list[PhaseKey]] = None,
          ) -> tuple[CommitmentStatus, list[str]]:
    """Grade a commitment using the appropriate per-type grader.

    `phases_in_window` is required for multi-phase commitments
    (NON_AGGRESSION, ALLIANCE_FOR, DEMILITARIZE) and ignored otherwise.
    The lifecycle is responsible for assembling the window.
    """
    t = c.type
    if t == CommitmentType.MOVE_TO:
        return _grade_move_to(c, index)
    if t == CommitmentType.NOT_MOVE_TO:
        return _grade_not_move_to(c, index)
    if t == CommitmentType.HOLD_AT:
        return _grade_hold_at(c, index)
    if t == CommitmentType.SUPPORT:
        return _grade_support(c, index)
    if t == CommitmentType.BUILD:
        return _grade_build(c, index)
    if t == CommitmentType.DISBAND:
        return _grade_disband(c, index)
    # Multi-phase
    if phases_in_window is None:
        return CommitmentStatus.IRRELEVANT, []
    if t == CommitmentType.NON_AGGRESSION:
        return _grade_non_aggression(c, index, phases_in_window)
    if t == CommitmentType.DEMILITARIZE:
        return _grade_demilitarize(c, index, phases_in_window)
    if t == CommitmentType.ALLIANCE_FOR:
        return _grade_alliance_for(c, index, phases_in_window)
    # Unknown commitment type — IRRELEVANT
    return CommitmentStatus.IRRELEVANT, []


# ============================================================================
# Sanity check + worked examples
# ============================================================================

if __name__ == "__main__":
    import time as _t
    from diplomacy_kg_schema import (
        new_id, MoveEvent, AdjustmentEvent, PhaseState,
        CommitmentNode, CommitmentStatus, CommitmentType,
    )

    # Build an event index for 1902-SPRING-MOVES.
    index = EventIndex()
    phase = "1902-SPRING-MOVES"

    # Russia moves A WAR -> GAL (success), and F SEV holds, A MOS -> UKR (success)
    russia_moves = [
        MoveEvent(id=new_id("mv"), phase=phase, power="RUSSIA",
                  unit_kind="A", origin="WAR", order_type="MOVE",
                  target="GAL", support_of=None, result="success",
                  resolved_at=_t.time()),
        MoveEvent(id=new_id("mv"), phase=phase, power="RUSSIA",
                  unit_kind="F", origin="SEV", order_type="HOLD",
                  target=None, support_of=None, result="success",
                  resolved_at=_t.time()),
        MoveEvent(id=new_id("mv"), phase=phase, power="RUSSIA",
                  unit_kind="A", origin="MOS", order_type="MOVE",
                  target="UKR", support_of=None, result="success",
                  resolved_at=_t.time()),
    ]
    for m in russia_moves:
        index.add_move(m)

    # Phase state showing Russia's footprint and France's SCs (for
    # non_aggression test below)
    ps = PhaseState(
        id=new_id("ps"), phase=phase,
        sc_owner={"PAR": "FRANCE", "MAR": "FRANCE", "BRE": "FRANCE",
                  "WAR": "RUSSIA", "MOS": "RUSSIA", "SEV": "RUSSIA",
                  "STP": "RUSSIA"},
        units_by_power={"RUSSIA": [("A", "GAL"), ("F", "SEV"), ("A", "UKR")]},
        eliminated=set(),
        captured_at=_t.time(),
    )
    index.add_phase_state(ps)

    print("=" * 72)
    print("GRADER TESTS — Russia at 1902-SPRING-MOVES")
    print("=" * 72)

    # Test 1: Russia promised to move A WAR -> GAL. KEPT.
    c1 = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:1",
        speaker="RUSSIA", addressees=["AUSTRIA"],
        type=CommitmentType.MOVE_TO,
        subject_unit="A WAR", subject_province="WAR",
        target_province="GAL", counterparty=None,
        deadline_phase=phase, conditional_on=None,
    )
    status, ev = grade(c1, index)
    print(f"  move_to A WAR -> GAL by S1902: {status.value}  "
          f"(evidence: {len(ev)} events)")
    assert status == CommitmentStatus.KEPT, status

    # Test 2: Russia promised NOT to move into UKR. BROKEN (they did).
    c2 = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:2",
        speaker="RUSSIA", addressees=["TURKEY"],
        type=CommitmentType.NOT_MOVE_TO,
        subject_unit=None, subject_province=None,
        target_province="UKR", counterparty=None,
        deadline_phase=phase, conditional_on=None,
    )
    status, ev = grade(c2, index)
    print(f"  not_move_to UKR by S1902: {status.value}  "
          f"(evidence: {len(ev)} events)")
    assert status == CommitmentStatus.BROKEN, status

    # Test 3: Russia promised NOT to move into BLA. KEPT (no such move).
    c3 = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:3",
        speaker="RUSSIA", addressees=["TURKEY"],
        type=CommitmentType.NOT_MOVE_TO,
        subject_unit=None, subject_province=None,
        target_province="BLA", counterparty=None,
        deadline_phase=phase, conditional_on=None,
    )
    status, ev = grade(c3, index)
    print(f"  not_move_to BLA by S1902: {status.value}  "
          f"(evidence: {len(ev)} events)")
    assert status == CommitmentStatus.KEPT, status

    # Test 4: Russia promised F SEV holds. KEPT.
    c4 = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:4",
        speaker="RUSSIA", addressees=["TURKEY"],
        type=CommitmentType.HOLD_AT,
        subject_unit="F SEV", subject_province="SEV",
        target_province=None, counterparty=None,
        deadline_phase=phase, conditional_on=None,
    )
    status, ev = grade(c4, index)
    print(f"  hold_at F SEV by S1902: {status.value}  "
          f"(evidence: {len(ev)} events)")
    assert status == CommitmentStatus.KEPT, status

    # Test 5: Russia promised non-aggression with FRANCE through 1902. KEPT.
    c5 = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:5",
        speaker="RUSSIA", addressees=["FRANCE"],
        type=CommitmentType.NON_AGGRESSION,
        subject_unit=None, subject_province=None,
        target_province=None, counterparty="FRANCE",
        deadline_phase="1902-WINTER-ADJUSTMENTS", conditional_on=None,
    )
    status, ev = grade(c5, index, phases_in_window=[phase])
    print(f"  non_aggression with FRANCE through 1902: {status.value}  "
          f"(evidence: {len(ev)} events)")
    assert status == CommitmentStatus.KEPT, status

    # Test 6: Russia promised to move A MOS -> SEV (didn't — moved to UKR). BROKEN.
    c6 = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg:6",
        speaker="RUSSIA", addressees=["TURKEY"],
        type=CommitmentType.MOVE_TO,
        subject_unit="A MOS", subject_province="MOS",
        target_province="SEV", counterparty=None,
        deadline_phase=phase, conditional_on=None,
    )
    status, ev = grade(c6, index)
    print(f"  move_to A MOS -> SEV by S1902: {status.value}  "
          f"(broken because they moved A MOS -> UKR instead)")
    assert status == CommitmentStatus.BROKEN, status

    print()
    print("All grader tests passed.")
