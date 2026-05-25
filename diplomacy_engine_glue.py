"""
diplomacy_engine_glue.py — bridge the existing engine to the new agent.

The legacy engine (diplomacy_engine/engine.py) returns a new GameState
after each adjudication. Our new mind layer needs structured event nodes
(MoveEvent, AdjustmentEvent, PhaseState).

This module does the translation:

  capture_move_events(orders, pre_state, post_state, adjudication_log)
    → list of MoveEvent (one per order, with success/bounced/dislodged)
  capture_adjustment_events(pre_state, post_state)
    → list of AdjustmentEvent (builds, disbands, eliminations)
  capture_phase_state(post_state)
    → single PhaseState snapshot

Plus a small helper, distribute_messages, that takes outgoing message
events from each agent's negotiation and routes them to recipients via
their intake_message() method. This is what the legacy session.py did
inline; pulling it out makes the session loop cleaner.
"""

from __future__ import annotations

import time as _t
from typing import Optional

from diplomacy_kg_schema import (
    MoveEvent, AdjustmentEvent, PhaseState, MessageEvent,
    PowerName, ProvinceCode, PhaseKey, new_id,
)
from diplomacy_agent_v2 import schema_phase_key, DiplomacyAgentV2


# ============================================================================
# Capture functions
# ============================================================================

def capture_move_events(
    *,
    orders: list,                    # list[Order] from the engine
    pre_state,                        # engine GameState before adjudication
    post_state,                       # engine GameState after adjudication
    adjudication_log: list[str],      # the engine's per-order log lines
) -> list[MoveEvent]:
    """Convert engine orders + pre/post state into MoveEvent nodes.

    The engine's adjudication_log contains lines like:
      "FRANCE: A PAR - BUR (success)"
      "FRANCE: A PAR - BUR (bounced)"
      "FRANCE: F MAO supports A LON - BRE (cut)"
      "GERMANY: A MUN dislodged"

    We use the log strings to determine the result; the orders themselves
    give us the structure. If the log is silent on an order, we infer
    success from the post_state (unit ended up at the target).
    """
    phase_key = schema_phase_key(pre_state.year, pre_state.season, pre_state.phase)
    events: list[MoveEvent] = []

    # Index post_state units by location for success-by-presence check
    post_units_by_loc = {(u.power, u.location): u for u in post_state.units}
    post_dislodged_locs = {(u.power, u.location) for u in post_state.dislodged}

    for o in orders:
        # Extract from the engine Order shape: power, unit_kind, location,
        # type, target, support_target_*, etc.
        # The engine's Order dataclass varies; we read defensively.
        result = _infer_result_from_log(o, adjudication_log, post_units_by_loc,
                                        post_dislodged_locs)
        target = getattr(o, "target", None)
        # Map engine order_type to schema order_type
        type_letter = getattr(o, "type", "H")
        order_type = {
            "H": "HOLD", "M": "MOVE", "S": "SUPPORT", "C": "CONVOY",
            "R": "RETREAT", "D": "DISBAND", "B": "BUILD",
        }.get(type_letter, type_letter.upper() if type_letter else "HOLD")

        support_of = None
        if order_type == "SUPPORT":
            sup_kind = getattr(o, "support_unit_kind", None) or "A"
            sup_loc = getattr(o, "support_unit_location", None) or ""
            sup_target = getattr(o, "support_target", None)
            if sup_target:
                support_of = f"{sup_kind} {sup_loc} -> {sup_target}"
            else:
                support_of = f"{sup_kind} {sup_loc}"

        events.append(MoveEvent(
            id=new_id("mv"),
            phase=phase_key,
            power=o.power,
            unit_kind=o.unit_kind,
            origin=o.location,
            order_type=order_type,
            target=target,
            support_of=support_of,
            result=result,
            resolved_at=_t.time(),
        ))
    return events


def _infer_result_from_log(
    order, log_lines: list[str],
    post_units_by_loc: dict, post_dislodged_locs: set,
) -> str:
    """Best-effort inference of (success/bounced/cut/dislodged/void) from
    the engine's free-form adjudication log.

    Conservative: if the log doesn't mention this order, assume success
    (the engine emits log lines mostly for things that didn't go cleanly).
    """
    sig_a = f"{order.unit_kind} {order.location}"
    sig_b = f"{order.unit_kind}{order.location}"
    target = getattr(order, "target", None)

    for line in log_lines:
        if (sig_a not in line and sig_b not in line):
            continue
        ln = line.lower()
        if "dislodg" in ln:
            return "dislodged"
        if "bounc" in ln:
            return "bounced"
        if " cut" in ln or "(cut)" in ln:
            return "cut"
        if "void" in ln or "fail" in ln:
            return "void"
        if "success" in ln:
            return "success"

    # Log silent: if it was a MOVE, check that the unit ended up at the target
    type_letter = getattr(order, "type", "H")
    if type_letter == "M":
        if target is not None and (order.power, target) in post_units_by_loc:
            return "success"
        if (order.power, order.location) in post_dislodged_locs:
            return "dislodged"
        return "bounced"
    # Hold/support/convoy: success unless dislodged
    if (order.power, order.location) in post_dislodged_locs:
        return "dislodged"
    return "success"


def capture_adjustment_events(
    *, pre_state, post_state,
) -> list[AdjustmentEvent]:
    """Detect builds/disbands/eliminations between pre and post state.

    Called after adjudicate_adjustments. We diff pre/post unit lists.
    """
    phase_key = schema_phase_key(post_state.year, post_state.season, post_state.phase)
    events: list[AdjustmentEvent] = []

    pre_units = {(u.power, u.kind, u.location) for u in pre_state.units}
    post_units = {(u.power, u.kind, u.location) for u in post_state.units}

    # Builds: in post but not in pre
    for power, kind, loc in (post_units - pre_units):
        events.append(AdjustmentEvent(
            id=new_id("adj"), phase=phase_key,
            power=power, kind="BUILD",
            unit_kind=kind, location=loc,
        ))
    # Disbands: in pre but not in post (but only counts during ADJUSTMENT phase
    # — disbands during retreat phase get marked too, equivalent semantics)
    for power, kind, loc in (pre_units - post_units):
        events.append(AdjustmentEvent(
            id=new_id("adj"), phase=phase_key,
            power=power, kind="DISBAND",
            unit_kind=kind, location=loc,
        ))

    # Eliminations: powers newly in post.eliminated
    pre_elim = set(pre_state.eliminated)
    post_elim = set(post_state.eliminated)
    for power in (post_elim - pre_elim):
        events.append(AdjustmentEvent(
            id=new_id("adj"), phase=phase_key,
            power=power, kind="ELIMINATED",
            unit_kind=None, location=None,
        ))

    return events


def capture_phase_state(state) -> PhaseState:
    """Snapshot the engine GameState as a schema PhaseState node."""
    phase_key = schema_phase_key(state.year, state.season, state.phase)
    units_by_power: dict[PowerName, list] = {}
    for u in state.units:
        units_by_power.setdefault(u.power, []).append((u.kind, u.location))
    return PhaseState(
        id=new_id("ps"), phase=phase_key,
        sc_owner=dict(state.sc_owner),
        units_by_power=units_by_power,
        eliminated=set(state.eliminated),
        captured_at=_t.time(),
    )


# ============================================================================
# Message routing
# ============================================================================

def distribute_messages(
    sent_messages: list[MessageEvent],
    agents: dict[PowerName, DiplomacyAgentV2],
) -> dict[PowerName, int]:
    """Route each message to its recipients via their intake_message().

    Returns a count of new commitments registered, per recipient power.
    Useful for telemetry.
    """
    counts = {p: 0 for p in agents}
    for msg in sent_messages:
        if msg.public:
            for power, agent in agents.items():
                if power == msg.sender:
                    continue
                counts[power] += agent.intake_message(msg)
        else:
            for r in msg.recipients:
                if r in agents:
                    counts[r] += agents[r].intake_message(msg)
    return counts


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    from diplomacy_engine import (
        initial_state, adjudicate_movement, parse_order, advance_phase,
        update_supply_centers,
    )

    print("=" * 72)
    print("ENGINE GLUE SANITY CHECK")
    print("=" * 72)

    state = initial_state()
    print(f"  Initial state: year={state.year}, season={state.season}, "
          f"phase={state.phase}")
    print(f"  Powers with units: {sorted(set(u.power for u in state.units))}")

    # A small set of orders for spring 1901
    raw_orders = [
        ("FRANCE",   "A PAR - BUR"),
        ("FRANCE",   "A MAR - SPA"),
        ("FRANCE",   "F BRE - MAO"),
        ("GERMANY",  "A MUN - RUH"),
        ("GERMANY",  "A BER - KIE"),
        ("GERMANY",  "F KIE - HOL"),
    ]
    parsed: list = []
    for power, line in raw_orders:
        o = parse_order(power, line)
        if o is not None:
            parsed.append(o)

    pre_state = state
    new_state, log = adjudicate_movement(state, parsed)
    print(f"  Adjudicated {len(parsed)} orders, log lines: {len(log)}")

    # Capture events
    move_events = capture_move_events(
        orders=parsed, pre_state=pre_state,
        post_state=new_state, adjudication_log=log,
    )
    print(f"  MoveEvents captured: {len(move_events)}")
    for m in move_events:
        print(f"    {m.power[:3]}: {m.unit_kind} {m.origin} {m.order_type}"
              f"{' -> ' + m.target if m.target else ''}  ({m.result})")

    ps = capture_phase_state(new_state)
    print(f"  PhaseState captured: phase={ps.phase}, "
          f"sc_owners={len(ps.sc_owner)}, units_by_power={list(ps.units_by_power.keys())}")

    print()
    print("Engine glue sanity check passed.")
