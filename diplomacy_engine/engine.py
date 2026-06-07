"""
Diplomacy game engine: state representation and order adjudication.

This is a simplified-but-functional adjudicator. It correctly handles:
- Holds, moves, supports, basic single-fleet convoys
- Support cuts (an attacked supporter loses its support)
- Bouncing (equal strength = no one moves)
- Self-bounce / self-dislodgement prevention
- Standoffs leaving provinces vacant
- Retreats and disbands
- Adjustment phase (build/disband based on supply-center count)

It does NOT fully resolve every DATC paradox case (e.g. circular convoy
paradoxes, multi-fleet convoy chains with cuts inside the chain). The goal
is a believable Diplomacy experience for the LLM agents, not a perfect
tournament adjudicator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from .map_data import (
    ADJ,
    ALL_SUPPLY_CENTERS,
    HOME_CENTERS,
    NEUTRAL_CENTERS,
    POWERS,
    PROVINCES,
    STARTING_UNITS,
    can_occupy,
    is_adjacent,
)


# ----------------------------- Data classes --------------------------------- #


@dataclass
class Unit:
    kind: str       # "A" or "F"
    power: str
    location: str

    def __repr__(self) -> str:
        return f"{self.power[:3]}-{self.kind}{self.location}"


@dataclass
class Order:
    """
    A single order. Types:
      H  = hold
      M  = move (target = destination)
      S  = support (target = supported unit's location;
                    target_dest = where supported unit is moving, or None for hold)
      C  = convoy (target = army's start, target_dest = army's destination)
      R  = retreat (target = retreat destination)
      D  = disband (during retreat or adjustment)
      B  = build (during adjustment; unit_kind + location)
    """
    power: str
    unit_kind: str           # "A" or "F"
    location: str            # where the unit is
    type: str                # H / M / S / C / R / D / B
    target: Optional[str] = None
    target_dest: Optional[str] = None

    # populated by adjudicator
    resolved: Optional[str] = None  # "succeeds" / "fails" / "dislodged"
    note: str = ""

    def signature(self) -> str:
        """Human-readable order text."""
        u = f"{self.unit_kind} {self.location}"
        if self.type == "H":
            return f"{u} H"
        if self.type == "M":
            return f"{u} - {self.target}"
        if self.type == "S":
            if self.target_dest and self.target_dest != self.target:
                return f"{u} S {self.target} - {self.target_dest}"
            return f"{u} S {self.target}"
        if self.type == "C":
            return f"{u} C {self.target} - {self.target_dest}"
        if self.type == "R":
            return f"{u} R {self.target}"
        if self.type == "D":
            return f"{u} D"
        if self.type == "B":
            return f"BUILD {self.unit_kind} {self.location}"
        return f"{u} ?"


@dataclass
class GameState:
    year: int = 1901
    season: str = "SPRING"        # SPRING / FALL / WINTER
    phase: str = "MOVEMENT"       # MOVEMENT / RETREAT / ADJUSTMENT
    units: list[Unit] = field(default_factory=list)
    sc_owner: dict[str, str] = field(default_factory=dict)  # province -> power
    dislodged: list[Unit] = field(default_factory=list)
    dislodged_from: dict[str, str] = field(default_factory=dict)  # unit_loc -> attacker_loc
    eliminated: set[str] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)


# ----------------------------- Setup ---------------------------------------- #


def initial_state() -> GameState:
    state = GameState()
    for power, units in STARTING_UNITS.items():
        for kind, loc in units:
            state.units.append(Unit(kind=kind, power=power, location=loc))
    for power, centers in HOME_CENTERS.items():
        for c in centers:
            state.sc_owner[c] = power
    return state


def units_by_power(state: GameState, power: str) -> list[Unit]:
    return [u for u in state.units if u.power == power]


def supply_centers_owned(state: GameState, power: str) -> list[str]:
    return [sc for sc, owner in state.sc_owner.items() if owner == power]


def unit_at(state: GameState, location: str) -> Optional[Unit]:
    for u in state.units:
        if u.location == location:
            return u
    return None


# ----------------------------- Adjudicator ---------------------------------- #


def _support_strength(orders_by_loc: dict[str, Order],
                      supported_loc: str,
                      supported_dest: Optional[str],
                      attacker_loc: Optional[str] = None) -> int:
    """
    Count valid supports for a unit at supported_loc moving to supported_dest
    (or holding if supported_dest is None). Skip cut supports.
    A support is cut if an enemy unit (not the supported unit and not from
    the attacked province) moves into the supporter's province.
    """
    strength = 1  # the unit itself
    for o in orders_by_loc.values():
        if o.type != "S":
            continue
        # Hold-support
        if supported_dest is None:
            if o.target == supported_loc and (o.target_dest is None or o.target_dest == supported_loc):
                if not _is_support_cut(orders_by_loc, o, attacker_loc):
                    strength += 1
        else:
            if o.target == supported_loc and o.target_dest == supported_dest:
                if not _is_support_cut(orders_by_loc, o, attacker_loc):
                    strength += 1
    return strength


def _is_support_cut(orders_by_loc: dict[str, Order],
                    support_order: Order,
                    attacker_loc: Optional[str]) -> bool:
    """
    Support is cut if any unit (other than the supported unit and other than
    a unit at the province being attacked by the supported move) moves into
    the supporter's location.
    """
    supporter_loc = support_order.location
    supported_dest = support_order.target_dest
    for o in orders_by_loc.values():
        if o.type != "M":
            continue
        if o.target != supporter_loc:
            continue
        # Same power's unit doesn't cut support
        if o.power == support_order.power:
            continue
        # The unit being supported moves into supporter? (impossible, skip)
        if o.location == support_order.target:
            continue
        # Special: the supporter is attacked from the province it's supporting
        # an attack against -> that doesn't cut (Diplomacy rule). I.e. if
        # supporter supports A->B, an attack from B doesn't cut.
        if supported_dest and o.location == supported_dest:
            continue
        return True
    return False


def adjudicate_movement(state: GameState, orders: list[Order]) -> tuple[GameState, list[str]]:
    """
    Run movement-phase adjudication. Returns (new_state, log_lines).
    """
    new_state = copy.deepcopy(state)
    new_state.dislodged = []
    new_state.dislodged_from = {}
    log: list[str] = []

    # Index orders by unit location. Default to HOLD for unordered units.
    orders_by_loc: dict[str, Order] = {}
    for u in new_state.units:
        orders_by_loc[u.location] = Order(
            power=u.power, unit_kind=u.kind, location=u.location, type="H"
        )
    for o in orders:
        # Validate the unit exists and matches
        u = unit_at(new_state, o.location)
        if u is None or u.kind != o.unit_kind or u.power != o.power:
            o.resolved = "fails"
            o.note = "no matching unit"
            continue
        orders_by_loc[o.location] = o

    # Validate move legality (adjacency / can-occupy). Convoyed moves: if a fleet
    # at intermediate sea is convoying, an army can move further. We support a
    # one-hop convoy: an army moves to an adjacent (via a fleet's sea province)
    # coastal province if exactly one fleet at a sea adjacent to both is C-ing.
    for o in list(orders_by_loc.values()):
        if o.type == "M":
            legal = is_adjacent(o.unit_kind, o.location, o.target) and can_occupy(o.unit_kind, o.target)
            if not legal and o.unit_kind == "A":
                # try simple convoy
                if _has_convoy_path(orders_by_loc, o):
                    legal = True
            if not legal:
                o.resolved = "fails"
                o.note = "illegal move"
                # demote to hold for adjudication purposes
                orders_by_loc[o.location] = Order(
                    power=o.power, unit_kind=o.unit_kind,
                    location=o.location, type="H"
                )
        elif o.type == "S":
            # support legal only if supporter could itself attack target_dest
            sup_dest = o.target_dest or o.target
            if not is_adjacent(o.unit_kind, o.location, sup_dest):
                o.resolved = "fails"
                o.note = "support out of range"
                orders_by_loc[o.location] = Order(
                    power=o.power, unit_kind=o.unit_kind,
                    location=o.location, type="H"
                )

    # Compute strengths and resolve.
    # We iterate to a fixed point because support cuts depend on move
    # outcomes for the simple cases we care about.
    # First pass: tentative attack strengths.
    move_orders = [o for o in orders_by_loc.values() if o.type == "M"]
    contested: dict[str, list[tuple[Order, int]]] = {}
    for mo in move_orders:
        strength = _support_strength(
            orders_by_loc, mo.location, mo.target, attacker_loc=mo.target
        )
        contested.setdefault(mo.target, []).append((mo, strength))

    # Holds + their defense strengths
    hold_strength: dict[str, int] = {}
    for loc, o in orders_by_loc.items():
        if o.type in ("H", "S", "C"):
            hold_strength[loc] = _support_strength(
                orders_by_loc, loc, None, attacker_loc=None
            )

    # Resolve contests
    winners: dict[str, Order] = {}   # destination -> winning move (or none)
    failed_moves: list[Order] = []
    for dest, attempts in contested.items():
        attempts.sort(key=lambda x: -x[1])
        top_strength = attempts[0][1]
        leaders = [a for a in attempts if a[1] == top_strength]

        defender_unit = unit_at(new_state, dest)
        defender_order = orders_by_loc.get(dest)
        defender_moving_away = (
            defender_order is not None
            and defender_order.type == "M"
            and defender_order.target != dest
        )
        # For now treat defender as holding if not moving away
        defender_strength = hold_strength.get(dest, 0) if defender_unit and not defender_moving_away else 0

        if len(leaders) > 1 or (defender_unit and not defender_moving_away and top_strength <= defender_strength):
            # bounce or repulsed
            for a, _s in attempts:
                a.resolved = "fails"
                a.note = "bounced" if len(leaders) > 1 else "repulsed"
                failed_moves.append(a)
            continue

        winning_move = leaders[0][0]
        # Self-dislodgement check: a power may not dislodge its own unit
        if defender_unit and defender_unit.power == winning_move.power and not defender_moving_away:
            winning_move.resolved = "fails"
            winning_move.note = "would self-dislodge"
            failed_moves.append(winning_move)
            for a, _s in attempts[1:]:
                a.resolved = "fails"
                a.note = "bounced"
                failed_moves.append(a)
            continue

        winning_move.resolved = "succeeds"
        winners[dest] = winning_move
        for a, _s in attempts:
            if a is not winning_move:
                a.resolved = "fails"
                a.note = "outbid"
                failed_moves.append(a)

    # Apply: move winners, mark dislodged
    moved_locations: set[str] = set()
    for dest, mv in winners.items():
        # Find the unit at mv.location
        u = unit_at(new_state, mv.location)
        if u is None:
            continue
        # Dislodge defender if any
        defender = unit_at(new_state, dest)
        if defender and defender is not u:
            new_state.dislodged.append(defender)
            new_state.dislodged_from[defender.location] = mv.location
            new_state.units = [x for x in new_state.units if x is not defender]
        u.location = dest
        moved_locations.add(mv.location)

    for o in orders_by_loc.values():
        if o.type == "M" and o.resolved is None:
            o.resolved = "fails"
        if o.type in ("H", "S", "C") and o.resolved is None:
            o.resolved = "succeeds"

    # Build log
    for o in orders_by_loc.values():
        if o.type == "M":
            log.append(f"{o.power}: {o.signature()} -> {o.resolved}{(' ('+o.note+')') if o.note else ''}")
        elif o.type == "S":
            log.append(f"{o.power}: {o.signature()} -> {o.resolved}")
    return new_state, log


def _has_convoy_path(orders_by_loc: dict[str, Order], move: Order) -> bool:
    """Very simple convoy check: any fleet ordered C from move.location to move.target."""
    for o in orders_by_loc.values():
        if o.type == "C" and o.target == move.location and o.target_dest == move.target:
            # also require fleet adjacent to both src and dst (1-hop convoy)
            f_loc = o.location
            if (move.location in ADJ.get(f_loc, {}).get("fleet", []) and
                    move.target in ADJ.get(f_loc, {}).get("fleet", [])):
                return True
    return False


# --------------------------- Phase progression ------------------------------ #


def adjudicate_retreats(state: GameState, retreat_orders: list[Order]) -> tuple[GameState, list[str]]:
    """Resolve retreat phase. Each dislodged unit either retreats or disbands."""
    new_state = copy.deepcopy(state)
    log: list[str] = []
    targets: dict[str, list[Order]] = {}

    # Validate retreats
    for ro in retreat_orders:
        if ro.type == "D":
            log.append(f"{ro.power}: {ro.unit_kind} {ro.location} disbands")
            continue
        if ro.type != "R":
            continue
        # must be a dislodged unit
        if not any(u.location == ro.location and u.power == ro.power for u in new_state.dislodged):
            continue
        # cannot retreat to attacker's origin or to any province with a unit
        attacker_origin = new_state.dislodged_from.get(ro.location)
        if ro.target == attacker_origin:
            log.append(f"{ro.power}: {ro.signature()} -> illegal (attacker's origin)")
            continue
        if unit_at(new_state, ro.target):
            log.append(f"{ro.power}: {ro.signature()} -> illegal (occupied)")
            continue
        if not is_adjacent(ro.unit_kind, ro.location, ro.target):
            log.append(f"{ro.power}: {ro.signature()} -> illegal (not adjacent)")
            continue
        targets.setdefault(ro.target, []).append(ro)

    # If two retreats target same province, both disband
    for tgt, lst in targets.items():
        if len(lst) > 1:
            for ro in lst:
                log.append(f"{ro.power}: retreat to {tgt} contested -> disband")
        else:
            ro = lst[0]
            # find and move the dislodged unit
            unit = next(u for u in new_state.dislodged
                        if u.location == ro.location and u.power == ro.power)
            unit.location = ro.target
            new_state.units.append(unit)
            log.append(f"{ro.power}: {ro.signature()} -> succeeds")

    # any remaining dislodged units disband
    handled_locs = {r.location for ro_list in targets.values() for r in ro_list if len(ro_list) == 1}
    for u in new_state.dislodged:
        if u.location not in handled_locs:
            log.append(f"{u.power}: {u.kind} {u.location} disbands (no retreat)")

    new_state.dislodged = []
    new_state.dislodged_from = {}
    return new_state, log


def update_supply_centers(state: GameState) -> list[str]:
    """After Fall movement+retreats, supply centers change ownership based on occupiers."""
    log: list[str] = []
    for u in state.units:
        if u.location in ALL_SUPPLY_CENTERS:
            prev = state.sc_owner.get(u.location)
            if prev != u.power:
                state.sc_owner[u.location] = u.power
                log.append(f"{u.power} captures {u.location} (was {prev or 'neutral'})")
    return log


def adjudicate_adjustments(state: GameState, adj_orders: list[Order]) -> tuple[GameState, list[str]]:
    """
    Build / disband phase. For each power compute SC-count vs unit-count.
    Build orders are honored if in own home centers that are unowned-by-no-one
    -- well, in own home centers currently owned and unoccupied. Excess units
    must disband (we auto-disband furthest from home if not specified).
    """
    new_state = copy.deepcopy(state)
    log: list[str] = []
    for power in POWERS:
        if power in new_state.eliminated:
            continue
        scs = supply_centers_owned(new_state, power)
        units = units_by_power(new_state, power)
        delta = len(scs) - len(units)
        if delta == 0:
            continue
        if delta > 0:
            # build, up to delta units, in own home centers that are owned and empty
            available = [c for c in HOME_CENTERS[power]
                         if new_state.sc_owner.get(c) == power
                         and not unit_at(new_state, c)]
            requested = [o for o in adj_orders
                         if o.power == power and o.type == "B"
                         and o.location in available]
            built = 0
            for o in requested:
                if built >= delta:
                    break
                new_state.units.append(Unit(kind=o.unit_kind, power=power, location=o.location))
                log.append(f"{power} builds {o.unit_kind} {o.location}")
                built += 1
        else:
            # must disband |delta| units
            requested = [o for o in adj_orders if o.power == power and o.type == "D"]
            disbanded = 0
            for o in requested:
                if disbanded >= -delta:
                    break
                u = unit_at(new_state, o.location)
                if u and u.power == power:
                    new_state.units.remove(u)
                    log.append(f"{power} disbands {u.kind} {u.location}")
                    disbanded += 1
            # auto-disband the rest
            while disbanded < -delta:
                remaining = units_by_power(new_state, power)
                if not remaining:
                    break
                # heuristic: disband first alphabetically
                victim = sorted(remaining, key=lambda x: x.location)[0]
                new_state.units.remove(victim)
                log.append(f"{power} auto-disbands {victim.kind} {victim.location}")
                disbanded += 1

    # mark eliminated powers
    for power in POWERS:
        if not units_by_power(new_state, power) and not supply_centers_owned(new_state, power):
            new_state.eliminated.add(power)

    return new_state, log


def advance_phase(state: GameState) -> GameState:
    """Advance to the next phase, season, or year."""
    new = copy.deepcopy(state)
    if new.phase == "MOVEMENT":
        if new.dislodged:
            new.phase = "RETREAT"
        elif new.season == "FALL":
            new.phase = "ADJUSTMENT"
        else:
            new.season = "FALL"
            new.phase = "MOVEMENT"
    elif new.phase == "RETREAT":
        if new.season == "FALL":
            new.phase = "ADJUSTMENT"
        else:
            new.season = "FALL"
            new.phase = "MOVEMENT"
    elif new.phase == "ADJUSTMENT":
        new.year += 1
        new.season = "SPRING"
        new.phase = "MOVEMENT"
    return new


# ------------------------------ Helpers ------------------------------------- #


def parse_order(power: str, text: str) -> Optional[Order]:
    """
    Lenient parser. Examples:
      A PAR - BUR
      F BRE H
      A MUN S A KIE - BER
      A PAR S A MAR
      F MAO C A LON - BRE
      BUILD A PAR
      DISBAND A PAR
    """
    t = text.strip().upper().replace("--", "-").replace("→", "-")
    if not t:
        return None
    parts = t.split()
    try:
        if parts[0] == "BUILD":
            return Order(power=power, unit_kind=parts[1],
                         location=parts[2], type="B")
        if parts[0] == "DISBAND":
            return Order(power=power, unit_kind=parts[1],
                         location=parts[2], type="D")
        if parts[0] in ("A", "F"):
            kind, loc = parts[0], parts[1]
            if len(parts) == 3 and parts[2] in ("H", "HOLD"):
                return Order(power=power, unit_kind=kind, location=loc, type="H")
            if "-" in parts:
                idx = parts.index("-")
                if idx == 2:
                    # M order: A PAR - BUR
                    return Order(power=power, unit_kind=kind, location=loc,
                                 type="M", target=parts[3])
            if "S" in parts:
                idx = parts.index("S")
                # A MUN S A KIE - BER  or A PAR S A MAR
                rest = parts[idx + 1:]
                if "-" in rest:
                    di = rest.index("-")
                    sup_loc = rest[di - 1]
                    sup_dest = rest[di + 1]
                    return Order(power=power, unit_kind=kind, location=loc,
                                 type="S", target=sup_loc, target_dest=sup_dest)
                else:
                    sup_loc = rest[-1]
                    return Order(power=power, unit_kind=kind, location=loc,
                                 type="S", target=sup_loc, target_dest=sup_loc)
            if "C" in parts:
                idx = parts.index("C")
                rest = parts[idx + 1:]
                di = rest.index("-")
                conv_src = rest[di - 1]
                conv_dst = rest[di + 1]
                return Order(power=power, unit_kind=kind, location=loc,
                             type="C", target=conv_src, target_dest=conv_dst)
            if "R" in parts:
                idx = parts.index("R")
                tgt = parts[idx + 1]
                return Order(power=power, unit_kind=kind, location=loc,
                             type="R", target=tgt)
    except (IndexError, ValueError):
        return None
    return None
