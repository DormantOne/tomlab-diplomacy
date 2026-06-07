"""
Top-level game session: ties the engine, the human, and the five LLM agents
together. Provides phase-by-phase advancement.

Phase order each turn:
  Spring negotiation -> Spring orders -> Spring retreats (if any)
  Fall negotiation   -> Fall orders   -> Fall retreats (if any) -> SC update
  Adjustment         -> next year

This module also handles:
  - cross-game KG persistence (theory_of_mind + counterfactuals carry over)
  - tutorial mode (scripted Spring 1901, deterministic AIs, step overlays)
  - human notes about each power (revealed alongside AI's notes about the human at end)
  - hide-personality / hide-AI-chatter toggles
  - click-to-order helpers (legal destinations / supports / convoys)
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from diplomacy_engine import (
    GameState, Order, POWERS, ADJ, PROVINCES,
    initial_state,
    adjudicate_movement, adjudicate_retreats, adjudicate_adjustments,
    update_supply_centers, advance_phase, parse_order,
    units_by_power, supply_centers_owned, ALL_SUPPLY_CENTERS,
    HOME_CENTERS, can_occupy, is_adjacent, unit_at,
)
from agents import Message, PERSONALITIES  # Message + PERSONALITIES kept for legacy bookkeeping
from .v2_bridge import V2BridgeAgent, make_default_llm_call
from .snapshot import SnapshotWriter


DEFAULT_ARCHETYPES = list(PERSONALITIES.keys())


# ============================================================================
# Persistence layer
# ============================================================================

PERSIST_DIR = Path.home() / ".diplomacy_llm"
AGENTS_DIR = PERSIST_DIR / "agents"
PROFILE_PATH = PERSIST_DIR / "profile.json"
HISTORY_PATH = PERSIST_DIR / "history.json"


def _ensure_dirs() -> None:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    _ensure_dirs()
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except Exception:
            pass
    return {"player_name": None, "games_played": 0}


def save_profile(profile: dict) -> None:
    _ensure_dirs()
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))


def load_archetype_kg(archetype: str) -> Optional[dict]:
    """Return saved KG dict for an archetype, or None if no save exists."""
    _ensure_dirs()
    p = AGENTS_DIR / f"{archetype}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_archetype_kg(archetype: str, kg_dict: dict) -> None:
    _ensure_dirs()
    p = AGENTS_DIR / f"{archetype}.json"
    p.write_text(json.dumps(kg_dict))


def reset_archetype_memories() -> int:
    """Wipe all archetype KG saves. Returns how many were deleted."""
    _ensure_dirs()
    n = 0
    for p in AGENTS_DIR.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return n


def append_history(entry: dict) -> None:
    _ensure_dirs()
    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text())
        except Exception:
            history = []
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def load_history() -> list:
    _ensure_dirs()
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            pass
    return []


# ============================================================================
# Game session
# ============================================================================

@dataclass
class GameSession:
    user_power: str
    state: GameState = field(default_factory=initial_state)
    agents: dict[str, V2BridgeAgent] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    pending_user_orders: list[Order] = field(default_factory=list)
    awaiting: str = "negotiation"
    turn_lock: threading.Lock = field(default_factory=threading.Lock)
    ai_busy: bool = False

    # Options
    hide_personalities: bool = False
    hide_ai_chatter: bool = True       # default ON — much better game
    tutorial_mode: bool = False
    tutorial_step: int = 0
    spectator_mode: bool = False       # all 6 powers played by AIs, human watches
    auto_playing: bool = False         # auto-runner currently looping
    auto_speed: float = 4.0            # seconds between phases in spectator
    player_name: str = ""
    user_notes: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished: bool = False
    winner: Optional[str] = None
    snapshot_writer: Optional[SnapshotWriter] = None
    # Per-power agent mode for ablation: power → "full" | "muted" | "raw_llm"
    # Powers not listed default to "full".
    agent_modes: dict = field(default_factory=dict)
    # LLM provider selection (None = auto-detect from env / Ollama). See llm_providers.
    llm_kind: Optional[str] = None
    llm_model: Optional[str] = None

    def __post_init__(self):
        # Initialize snapshot writer for this game
        try:
            self.snapshot_writer = SnapshotWriter()
        except Exception as e:
            print(f"snapshot writer init failed (non-fatal): {e}")
            self.snapshot_writer = None
        if self.spectator_mode:
            agent_powers = list(POWERS)            # all six
            self.user_power = ""                    # no human seat
        else:
            agent_powers = [p for p in POWERS if p != self.user_power]
        rng = random.Random(0xD1)
        archetypes = list(DEFAULT_ARCHETYPES)
        rng.shuffle(archetypes)
        # In spectator we have 6 powers but only 5 archetypes — repeat one.
        # Pick an extra deterministically so the same combo recurs each spectator game.
        if len(agent_powers) > len(archetypes):
            archetypes = archetypes + [archetypes[0]]
        # Build a single LLM caller shared by all six agents. If the API
        # key is missing we still construct agents — the error surfaces in
        # _check_llm_health() via a clear log line, and tutorial mode
        # doesn't actually call the LLM.
        try:
            from .v2_bridge import make_default_llm_call_labeled
            llm_call, self._llm_kind, self._llm_model = \
                make_default_llm_call_labeled(self.llm_kind, self.llm_model)
        except RuntimeError:
            llm_call = None  # _check_llm_health will report this
            self._llm_kind, self._llm_model = None, None
        _model_label = self._llm_model or "claude-haiku-4-5-20251001"
        for power, archetype in zip(agent_powers, archetypes):
            mode = self.agent_modes.get(power, "full")
            if mode == "raw_llm":
                from .raw_llm_agent import RawLLMAgent
                agent = RawLLMAgent(
                    power=power,
                    llm_call=llm_call or (lambda p: ""),
                    model_label=_model_label,
                )
            else:
                agent = V2BridgeAgent(
                    power=power, archetype=archetype,
                    llm_call=llm_call or (lambda p: ""),
                    muted=(mode == "muted"),
                    model_label=_model_label,
                )
            self.agents[power] = agent
        for p in POWERS:
            self.user_notes.setdefault(p, "")
        if self.spectator_mode:
            self.log.append("Spectator mode: all six powers are LLMs. Press Play to start.")
        else:
            self.log.append(f"Game started. You play {self.user_power}.")
        if self.tutorial_mode:
            self.log.append("(Tutorial mode: AIs play scripted, no LLM is called.)")
        else:
            self._check_llm_health()

    def _check_llm_health(self) -> None:
        """Report which LLM provider is driving the agents (or that none is
        available) so the user sees an obvious status before the first turn."""
        if not self.agents:
            return
        import llm_providers
        kind = getattr(self, "_llm_kind", None)
        model = getattr(self, "_llm_model", None)
        if kind:
            self.log.append(
                f"LLM ready: {llm_providers.label(kind, model)} for all powers. "
                f"({llm_providers.describe()})")
        else:
            self.log.append(
                "WARNING: no LLM provider available. AIs will fall back to "
                "all-hold heuristics and won't speak. Set ANTHROPIC_API_KEY "
                "(or OPENAI_API_KEY / GOOGLE_API_KEY), or start Ollama locally, "
                f"then restart. ({llm_providers.describe()})")

    # ====================================================================
    # Visible board info for the UI
    # ====================================================================

    def _agent_display_name(self, power: str) -> str:
        """Safe display-name lookup. Handles raw_llm agents whose
        personality_key isn't in PERSONALITIES."""
        if power not in self.agents:
            return "Unknown"
        key = self.agents[power].personality_key
        info = PERSONALITIES.get(key)
        if info:
            return info["display_name"]
        # Fallback for raw_llm or other non-archetype agents
        return key.replace("_", " ").title()

    def _agent_tagline(self, power: str) -> str:
        if power not in self.agents:
            return ""
        key = self.agents[power].personality_key
        info = PERSONALITIES.get(key)
        return info["tagline"] if info else "(ablation agent)"

    def board_summary(self) -> dict:
        return {
            "year": self.state.year,
            "season": self.state.season,
            "phase": self.state.phase,
            "awaiting": self.awaiting,
            "user_power": self.user_power,
            "powers": [
                {
                    "name": p,
                    "is_user": p == self.user_power,
                    "agent": (
                        ("(you)" if p == self.user_power and not self.spectator_mode else
                         (self._agent_display_name(p)
                          if (p in self.agents and not self.hide_personalities) else "Unknown"))
                    ),
                    "archetype": (None if (self.hide_personalities or p == self.user_power)
                                  else (self.agents[p].personality_key if p in self.agents else None)),
                    "supply_centers": sorted(supply_centers_owned(self.state, p)),
                    "units": [{"kind": u.kind, "location": u.location}
                              for u in units_by_power(self.state, p)],
                    "eliminated": p in self.state.eliminated,
                }
                for p in POWERS
            ],
            "all_sc": ALL_SUPPLY_CENTERS,
            "dislodged": [{"power": u.power, "kind": u.kind, "location": u.location,
                           "from": self.state.dislodged_from.get(u.location, "")}
                          for u in self.state.dislodged],
            "tutorial_mode": self.tutorial_mode,
            "tutorial_step": self.tutorial_step,
            "spectator_mode": self.spectator_mode,
            "auto_playing": self.auto_playing,
            "auto_speed": self.auto_speed,
            "hide_personalities": self.hide_personalities,
            "hide_ai_chatter": self.hide_ai_chatter,
            "player_name": self.player_name,
            "finished": self.finished,
            "winner": self.winner,
            "staged_orders": [o.signature() for o in self.pending_user_orders],
        }

    # ====================================================================
    # Message visibility
    # ====================================================================

    def visible_messages_for_user(self) -> list[Message]:
        if self.spectator_mode:
            return list(self.messages)   # spectator sees everything
        out = []
        for m in self.messages:
            if m.sender == self.user_power:
                out.append(m); continue
            if m.public:
                out.append(m); continue
            if self.user_power in m.recipients:
                out.append(m); continue
            if not self.hide_ai_chatter:
                out.append(m)
        return out

    # ====================================================================
    # Click-to-order helpers
    # ====================================================================

    def legal_destinations(self, location: str) -> dict:
        units = units_by_power(self.state, self.user_power)
        unit = next((u for u in units if u.location == location), None)
        if not unit:
            return {"error": "no unit there"}
        key = "army" if unit.kind == "A" else "fleet"
        adj = ADJ.get(location, {}).get(key, [])
        moves = [n for n in adj if can_occupy(unit.kind, n)]

        support_holds = []
        for prov in adj:
            here = next((u for u in self.state.units if u.location == prov), None)
            if here:
                support_holds.append({
                    "location": prov, "kind": here.kind, "power": here.power,
                })
        support_moves = []
        for target_prov in adj:
            for u in self.state.units:
                if u.location == location:
                    continue
                ukey = "army" if u.kind == "A" else "fleet"
                if target_prov in ADJ.get(u.location, {}).get(ukey, []) \
                   and can_occupy(u.kind, target_prov):
                    support_moves.append({
                        "from": u.location, "to": target_prov,
                        "kind": u.kind, "power": u.power,
                    })

        convoys = []
        if unit.kind == "F" and PROVINCES[location][0] == "sea":
            sea_neighbors = ADJ.get(location, {}).get("fleet", [])
            coastal_pairs = [p for p in sea_neighbors
                             if PROVINCES[p][0] == "coast"]
            for src in coastal_pairs:
                src_unit = next((u for u in self.state.units if u.location == src
                                 and u.kind == "A"), None)
                if not src_unit:
                    continue
                for dst in coastal_pairs:
                    if dst == src:
                        continue
                    convoys.append({
                        "army_from": src, "army_to": dst,
                        "army_power": src_unit.power,
                    })

        return {
            "unit": {"kind": unit.kind, "location": location},
            "moves": moves,
            "support_holds": support_holds,
            "support_moves": support_moves,
            "convoys": convoys,
            "can_hold": True,
        }

    def legal_build_options(self) -> dict:
        """For the adjustment phase: which home centers can the user build in,
        and how many builds/disbands are due."""
        scs = supply_centers_owned(self.state, self.user_power)
        units = units_by_power(self.state, self.user_power)
        delta = len(scs) - len(units)
        available_homes = [c for c in HOME_CENTERS[self.user_power]
                           if self.state.sc_owner.get(c) == self.user_power
                           and not unit_at(self.state, c)]
        return {
            "delta": delta,
            "build_count": max(0, delta),
            "disband_count": max(0, -delta),
            "available_homes": available_homes,
            "current_units": [{"kind": u.kind, "location": u.location} for u in units],
        }

    # ====================================================================
    # Negotiation step
    # ====================================================================

    def run_ai_negotiation(self) -> None:
        if self.ai_busy or self.tutorial_mode:
            return
        with self.turn_lock:
            self.ai_busy = True
            try:
                self.log.append(f"--- {self.state.year} {self.state.season}: negotiation round ---")
                active = [(p, a) for p, a in self.agents.items()
                          if p not in self.state.eliminated]
                for i, (power, agent) in enumerate(active, 1):
                    self.log.append(f"  [{i}/{len(active)}] {power} thinking…")
                    try:
                        visible = [m for m in self.messages
                                   if m.public or power in m.recipients or m.sender == power]
                        new_msgs = agent.negotiate(self.state, visible)
                    except Exception as e:
                        self.log.append(f"  {power}: NEGOTIATION ERROR: {e}")
                        continue
                    self.messages.extend(new_msgs)
                    # CRITICAL: feed new messages into every recipient's V2
                    # mind so they parse incoming commitspeak. Without this,
                    # incoming_commitments stays empty forever (the bug we
                    # found by analyzing the first long game).
                    if new_msgs:
                        try:
                            from .v2_bridge import distribute_messages_to_agents
                            phase_str = (f"{self.state.year}-{self.state.season}-"
                                         f"{self.state.phase}")
                            distribute_messages_to_agents(
                                new_msgs, self.agents, phase_str)
                        except Exception as e:
                            self.log.append(f"  [intake error: {e}]")
                    if not new_msgs:
                        self.log.append(f"  {power} stays silent.")
                    for m in new_msgs:
                        if m.public:
                            target = "ALL"
                        elif self.user_power and (self.user_power in m.recipients or m.sender == self.user_power):
                            target = ",".join(m.recipients) or "ALL"
                        else:
                            if self.hide_ai_chatter:
                                self.log.append(f"  {m.sender} sent a private message")
                                continue
                            target = ",".join(m.recipients) or "ALL"
                        # Log a SHORT summary; full text is in the messages tab.
                        preview = m.text.replace("\n", " ").strip()
                        if len(preview) > 100:
                            preview = preview[:97] + "…"
                        self.log.append(f"  {m.sender} -> {target}: {preview}")
            finally:
                self.ai_busy = False

    def user_send_message(self, recipients: list[str], text: str, public: bool) -> None:
        if not text.strip():
            return
        recipients = [r for r in recipients if r in POWERS and r != self.user_power]
        m = Message(sender=self.user_power, recipients=recipients, text=text.strip(),
                    season=self.state.season, year=self.state.year,
                    public=public or not recipients)
        self.messages.append(m)
        # Distribute to AI agents so they parse the human's commitspeak too
        try:
            from .v2_bridge import distribute_messages_to_agents
            phase_str = (f"{self.state.year}-{self.state.season}-"
                         f"{self.state.phase}")
            distribute_messages_to_agents([m], self.agents, phase_str)
        except Exception as e:
            self.log.append(f"  [intake error: {e}]")
        target = "ALL" if m.public else ",".join(m.recipients)
        self.log.append(f"  {self.user_power} -> {target}: {m.text}")

    # ====================================================================
    # Orders step
    # ====================================================================

    def submit_user_orders(self, order_lines: list[str]) -> list[str]:
        errors: list[str] = []
        my_units = units_by_power(self.state, self.user_power)
        parsed: list[Order] = []
        for line in order_lines:
            if not line.strip():
                continue
            o = parse_order(self.user_power, line)
            if not o:
                errors.append(f"Could not parse: {line}")
                continue
            if o.type in ("M", "S", "C", "H"):
                if not any(u.location == o.location and u.kind == o.unit_kind
                           for u in my_units):
                    errors.append(f"No matching unit for: {line}")
                    continue
            parsed.append(o)
        ordered_locs = {o.location for o in parsed}
        for u in my_units:
            if u.location not in ordered_locs:
                parsed.append(Order(power=self.user_power, unit_kind=u.kind,
                                    location=u.location, type="H"))
        self.pending_user_orders = parsed
        return errors

    def run_movement_phase(self) -> None:
        if self.ai_busy:
            return
        with self.turn_lock:
            self.ai_busy = True
            try:
                all_orders: list[Order] = list(self.pending_user_orders)

                if self.tutorial_mode:
                    all_orders.extend(self._scripted_ai_orders())
                else:
                    self.log.append(f"--- {self.state.year} {self.state.season}: orders step ---")
                    active = [(p, a) for p, a in self.agents.items()
                              if p not in self.state.eliminated]
                    for i, (power, agent) in enumerate(active, 1):
                        self.log.append(f"  [{i}/{len(active)}] {power} ordering units…")
                        try:
                            visible = [m for m in self.messages
                                       if m.public or power in m.recipients or m.sender == power]
                            out = agent.decide_orders(self.state, visible)
                        except Exception as e:
                            self.log.append(f"  {power}: ORDERS ERROR: {e}")
                            continue
                        all_orders.extend(out.orders)
                        for note in out.reflection_notes:
                            self.log.append(f"  {power}: {note}")

                self.log.append(f"--- {self.state.year} {self.state.season} adjudication ---")
                pre_state = self.state
                new_state, lines = adjudicate_movement(self.state, all_orders)
                for ln in lines:
                    self.log.append("  " + ln)
                # GRADERS — process incoming/self commitments + predictions.
                # Without this all such records stay PENDING/OPEN forever.
                try:
                    from .v2_bridge import absorb_phase_for_all_agents
                    resolved_phase = (f"{pre_state.year}-{pre_state.season}-"
                                      f"{pre_state.phase}")
                    next_phase = (f"{new_state.year}-{new_state.season}-"
                                  f"{new_state.phase}")
                    tele = absorb_phase_for_all_agents(
                        self.agents, pre_state, new_state, all_orders,
                        lines, resolved_phase, next_phase)
                    summary_bits = []
                    for p, t in tele.items():
                        cg = t.get("commitments_graded", 0)
                        pg = t.get("predictions_graded", 0)
                        if cg or pg:
                            summary_bits.append(f"{p[:3]}:cmt{cg}/pred{pg}")
                    if summary_bits:
                        self.log.append(f"  graders: {' '.join(summary_bits)}")
                except Exception as e:
                    self.log.append(f"  [grader error: {e}]")
                self.state = new_state
                self.pending_user_orders = []
            finally:
                self.ai_busy = False
        self._post_movement_phase_transition()
        self._take_snapshot("after-movement")

    def _post_movement_phase_transition(self) -> None:
        if self.state.dislodged:
            self.awaiting = "retreats"
            self.state.phase = "RETREAT"
        else:
            self._after_retreats()

    def run_retreat_phase(self) -> None:
        retreat_orders: list[Order] = list(self.pending_user_orders)
        for power, agent in self.agents.items():
            if power in self.state.eliminated:
                continue
            retreat_orders.extend(agent.decide_retreats(self.state))
        pre_state = self.state
        new_state, lines = adjudicate_retreats(self.state, retreat_orders)
        for ln in lines:
            self.log.append("  " + ln)
        try:
            from .v2_bridge import absorb_phase_for_all_agents
            resolved_phase = (f"{pre_state.year}-{pre_state.season}-"
                              f"{pre_state.phase}")
            next_phase = (f"{new_state.year}-{new_state.season}-"
                          f"{new_state.phase}")
            absorb_phase_for_all_agents(
                self.agents, pre_state, new_state, retreat_orders,
                lines, resolved_phase, next_phase)
        except Exception as e:
            self.log.append(f"  [grader error: {e}]")
        self.state = new_state
        self.pending_user_orders = []
        self._after_retreats()
        self._take_snapshot("after-retreats")

    def submit_user_retreats(self, order_lines: list[str]) -> list[str]:
        errors: list[str] = []
        for line in order_lines:
            if not line.strip():
                continue
            o = parse_order(self.user_power, line)
            if o and o.type in ("R", "D"):
                self.pending_user_orders.append(o)
            else:
                errors.append(f"Could not parse retreat: {line}")
        return errors

    def _after_retreats(self) -> None:
        if self.state.season == "FALL":
            sc_lines = update_supply_centers(self.state)
            for ln in sc_lines:
                self.log.append("  " + ln)
            for power in POWERS:
                if len(supply_centers_owned(self.state, power)) >= 18:
                    self.log.append(f"*** {power} has won the game with 18+ supply centers ***")
                    self.awaiting = "done"
                    self.finished = True
                    self.winner = power
                    self._on_game_end()
                    return
            self.awaiting = "builds"
            self.state.phase = "ADJUSTMENT"
        else:
            self.state.season = "FALL"
            self.state.phase = "MOVEMENT"
            self.awaiting = "negotiation"

    def run_adjustment_phase(self) -> None:
        adj_orders: list[Order] = list(self.pending_user_orders)
        for power, agent in self.agents.items():
            if power in self.state.eliminated:
                continue
            adj_orders.extend(agent.decide_builds(self.state))
        pre_state = self.state
        new_state, lines = adjudicate_adjustments(self.state, adj_orders)
        for ln in lines:
            self.log.append("  " + ln)
        try:
            from .v2_bridge import absorb_phase_for_all_agents
            resolved_phase = (f"{pre_state.year}-{pre_state.season}-"
                              f"{pre_state.phase}")
            next_phase = (f"{new_state.year}-{new_state.season}-"
                          f"{new_state.phase}")
            absorb_phase_for_all_agents(
                self.agents, pre_state, new_state, adj_orders,
                lines, resolved_phase, next_phase)
        except Exception as e:
            self.log.append(f"  [grader error: {e}]")
        self.state = new_state
        # Elimination hook — fire BEFORE advancing year, so phase_label is
        # the resolved-phase. Runs AFTER absorb_phase_for_all_agents so
        # commits made on the elimination phase can still grade KEPT/BROKEN.
        newly_eliminated = new_state.eliminated - pre_state.eliminated
        if newly_eliminated:
            phase_label = (f"{pre_state.year}-{pre_state.season}-"
                           f"{pre_state.phase}")
            for ep in sorted(newly_eliminated):
                self.log.append(f"  *** {ep} eliminated — running cleanup ***")
                try:
                    self._on_power_eliminated(ep, phase_label)
                except Exception as e:
                    self.log.append(f"  [elimination cleanup failed for {ep}: {e}]")
        self.pending_user_orders = []
        self.state.year += 1
        self.state.season = "SPRING"
        self.state.phase = "MOVEMENT"
        self.awaiting = "negotiation"
        self._take_snapshot("after-adjustment")


    def _on_power_eliminated(self, eliminated_power: str, phase_label: str) -> None:
        """Retire/mark substrate state tied to a now-eliminated power.

        Without this hook, beliefs about an eliminated power persist forever
        and the substrate's dreams keep citing them as live actors. (In one
        16-year run, France at 14 SC still dreamed in year 1915 about
        'Germany enforces the west' four years after Germany's elimination.)
        """
        from diplomacy_kg_schema import (
            BeliefStatus, PredictionStatus, CommitmentStatus,
        )
        for power, agent in self.agents.items():
            if power == eliminated_power:
                continue
            if power in self.state.eliminated:
                continue
            mind = getattr(getattr(agent, "v2", None), "mind", None)
            if mind is None:
                continue
            # Skip raw_llm — _NoMind has class-level mutable defaults; mutating
            # would corrupt class state.
            if getattr(agent.v2, "archetype", "") == "RAW_LLM":
                continue

            retired_b = retired_p = retired_in = retired_self = 0
            for bnode in mind.beliefs.values():
                if (getattr(bnode, "about_power", None) == eliminated_power
                        and bnode.status not in (BeliefStatus.RETIRED,
                                                 BeliefStatus.REVISED)):
                    bnode.status = BeliefStatus.RETIRED
                    bnode.retire_reason = "target_eliminated"
                    bnode.last_updated_phase = phase_label
                    retired_b += 1
            for pnode in mind.predictions.values():
                if (getattr(pnode, "about_power", None) == eliminated_power
                        and pnode.status == PredictionStatus.OPEN):
                    pnode.status = PredictionStatus.SUPERSEDED
                    pnode.resolved_at_phase = phase_label
                    retired_p += 1
            for cnode in mind.incoming_commitments.values():
                if (getattr(cnode, "speaker", None) == eliminated_power
                        and cnode.status == CommitmentStatus.PENDING):
                    cnode.status = CommitmentStatus.IRRELEVANT
                    cnode.resolved_at_phase = phase_label
                    retired_in += 1
            for cnode in mind.self_commitments.values():
                if (getattr(cnode, "target_power", None) == eliminated_power
                        and cnode.status == CommitmentStatus.PENDING):
                    cnode.status = CommitmentStatus.IRRELEVANT
                    cnode.resolved_at_phase = phase_label
                    retired_self += 1

            mind.private_journal.append({
                "phase": f"ELIMINATION-{eliminated_power}",
                "kind": "power-eliminated",
                "target": eliminated_power,
                "text": (
                    f"{eliminated_power} has been eliminated from the game. "
                    f"Beliefs and pending promises tied to that power are now "
                    f"void: retired {retired_b} belief(s), "
                    f"{retired_p} prediction(s), "
                    f"{retired_in} incoming promise(s), "
                    f"{retired_self} of my promise(s) to them. Update your "
                    f"theory of the board accordingly — this actor is gone."
                ),
            })
            self.log.append(
                f"  [elimination cleanup] {power}'s mind: retired "
                f"{retired_b}b/{retired_p}p/{retired_in + retired_self}c "
                f"tied to {eliminated_power}"
            )

    def _take_snapshot(self, label_suffix: str) -> None:
        """Write a per-phase snapshot. Failures here never abort the game."""
        if self.snapshot_writer is None:
            return
        try:
            phase_label = (f"{self.state.year}-{self.state.season}-"
                           f"{self.state.phase}_{label_suffix}")
            self.snapshot_writer.snapshot(self, phase_label)
        except Exception as e:
            print(f"snapshot failed (non-fatal): {e}")

    def submit_user_builds(self, order_lines: list[str]) -> list[str]:
        errors: list[str] = []
        for line in order_lines:
            if not line.strip():
                continue
            o = parse_order(self.user_power, line)
            if o and o.type in ("B", "D"):
                self.pending_user_orders.append(o)
            else:
                errors.append(f"Could not parse build/disband: {line}")
        return errors

    # ====================================================================
    # Notes
    # ====================================================================

    def set_user_note(self, power: str, text: str) -> None:
        if power in POWERS:
            self.user_notes[power] = text

    def get_user_notes(self) -> dict:
        return dict(self.user_notes)

    # ====================================================================
    # KG inspection
    # ====================================================================

    def get_kg_dump(self, power: str, graph: str) -> dict:
        if power not in self.agents:
            return {"error": "no agent for that power"}
        if graph not in self.agents[power].kgs.graphs:
            return {"error": "no such graph"}
        return self.agents[power].kgs.graphs[graph].to_dict()

    # ====================================================================
    # End-of-game reveal
    # ====================================================================

    def end_game_reveal(self) -> dict:
        all_msgs = [{
            "sender": m.sender, "recipients": list(m.recipients),
            "text": m.text, "season": m.season, "year": m.year,
            "public": m.public,
        } for m in self.messages]

        ai_views_of_user: dict[str, dict] = {}
        for power, agent in self.agents.items():
            tom = agent.kgs.graphs["theory_of_mind"]
            user_node = tom.nodes.get(f"power:{self.user_power}")
            cf = agent.kgs.graphs["counterfactuals"]
            cfs_about_user = []
            for nid, n in cf.nodes.items():
                if n.type != "counterfactual":
                    continue
                attrs = n.attrs
                premise = attrs.get("premise", "")
                expected = attrs.get("expected", "")
                if self.user_power in str(premise).upper() or self.user_power in str(expected).upper():
                    cfs_about_user.append({
                        "label": nid, "premise": premise, "expected": expected,
                    })
            archetype_name = self._agent_display_name(power)
            ai_views_of_user[power] = {
                "archetype": agent.personality_key,
                "archetype_display": archetype_name,
                "trust_in_user": user_node.attrs.get("trust", 0.0) if user_node else 0.0,
                "predicted_intent": user_node.attrs.get("predicted_intent", "") if user_node else "",
                "comm_count": user_node.attrs.get("communication_count", 0) if user_node else 0,
                "last_reason": user_node.attrs.get("last_reason", "") if user_node else "",
                "counterfactuals_about_user": cfs_about_user,
            }

        return {
            "winner": self.winner,
            "messages": all_msgs,
            "ai_views_of_user": ai_views_of_user,
            "user_notes": self.user_notes,
            "personalities_revealed": {
                p: {
                    "archetype": self.agents[p].personality_key,
                    "display_name": self._agent_display_name(p),
                    "tagline": self._agent_tagline(p),
                }
                for p in self.agents
            },
        }

    def _on_game_end(self) -> None:
        if self.tutorial_mode:
            return
        # V2 cross-game persistence is handled separately by
        # diplomacy_persistence.save_mind() — not currently wired into
        # the live session. The end-of-game append still happens.
        append_history({
            "ended_at": time.time(),
            "user_power": self.user_power,
            "player_name": self.player_name,
            "winner": self.winner,
            "year_reached": self.state.year,
        })

    def end_game_now(self, reason: str = "ended early") -> None:
        if self.finished:
            return
        self.finished = True
        self.winner = None
        self.awaiting = "done"
        self.log.append(f"*** Game ended: {reason} ***")
        self._on_game_end()

    # ====================================================================
    # Tutorial mode (scripted Spring/Fall 1901 against deterministic AIs)
    # ====================================================================

    _TUTORIAL_AI_SPRING = {
        "ENGLAND": ["F LON - NTH", "F EDI - NWG", "A LVP - YOR"],
        "GERMANY": ["A BER - KIE", "A MUN - RUH", "F KIE - DEN"],
        "RUSSIA":  ["A MOS - UKR", "A WAR - GAL", "F SEV - BLA", "F STP - BOT"],
        "AUSTRIA": ["A VIE - GAL", "A BUD - SER", "F TRI - ALB"],
        "TURKEY":  ["F ANK - BLA", "A CON - BUL", "A SMY - CON"],
    }
    _TUTORIAL_AI_FALL = {
        "ENGLAND": ["F NTH - NWY", "F NWG - BAR", "A YOR - LON"],
        "GERMANY": ["A KIE - HOL", "A RUH - BEL", "F DEN H"],
        "RUSSIA":  ["A UKR - RUM", "A GAL S A UKR - RUM", "F BLA H", "F BOT - SWE"],
        "AUSTRIA": ["A GAL S A UKR - RUM", "A SER - BUD", "F ALB - GRE"],
        "TURKEY":  ["F BLA H", "A BUL H", "A CON - BUL"],
    }

    def _scripted_ai_orders(self) -> list[Order]:
        season = self.state.season
        script = self._TUTORIAL_AI_SPRING if season == "SPRING" else self._TUTORIAL_AI_FALL
        out: list[Order] = []
        for power, lines in script.items():
            if power == self.user_power or power in self.state.eliminated:
                continue
            for ln in lines:
                o = parse_order(power, ln)
                if o:
                    out.append(o)
        return out

    def advance_tutorial_step(self) -> None:
        self.tutorial_step += 1

    # ====================================================================
    # Spectator auto-play loop
    # ====================================================================

    def _step_one_phase(self, negotiation_rounds: int = 2) -> None:
        """Run whichever phase is next. Used by both auto-play loop and Step button."""
        if self.finished:
            return
        if self.awaiting == "negotiation":
            for i in range(max(1, negotiation_rounds)):
                if not (self.spectator_mode and self.auto_playing) and i > 0:
                    # Step mode = only one round per click; auto-play loops more.
                    break
                self.run_ai_negotiation()
                if self.spectator_mode and self.auto_playing:
                    time.sleep(min(self.auto_speed * 0.6, 4.0))
            self.run_movement_phase()
        elif self.awaiting == "retreats":
            self.run_retreat_phase()
        elif self.awaiting == "builds":
            self.run_adjustment_phase()
        # After every phase resolution, fire the substrate observer in the
        # background so the Substrate tab updates phase-by-phase.
        # Failures are caught and logged but never break the game loop.
        self._fire_substrate_rebuild()

    def _fire_substrate_rebuild(self) -> None:
        """No-op since the V2 migration. Minds are LIVE now — the inspector
        reads them directly from each agent. Kept as a method so call sites
        like _after_retreats() don't need editing."""
        return

    def run_auto_loop(self) -> None:
        """Background-thread loop: keeps stepping phases until paused or game ends."""
        if not self.spectator_mode:
            return
        self.auto_playing = True
        self.log.append("=== spectator auto-play started ===")
        while self.auto_playing and not self.finished:
            try:
                phase = self.awaiting
                year_before = self.state.year
                season_before = self.state.season
                self._step_one_phase(negotiation_rounds=2)
                if not self.auto_playing or self.finished:
                    break
                # Brief pause between phases so the human can read messages /
                # see resolution before the next phase fires.
                time.sleep(self.auto_speed)
                # If nothing changed (stuck), break to avoid infinite loop.
                if (self.awaiting == phase
                    and self.state.year == year_before
                    and self.state.season == season_before
                    and self.awaiting not in ("negotiation",)):
                    self.log.append(f"  (auto-play: stuck on {phase}, pausing)")
                    break
            except Exception as e:
                import traceback
                self.log.append(f"AUTO-PLAY ERROR: {e}")
                self.log.append(traceback.format_exc()[:600])
                time.sleep(2)
                # don't infinite-loop on a persistent error
                break
        self.auto_playing = False
        self.log.append("=== spectator auto-play paused ===")

    def pause_auto(self) -> None:
        self.auto_playing = False

    def step_once(self) -> None:
        """Run a single phase. Used by manual Step button in spectator."""
        if not self.spectator_mode:
            return
        threading.Thread(target=self._step_one_phase, daemon=True).start()
