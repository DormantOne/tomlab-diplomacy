"""
live_session.py — the substrate-driven live game session.

Owns:
  - the engine GameState
  - one DiplomacyAgentV2 per power
  - the message log (accumulated MessageEvents)
  - per-phase snapshots (cached as the game progresses, so the viewer can
    scrub backwards without re-serializing)
  - per-phase log entries (lifecycle outcomes: what happened this turn)
  - a background worker thread that runs phases when asked

This module is the runtime equivalent of run_v2.py but exposed as a class
the Flask app can poke. The phase loop logic itself stays faithful to
run_one_phase() in run_v2.py — but we capture much richer telemetry as
we go (so the UI can show "what just happened" without re-parsing files).

Threading model:
  - The web app thread reads the session's current state through accessor
    methods that grab a lock briefly and return immutable copies.
  - The worker thread holds the lock while mutating engine state, agents,
    and snapshots; it only releases between LLM calls so the page can show
    "thinking" indicators.
  - In auto-play mode the worker keeps running phases until told to pause
    or until the game ends.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Engine surface
from diplomacy_engine import (
    POWERS, PROVINCES, initial_state,
    adjudicate_movement, adjudicate_retreats, adjudicate_adjustments,
    update_supply_centers, advance_phase, parse_order,
)

# Substrate surface
from diplomacy_kg_schema import MessageEvent
from diplomacy_agent_v2 import DiplomacyAgentV2, schema_phase_key
from diplomacy_engine_glue import (
    capture_move_events, capture_adjustment_events, capture_phase_state,
    distribute_messages,
)
from diplomacy_biopsy import (
    make_run_dir, write_agent_snapshot, append_board_state,
    append_messages, write_run_summary,
)


# ============================================================================
# Archetype assignments + briefs (mirrors run_v2.py)
# ============================================================================

DEFAULT_BRIEFS = {
    "MARSHAL_VEIL": ("I am Marshal Veil. I plan in arcs. I keep my word "
                     "when watched and remember when others do not."),
    "CARDINAL_FOX": ("I am Cardinal Fox. I trade in stories and what they "
                     "imply. I prefer a beautiful turn to a safe one."),
    "PARSON_HAWTHORNE": ("I am Parson Hawthorne. My word is given carefully "
                        "and kept absolutely."),
    "BARON_KORVIN": ("I am Baron Korvin. I trust no one before they have "
                    "earned it twice."),
    "ARCHITECT_LIRA": ("I am Architect Lira. I look at the whole table "
                      "and design the equilibrium I prefer."),
    "PLAYER_DEFAULT": "An LLM-driven Diplomacy player.",
}

DEFAULT_ARCHETYPES = {
    "AUSTRIA": "MARSHAL_VEIL",
    "ENGLAND": "PARSON_HAWTHORNE",
    "FRANCE":  "ARCHITECT_LIRA",
    "GERMANY": "BARON_KORVIN",
    "RUSSIA":  "CARDINAL_FOX",
    "TURKEY":  "PLAYER_DEFAULT",
}


# ============================================================================
# LLM caller factory
# ============================================================================

def _append_debug_log(path: str, prompt: str, response: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 72 + "\n")
            f.write(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("--- PROMPT ---\n" + prompt + "\n")
            f.write("--- RESPONSE ---\n" + response + "\n")
    except Exception:
        pass


def make_anthropic_caller(model: str, debug_log_path: Optional[str] = None
                          ) -> Callable[[str], str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    def _call(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                blocks = data.get("content", [])
                response = "\n".join(
                    b.get("text", "") for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                if debug_log_path:
                    _append_debug_log(debug_log_path, prompt, response)
                return response
        except Exception as e:
            print(f"  [anthropic error] {e}", file=sys.stderr)
            if debug_log_path:
                _append_debug_log(debug_log_path, prompt, f"[ERROR: {e}]")
            return "{}"
    return _call


def make_ollama_caller(url: str, model: str, timeout: float = 300.0,
                       debug_log_path: Optional[str] = None) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7},
        }).encode("utf-8")
        req = urllib.request.Request(
            url + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response = data.get("response", "")
                if debug_log_path:
                    _append_debug_log(debug_log_path, prompt, response)
                return response
        except Exception as e:
            print(f"  [ollama error] {e}", file=sys.stderr)
            if debug_log_path:
                _append_debug_log(debug_log_path, prompt, f"[ERROR: {e}]")
            return "{}"
    return _call


def make_stub_caller() -> Callable[[str], str]:
    """Deterministic JSON-emitting stub for testing without any model."""
    from run_v2 import StubLLM
    return StubLLM()


# ============================================================================
# Phase log entry — what happened during one phase (for the UI)
# ============================================================================

@dataclass
class PhaseLogEntry:
    phase: str               # schema phase key, e.g. "1901-SPRING-MOVES"
    phase_kind: str          # MOVEMENT / RETREAT / ADJUSTMENT
    started_at: float
    finished_at: Optional[float] = None
    n_messages: int = 0
    n_messages_with_commitspeak: int = 0
    n_orders: int = 0
    n_predictions: int = 0
    adjudication_lines: list[str] = field(default_factory=list)
    per_agent: dict = field(default_factory=dict)  # power -> {commitments_graded, beliefs_promoted, ...}
    sc_changes: list[dict] = field(default_factory=list)  # [{prov, from, to}]

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "phase_kind": self.phase_kind,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "n_messages": self.n_messages,
            "n_messages_with_commitspeak": self.n_messages_with_commitspeak,
            "n_orders": self.n_orders,
            "n_predictions": self.n_predictions,
            "adjudication_lines": self.adjudication_lines[:30],  # cap for UI
            "per_agent": self.per_agent,
            "sc_changes": self.sc_changes,
        }


# ============================================================================
# Live session
# ============================================================================

class LiveSession:
    """A single live substrate-driven game.

    Lifecycle:
        s = LiveSession()
        s.start(llm_kind='anthropic', llm_options={'model':'claude-haiku-4-5-20251001'})
        s.run_next_phase()        # blocking on caller; runs one phase
        s.start_auto_play()       # starts background loop
        s.pause_auto_play()
        s.snapshot_for(power, phase) -> dict
        s.public_state() -> dict
    """

    def __init__(self):
        self.lock = threading.Lock()

        self.started: bool = False
        self.engine_state = None
        self.agents: dict[str, DiplomacyAgentV2] = {}
        self.message_log: list[MessageEvent] = []

        # Phase telemetry
        self.completed_phases: list[str] = []     # in order of completion
        self.snapshots: dict[str, dict[str, dict]] = {}   # phase -> power -> snap
        self.board_states: dict[str, dict] = {}   # phase -> board dict
        self.messages_by_phase: dict[str, list[dict]] = {}  # phase -> messages
        self.phase_log: list[PhaseLogEntry] = []

        # Worker control
        self.worker: Optional[threading.Thread] = None
        self.auto_play: bool = False
        self.is_running_phase: bool = False
        self.current_phase_key: Optional[str] = None  # phase being computed
        self.last_error: Optional[str] = None

        # LLM config
        self.llm_kind: Optional[str] = None
        self.llm_options: dict = {}
        self.llm_call: Optional[Callable] = None

        # Archetype assignment
        self.archetype_assignment: dict[str, str] = {}

        # Biopsy directory (we still write to disk for posterity)
        self.run_dir: Optional[str] = None
        self.debug_log_path: Optional[str] = None

        # Termination
        self.max_phases: int = 24  # default cap so spectator runs don't go forever
        self.ended: bool = False
        self.end_reason: Optional[str] = None

    # --- start ----------------------------------------------------------

    def start(self, *, llm_kind: str, llm_options: dict,
              archetype_assignment: Optional[dict] = None,
              max_phases: int = 24,
              biopsy_dir: Optional[str] = None) -> dict:
        """Initialize a fresh game. Idempotent: if already started, returns
        a public-state snapshot instead of restarting."""
        with self.lock:
            if self.started:
                return self._public_state_locked()

            self.llm_kind = llm_kind
            self.llm_options = dict(llm_options or {})
            self.archetype_assignment = dict(archetype_assignment or DEFAULT_ARCHETYPES)
            self.max_phases = max_phases

            # Set up biopsy dir + debug log
            if biopsy_dir is None:
                biopsy_dir = make_run_dir()
            self.run_dir = biopsy_dir
            self.debug_log_path = os.path.join(biopsy_dir, "llm_debug.log")

            # Build LLM caller
            try:
                if llm_kind == "anthropic":
                    self.llm_call = make_anthropic_caller(
                        model=self.llm_options.get("model", "claude-haiku-4-5-20251001"),
                        debug_log_path=self.debug_log_path,
                    )
                elif llm_kind == "ollama":
                    self.llm_call = make_ollama_caller(
                        url=self.llm_options.get("url", "http://localhost:11434"),
                        model=self.llm_options.get("model", "gpt-oss:20b"),
                        timeout=float(self.llm_options.get("timeout", 300.0)),
                        debug_log_path=self.debug_log_path,
                    )
                elif llm_kind == "stub":
                    self.llm_call = make_stub_caller()
                else:
                    raise ValueError(f"unknown llm_kind: {llm_kind}")
            except Exception as e:
                self.last_error = f"llm setup failed: {e}"
                raise

            # Build agents
            self.agents = {
                power: DiplomacyAgentV2(
                    power=power, archetype=archetype,
                    character_brief_text=DEFAULT_BRIEFS.get(
                        archetype, DEFAULT_BRIEFS["PLAYER_DEFAULT"]),
                    llm_call=self.llm_call,
                    valid_powers=set(POWERS),
                    valid_provinces=set(PROVINCES.keys()),
                )
                for power, archetype in self.archetype_assignment.items()
            }

            # Initial engine state
            self.engine_state = initial_state()

            self.started = True
            self.last_error = None

            return self._public_state_locked()

    # --- run a phase ----------------------------------------------------

    def run_next_phase(self) -> dict:
        """Run one phase synchronously (caller blocks). Returns public state.

        Safe to call when auto_play is off and no other phase is running.
        Returns immediately with current state if game has ended.
        """
        with self.lock:
            if not self.started:
                return {"error": "not started"}
            if self.ended:
                return self._public_state_locked()
            if self.is_running_phase:
                return self._public_state_locked()
            self.is_running_phase = True
            self.current_phase_key = schema_phase_key(
                self.engine_state.year,
                self.engine_state.season,
                self.engine_state.phase)

        try:
            self._run_one_phase_unlocked()
        except Exception as e:
            with self.lock:
                self.last_error = f"phase failed: {e}"
                self.is_running_phase = False
                self.current_phase_key = None
            raise

        with self.lock:
            self.is_running_phase = False
            self.current_phase_key = None
            return self._public_state_locked()

    def _run_one_phase_unlocked(self) -> None:
        """The phase loop, faithful to run_v2.run_one_phase() but capturing
        rich telemetry into PhaseLogEntry / snapshots dicts.

        Runs WITHOUT the global lock held during LLM calls (which can take
        many seconds). Re-acquires the lock at the end to commit results.
        """
        # Snapshot pre-state for the phase log under brief lock
        with self.lock:
            state = self.engine_state
            agents = self.agents
            schema_phase = schema_phase_key(state.year, state.season, state.phase)
            entry = PhaseLogEntry(
                phase=schema_phase,
                phase_kind=state.phase,
                started_at=time.time(),
            )
            pre_sc = self._supply_centers_snapshot(state)

        pre_state = state
        all_messages_collected: list[MessageEvent] = []
        all_orders = []
        adjudication_log = []
        new_state = state

        if state.phase == "MOVEMENT":
            # 1. Negotiation round
            for power, agent in agents.items():
                if power in state.eliminated:
                    continue
                # Release lock during LLM call by simply not holding it
                # (we never grabbed it for the agent calls)
                msgs = agent.negotiate(state, self.message_log)
                all_messages_collected.extend(msgs)

            distribute_messages(all_messages_collected, agents)
            with self.lock:
                self.message_log.extend(all_messages_collected)
            entry.n_messages = len(all_messages_collected)
            entry.n_messages_with_commitspeak = sum(
                1 for m in all_messages_collected if m.commitspeak_tail)

            # 2. Orders
            n_predictions_total = 0
            for power, agent in agents.items():
                if power in state.eliminated:
                    continue
                order_strings, out = agent.decide_orders(state, self.message_log)
                for line in order_strings:
                    o = parse_order(power, line)
                    if o is not None:
                        all_orders.append(o)
                n_predictions_total += len(out.predictions)
            entry.n_orders = len(all_orders)
            entry.n_predictions = n_predictions_total

            # 3. Adjudicate
            new_state, adjudication_log = adjudicate_movement(state, all_orders)
            entry.adjudication_lines = list(adjudication_log)

            # 4. Capture events; absorb
            move_events = capture_move_events(
                orders=all_orders, pre_state=pre_state,
                post_state=new_state, adjudication_log=adjudication_log,
            )
            adj_events = capture_adjustment_events(
                pre_state=pre_state, post_state=new_state,
            )
            phase_state = capture_phase_state(new_state)

            peek = advance_phase(self._clone_state(new_state))
            next_schema = schema_phase_key(peek.year, peek.season, peek.phase)

            for power, agent in agents.items():
                if power in state.eliminated:
                    continue
                log = agent.absorb_phase_resolution(
                    resolved_phase=schema_phase,
                    move_events=move_events,
                    adjustment_events=adj_events,
                    phase_state=phase_state,
                    next_phase=next_schema,
                )
                entry.per_agent[power] = {
                    "commitments_graded": log.commitments_graded,
                    "predictions_graded": log.predictions_graded,
                    "beliefs_promoted":   log.beliefs_promoted,
                    "beliefs_retired":    log.beliefs_retired,
                    "intents_promoted":   log.intents_promoted,
                    "intents_retired":    log.intents_retired,
                    "revision_proposals_made": log.revision_proposals_made,
                    "near_term_synthesized":   log.near_term_synthesized,
                }

            self._write_biopsy_artifacts(
                kind="movement", schema_phase=schema_phase,
                pre_state=pre_state, post_state=new_state,
                messages=all_messages_collected,
            )

            new_state = advance_phase(new_state)

        elif state.phase == "RETREAT":
            for power, agent in agents.items():
                for line in agent.decide_retreats(state):
                    o = parse_order(power, line)
                    if o is not None:
                        all_orders.append(o)
            new_state, _retreat_log = adjudicate_retreats(state, all_orders)
            adj_events = capture_adjustment_events(
                pre_state=pre_state, post_state=new_state)
            phase_state = capture_phase_state(new_state)
            peek = advance_phase(self._clone_state(new_state))
            next_schema = schema_phase_key(peek.year, peek.season, peek.phase)
            for power, agent in agents.items():
                if power in state.eliminated:
                    continue
                agent.absorb_phase_resolution(
                    resolved_phase=schema_phase,
                    move_events=[], adjustment_events=adj_events,
                    phase_state=phase_state, next_phase=next_schema,
                )
            entry.adjudication_lines = ["retreats resolved (heuristic)"]
            self._write_biopsy_artifacts(
                kind="retreat", schema_phase=schema_phase,
                pre_state=pre_state, post_state=new_state, messages=[])
            new_state = advance_phase(new_state)

        elif state.phase == "ADJUSTMENT":
            update_supply_centers(state)
            new_state = state
            for power, agent in agents.items():
                if power in new_state.eliminated:
                    continue
                for line in agent.decide_builds(new_state):
                    o = parse_order(power, line)
                    if o is not None:
                        all_orders.append(o)
            new_state, _adj_log = adjudicate_adjustments(new_state, all_orders)
            adj_events = capture_adjustment_events(
                pre_state=pre_state, post_state=new_state)
            phase_state = capture_phase_state(new_state)
            peek = advance_phase(self._clone_state(new_state))
            next_schema = schema_phase_key(peek.year, peek.season, peek.phase)
            for power, agent in agents.items():
                if power in state.eliminated:
                    continue
                agent.absorb_phase_resolution(
                    resolved_phase=schema_phase,
                    move_events=[], adjustment_events=adj_events,
                    phase_state=phase_state, next_phase=next_schema,
                )
            entry.adjudication_lines = [f"adjustments resolved: {len(adj_events)} events"]
            # End-of-year is a natural snapshot point — write artifacts
            self._write_biopsy_artifacts(
                kind="adjustment", schema_phase=schema_phase,
                pre_state=pre_state, post_state=new_state, messages=[])
            new_state = advance_phase(new_state)
        else:
            # unknown phase kind; just advance
            new_state = advance_phase(state)

        # SC delta
        post_sc = self._supply_centers_snapshot(new_state)
        entry.sc_changes = self._sc_diff(pre_sc, post_sc)
        entry.finished_at = time.time()

        # Commit: install new state and snapshots under lock
        with self.lock:
            self.engine_state = new_state
            self.completed_phases.append(schema_phase)
            self.phase_log.append(entry)
            # Snapshots: read back what write_agent_snapshot just wrote to disk.
            # This keeps us in lockstep with the biopsy format (so live and
            # offline viewers render identically).
            self.snapshots.setdefault(schema_phase, {})
            for power, agent in self.agents.items():
                snap_dict = self._read_snapshot_from_disk(schema_phase, power)
                if snap_dict is not None:
                    self.snapshots[schema_phase][power] = snap_dict
            # Cache board state and messages for this phase
            self.board_states[schema_phase] = self._board_dict_locked(new_state)
            self.messages_by_phase[schema_phase] = [
                self._message_to_dict(m) for m in all_messages_collected]
            # Termination check
            if len(self.completed_phases) >= self.max_phases:
                self.ended = True
                self.end_reason = f"max phases reached ({self.max_phases})"
            elif len([p for p in POWERS if not self._is_eliminated(new_state, p)]) <= 1:
                self.ended = True
                self.end_reason = "only one power remains"

    # --- auto-play ------------------------------------------------------

    def start_auto_play(self) -> dict:
        with self.lock:
            if not self.started or self.ended:
                return self._public_state_locked()
            if self.auto_play:
                return self._public_state_locked()
            self.auto_play = True
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._auto_loop, daemon=True)
                self.worker.start()
            return self._public_state_locked()

    def pause_auto_play(self) -> dict:
        with self.lock:
            self.auto_play = False
            return self._public_state_locked()

    def _auto_loop(self):
        while True:
            with self.lock:
                if not self.auto_play or self.ended:
                    break
                if self.is_running_phase:
                    pass
                else:
                    self.is_running_phase = True
                    self.current_phase_key = schema_phase_key(
                        self.engine_state.year,
                        self.engine_state.season,
                        self.engine_state.phase)
            try:
                self._run_one_phase_unlocked()
            except Exception as e:
                with self.lock:
                    self.last_error = f"phase failed: {e}"
                    self.auto_play = False
                    self.is_running_phase = False
                    self.current_phase_key = None
                break
            with self.lock:
                self.is_running_phase = False
                self.current_phase_key = None
            # Brief sleep so the UI can poll a "settled" state between phases
            time.sleep(0.4)

    # --- accessors ------------------------------------------------------

    def public_state(self) -> dict:
        with self.lock:
            return self._public_state_locked()

    def _public_state_locked(self) -> dict:
        if not self.started or self.engine_state is None:
            return {
                "started": False,
                "ended": False,
                "auto_play": False,
                "is_running_phase": False,
                "completed_phases": [],
                "current_phase": None,
                "computing_phase": None,
                "last_error": self.last_error,
            }
        s = self.engine_state
        cur = schema_phase_key(s.year, s.season, s.phase)
        return {
            "started": self.started,
            "ended": self.ended,
            "end_reason": self.end_reason,
            "auto_play": self.auto_play,
            "is_running_phase": self.is_running_phase,
            "completed_phases": list(self.completed_phases),
            "current_phase": cur,                 # next phase to be run
            "computing_phase": self.current_phase_key if self.is_running_phase else None,
            "last_error": self.last_error,
            "max_phases": self.max_phases,
            "llm_kind": self.llm_kind,
            "llm_model": self.llm_options.get("model"),
            "archetype_assignment": dict(self.archetype_assignment),
            "run_dir": self.run_dir,
        }

    def snapshots_payload(self) -> dict:
        with self.lock:
            return {
                "snapshots": {
                    p: {pw: snap for pw, snap in by_pw.items()}
                    for p, by_pw in self.snapshots.items()
                },
            }

    def board_payload(self) -> dict:
        with self.lock:
            # Always include current (in-progress) board too, so the UI can
            # render the state even before any phase has been completed.
            current = (
                self._board_dict_locked(self.engine_state)
                if self.engine_state else {})
            cur_key = (schema_phase_key(self.engine_state.year,
                                        self.engine_state.season,
                                        self.engine_state.phase)
                       if self.engine_state else None)
            board = dict(self.board_states)
            if cur_key and cur_key not in board:
                board[cur_key] = current
            return {"board": board}

    def messages_payload(self) -> dict:
        with self.lock:
            return {"messages": dict(self.messages_by_phase)}

    def log_payload(self) -> dict:
        with self.lock:
            return {"log": [e.to_dict() for e in self.phase_log]}

    def summary_payload(self) -> dict:
        """Mimics the post-run summary structure used by the viewer."""
        with self.lock:
            agents_summary = {}
            # Use the most recent cached snapshot per power (or fall back to
            # an empty summary if no phase has completed yet).
            for power, agent in self.agents.items():
                snap = self._latest_snapshot_for_power(power)
                s = (snap or {}).get("summary", {})
                # Compute self-commitment kept/broken from the agent's mind directly
                kept_self = sum(
                    1 for c in agent.mind.self_commitments.values()
                    if c.status.value == "kept")
                broken_self = sum(
                    1 for c in agent.mind.self_commitments.values()
                    if c.status.value == "broken")
                agents_summary[power] = {
                    "archetype": (snap or {}).get("archetype", agent.mind.archetype),
                    "final_sc": self._count_sc_for(self.engine_state, power),
                    "messages_total": sum(
                        1 for m in self.message_log
                        if m.sender == power
                        or power in m.recipients
                        or m.public),
                    "beliefs": {
                        "active":  s.get("beliefs_active", 0),
                        "proto":   s.get("beliefs_proto", 0),
                        "retired": s.get("beliefs_retired", 0),
                    },
                    "intents": {
                        "active":    s.get("intents_active", 0),
                        "succeeded": s.get("intents_succeeded", 0),
                        "failed":    0,
                        "retired":   0,
                    },
                    "incoming_commitments": {
                        "kept":    s.get("incoming_commitments_kept", 0),
                        "broken":  s.get("incoming_commitments_broken", 0),
                        "pending": s.get("incoming_commitments_pending", 0),
                    },
                    "self_commitments": {
                        "kept":   kept_self,
                        "broken": broken_self,
                    },
                    "predictions": {
                        "confirmed": s.get("predictions_confirmed", 0),
                        "refuted":   s.get("predictions_refuted", 0),
                        "open":      s.get("predictions_open", 0),
                        "partial":   0,
                    },
                }
            return {
                "run_metadata": {
                    "llm_kind":  self.llm_kind,
                    "model":     self.llm_options.get("model"),
                    "phases_run": len(self.completed_phases),
                    "ended": self.ended,
                    "end_reason": self.end_reason,
                },
                "agents": agents_summary,
            }

    def _latest_snapshot_for_power(self, power: str) -> Optional[dict]:
        # Walk completed_phases backwards to find this power's most recent snap
        for phase in reversed(self.completed_phases):
            snap = self.snapshots.get(phase, {}).get(power)
            if snap is not None:
                return snap
        return None

    def _read_snapshot_from_disk(self, schema_phase: str, power: str) -> Optional[dict]:
        if not self.run_dir:
            return None
        path = os.path.join(self.run_dir, f"{power}_{schema_phase}.json")
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _clone_state(state):
        return copy.copy(state)

    @staticmethod
    def _is_eliminated(state, power: str) -> bool:
        return power in getattr(state, "eliminated", set())

    def _supply_centers_snapshot(self, state) -> dict:
        """{prov: power} for every owned SC. Pre/post comparison for sc_changes."""
        return dict(getattr(state, "sc_owner", {}) or {})

    def _sc_diff(self, pre: dict, post: dict) -> list[dict]:
        diffs = []
        keys = set(pre.keys()) | set(post.keys())
        for k in sorted(keys):
            a, b = pre.get(k), post.get(k)
            if a != b:
                diffs.append({"prov": k, "from": a, "to": b})
        return diffs

    def _count_sc_for(self, state, power: str) -> int:
        if state is None:
            return 0
        sc_owner = getattr(state, "sc_owner", {}) or {}
        return sum(1 for owner in sc_owner.values() if owner == power)

    def _board_dict_locked(self, state) -> dict:
        """Format equivalent to the viewer's parsed board: per-power SC + units."""
        out = {}
        if state is None:
            return out
        sc_owner = getattr(state, "sc_owner", {}) or {}
        for power in POWERS:
            sc_count = sum(1 for owner in sc_owner.values() if owner == power)
            units = []
            for u in getattr(state, "units", []):
                if getattr(u, "power", None) == power:
                    units.append({"kind": u.kind, "prov": u.location})
            out[power] = {"sc_count": sc_count, "units": units}
        return out

    def _message_to_dict(self, m: MessageEvent) -> dict:
        return {
            "from": m.sender,
            "to": list(m.recipients) if m.recipients else (["ALL"] if m.public else []),
            "text": m.body + (("\n[[commit\n" + m.commitspeak_tail + "\n]]")
                              if m.commitspeak_tail else ""),
            "has_commitspeak": bool(m.commitspeak_tail),
        }

    def _write_biopsy_artifacts(self, *, kind: str, schema_phase: str,
                                pre_state, post_state, messages):
        if not self.run_dir:
            return
        try:
            phase_label = f"{pre_state.year}_{pre_state.season}_{pre_state.phase}"
            if messages:
                append_messages(messages, self.run_dir, phase_label=phase_label)
            append_board_state(post_state, self.run_dir, phase_label=phase_label)
            for power, agent in self.agents.items():
                if power in pre_state.eliminated:
                    continue
                write_agent_snapshot(agent.mind, schema_phase, self.run_dir)
        except Exception as e:
            print(f"  [biopsy write failed] {e}", file=sys.stderr)
