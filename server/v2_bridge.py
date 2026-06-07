"""
server/v2_bridge.py — adapts DiplomacyAgentV2 to fit the legacy
session.py interface.

Why this exists
---------------
session.py was written against the LLMAgent (V1) interface:

    agent.negotiate(state, visible_messages: list[Message]) -> list[Message]
    agent.decide_orders(state, visible_messages) -> obj_with(.orders, .reflection_notes)
    agent.decide_retreats(state) -> list[Order]
    agent.decide_builds(state) -> list[Order]
    agent.kgs.graphs[name].to_dict()
    agent.personality_key, agent.model

DiplomacyAgentV2 has a similar but not identical surface:

    agent.negotiate(state, recent_messages) -> list[MessageEvent]
    agent.decide_orders(state, recent_messages) -> tuple[list[str], OrdersOutput]
    agent.decide_retreats(state) -> list[str]
    agent.decide_builds(state) -> list[str]
    agent.mind  (a typed AgentMind, not a graph dict)

This module wraps a DiplomacyAgentV2 in a thin facade that presents the
LLMAgent-shaped interface, so session.py changes are minimal.

Design choices
--------------
1. Messages flowing through session.messages stay as legacy Message dataclass
   instances. Conversion happens at the V2 boundary only.
2. Orders flowing back are converted from V2's `list[str]` (engine command
   strings) to engine `Order` objects via `parse_order`.
3. A `.kgs` shim provides the legacy KG accessor for any code that still
   reads agent.kgs.graphs[<name>] — it returns empty dicts for legacy graph
   names (personality / soul / etc.) and a 'theory_of_mind' graph backed by
   the actual mind. The end-of-game reveal works without crashing.
4. The bridge exposes the underlying V2 agent as `.v2` for direct access
   (used by the snapshot writer and the inspector route).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from diplomacy_engine import POWERS, PROVINCES, parse_order, Order
from diplomacy_kg_schema import MessageEvent, new_id
from diplomacy_agent_v2 import DiplomacyAgentV2, schema_phase_key
from diplomacy_commitspeak import extract_block

# Auto-enable fovea v2 (richer prompt content, plus the journal/suspicions
# render hook). The activation is idempotent so import order doesn't matter.
try:
    from diplomacy_fovea_v2 import enable_fovea_v2
    enable_fovea_v2()
except Exception as _e:
    print(f"[bridge] fovea v2 enable failed (non-fatal): {_e}")
try:
    from diplomacy_message_compress import enable_compressed_messages
    enable_compressed_messages()
except Exception as _e:
    print(f"[bridge] message compression enable failed (non-fatal): {_e}")

from agents import Message  # legacy Message dataclass


# ----------------------------------------------------------------------
# Default character briefs per archetype. These match run_v2's defaults
# so the bridge produces identical agents to what the eval harness uses.
# ----------------------------------------------------------------------

DEFAULT_BRIEFS = {
    "MARSHAL_VEIL": (
        "I am Marshal Veil. I plan in arcs. I keep my word "
        "when watched and remember when others do not."),
    "CARDINAL_FOX": (
        "I am Cardinal Fox. I trade in stories and what they "
        "imply. I prefer a beautiful turn to a safe one."),
    "PARSON_HAWTHORNE": (
        "I am Parson Hawthorne. My word is given carefully "
        "and kept absolutely."),
    "BARON_KORVIN": (
        "I am Baron Korvin. I trust no one before they have "
        "earned it twice."),
    "ARCHITECT_LIRA": (
        "I am Architect Lira. I look at the whole table "
        "and design the equilibrium I prefer."),
    "PLAYER_DEFAULT": "An LLM-driven Diplomacy player.",
}


# ----------------------------------------------------------------------
# Message conversion
# ----------------------------------------------------------------------


def legacy_to_event(m: Message, phase: str, sender_idx: int = 0) -> MessageEvent:
    """Convert legacy Message → MessageEvent (what V2 expects).

    Critically, this extracts the commitspeak_tail from the body so
    intake_message() will actually parse it. Without this the entire
    incoming-commitment pipeline silently no-ops.
    """
    body = m.text or ""
    tail = extract_block(body)  # returns the [[commit ...]] block text or None
    return MessageEvent(
        id=new_id("msg"),
        phase=phase,
        sender=m.sender,
        recipients=list(m.recipients or []),
        public=bool(m.public),
        body=body,
        commitspeak_tail=tail,
        sent_at=time.time() + sender_idx * 0.001,  # preserve order
    )


def event_to_legacy(e: MessageEvent, season: str, year: int) -> Message:
    """Convert MessageEvent → legacy Message (what session.messages stores)."""
    return Message(
        sender=e.sender,
        recipients=list(e.recipients or []),
        text=e.body or "",
        season=season,
        year=year,
        public=bool(e.public),
    )


# ----------------------------------------------------------------------
# Orders shim — legacy decide_orders returned an object with .orders /
# .reflection_notes attrs. V2 returns (list[str], OrdersOutput).
# ----------------------------------------------------------------------


class _OrdersResult:
    """Mimics LLMAgent's decide_orders return shape.
    Attributes: .orders (list[Order]), .reflection_notes (list[str])."""
    def __init__(self, orders: list[Order], notes: list[str]):
        self.orders = orders
        self.reflection_notes = notes


# ----------------------------------------------------------------------
# Legacy-graph shim
# ----------------------------------------------------------------------
# session.end_game_reveal references agent.kgs.graphs["theory_of_mind"] and
# agent.kgs.graphs["counterfactuals"] for the user-trust readout. We give
# it the minimum surface so it doesn't crash. Everything is empty.


class _EmptyGraph:
    def __init__(self):
        self.nodes = {}
    def to_dict(self):
        return {"nodes": [], "edges": []}


class _KgsShim:
    """Minimal stand-in for the legacy KGs object so the end-of-game reveal
    still runs. Returns empty graphs for any legacy graph name."""
    def __init__(self):
        self.graphs = {
            "personality": _EmptyGraph(),
            "soul": _EmptyGraph(),
            "ethics": _EmptyGraph(),
            "theory_of_mind": _EmptyGraph(),
            "strategy": _EmptyGraph(),
            "counterfactuals": _EmptyGraph(),
        }
    def to_dict(self):
        return {g: graph.to_dict() for g, graph in self.graphs.items()}


# ----------------------------------------------------------------------
# The bridge agent
# ----------------------------------------------------------------------


class V2BridgeAgent:
    """Wraps a DiplomacyAgentV2 to present the LLMAgent-shaped interface.

    Attributes available to session.py:
      .power, .archetype                    — strings
      .personality_key                      — alias for archetype (legacy name)
      .model                                — model string for the health log
      .kgs                                   — empty graphs shim (for end-game reveal)
      .v2                                    — the wrapped DiplomacyAgentV2
      .negotiate(state, visible_messages)    — returns list[Message]
      .decide_orders(state, visible_messages) → object with .orders, .reflection_notes
      .decide_retreats(state) → list[Order]
      .decide_builds(state) → list[Order]
    """

    def __init__(
        self, *,
        power: str,
        archetype: str,
        llm_call,
        model_label: str = "claude-haiku-4-5-20251001",
        muted: bool = False,
    ):
        self.power = power
        self.archetype = archetype
        self.personality_key = archetype          # legacy alias
        self.model = model_label                   # for health-check log only
        self.muted = bool(muted)
        self.kgs = _KgsShim()
        # If muted, use MutableAgent (DiplomacyAgentV2 subclass with the
        # kg_advisory_mute switch). Otherwise use plain V2.
        if muted:
            from diplomacy_mute import MutableAgent
            self.v2 = MutableAgent(
                power=power,
                archetype=archetype,
                character_brief_text=DEFAULT_BRIEFS.get(
                    archetype, DEFAULT_BRIEFS["PLAYER_DEFAULT"]),
                llm_call=llm_call,
                valid_powers=set(POWERS),
                valid_provinces=set(PROVINCES.keys()),
                kg_advisory_mute=True,
            )
        else:
            self.v2 = DiplomacyAgentV2(
                power=power,
                archetype=archetype,
                character_brief_text=DEFAULT_BRIEFS.get(
                    archetype, DEFAULT_BRIEFS["PLAYER_DEFAULT"]),
                llm_call=llm_call,
                valid_powers=set(POWERS),
                valid_provinces=set(PROVINCES.keys()),
            )
        # Attach the prompt recorder so the BIOPSY route can see exactly
        # what prompts went to the LLM and what came back. Capacity 60 ≈
        # several phases worth of calls (negotiate + orders + revisions).
        from diplomacy_prompt_recorder import attach_recorder
        try:
            attach_recorder(self.v2, capacity=60)
        except Exception as e:
            print(f"[bridge] recorder attach failed for {power}: {e}")

    # ---- engine-facing methods ----

    def negotiate(self, state, visible: list[Message]) -> list[Message]:
        phase = schema_phase_key(state.year, state.season, state.phase)
        # Convert legacy Message → MessageEvent for V2 input
        recent_events = [
            legacy_to_event(m, phase, i) for i, m in enumerate(visible)
        ]
        new_events = self.v2.negotiate(state, recent_events)
        # Convert V2 MessageEvent → legacy Message for session.messages
        return [event_to_legacy(e, state.season, state.year) for e in new_events]

    def decide_orders(self, state, visible: list[Message]):
        phase = schema_phase_key(state.year, state.season, state.phase)
        recent_events = [
            legacy_to_event(m, phase, i) for i, m in enumerate(visible)
        ]
        order_strs, _output = self.v2.decide_orders(state, recent_events)
        # Convert engine command strings to Order objects
        orders: list[Order] = []
        bad: list[str] = []
        for s in order_strs:
            o = parse_order(self.power, s)
            if o is not None:
                orders.append(o)
            else:
                bad.append(s)
        notes = []
        if bad:
            notes.append(f"could not parse: {bad}")
        return _OrdersResult(orders, notes)

    def decide_retreats(self, state) -> list[Order]:
        out_strs = self.v2.decide_retreats(state)
        result: list[Order] = []
        for s in out_strs:
            o = parse_order(self.power, s)
            if o is not None:
                result.append(o)
        return result

    def decide_builds(self, state) -> list[Order]:
        out_strs = self.v2.decide_builds(state)
        result: list[Order] = []
        for s in out_strs:
            o = parse_order(self.power, s)
            if o is not None:
                result.append(o)
        return result


# ----------------------------------------------------------------------
# Substrate pipeline helpers — distribute messages, absorb phase resolution.
# These are the THREE GRADER PIPELINES that were silently no-oping in the
# 12-year game we analyzed. Without them:
#   - incoming_commitments stays empty (parser never extracts inbound promises)
#   - self_commitments stays PENDING forever (grader never marks them)
#   - predictions stays OPEN forever (grader never marks them)
# ----------------------------------------------------------------------


def distribute_messages_to_agents(
    sent_messages: list[Message],     # legacy Message objects from session.messages
    agents: dict[str, "V2BridgeAgent"],
    phase_str: str,
) -> dict[str, int]:
    """For each newly-sent message, route it to every recipient's V2 agent
    via intake_message(). Returns count of new commitments registered per
    recipient power (telemetry).

    Agents that don't expose intake_message (e.g. RawLLMAgent stubs) are
    silently skipped — no substrate to write into is fine.

    Call this AFTER messages have been added to session.messages but
    before the next phase begins, so each agent's mind has the inbound
    commitments registered and available for grading.
    """
    counts = {p: 0 for p in agents}
    for i, m in enumerate(sent_messages):
        ev = legacy_to_event(m, phase_str, sender_idx=i)
        if ev.public:
            for power, bridge in agents.items():
                if power == ev.sender:
                    continue
                v2 = getattr(bridge, "v2", bridge)
                if hasattr(v2, "intake_message"):
                    try:
                        counts[power] += v2.intake_message(ev)
                    except Exception:
                        pass
        else:
            for r in ev.recipients:
                if r in agents and r != ev.sender:
                    v2 = getattr(agents[r], "v2", agents[r])
                    if hasattr(v2, "intake_message"):
                        try:
                            counts[r] += v2.intake_message(ev)
                        except Exception:
                            pass
    return counts


def absorb_phase_for_all_agents(
    agents: dict[str, "V2BridgeAgent"],
    pre_state,
    post_state,
    orders_executed: list,
    adjudication_log: list[str],
    resolved_phase_str: str,
    next_phase_str: str,
) -> dict[str, dict]:
    """Capture move/adjustment/phase events from the engine adjudication
    and feed them into every agent's absorb_phase_resolution(). This is
    where commitment grading + prediction grading + belief lifecycle
    actually run. Returns per-power telemetry.

    Failures in one agent don't abort the others; they're isolated.
    """
    from diplomacy_engine_glue import (
        capture_move_events, capture_adjustment_events, capture_phase_state,
    )
    try:
        move_events = capture_move_events(
            orders=orders_executed, pre_state=pre_state,
            post_state=post_state, adjudication_log=adjudication_log,
        )
    except Exception as e:
        print(f"[grader] capture_move_events failed: {e}")
        move_events = []
    try:
        adj_events = capture_adjustment_events(
            pre_state=pre_state, post_state=post_state,
        )
    except Exception as e:
        print(f"[grader] capture_adjustment_events failed: {e}")
        adj_events = []
    try:
        phase_state = capture_phase_state(post_state)
    except Exception as e:
        print(f"[grader] capture_phase_state failed: {e}")
        return {}

    telemetry = {}
    for power, bridge in agents.items():
        if power in post_state.eliminated:
            continue
        try:
            log = bridge.v2.absorb_phase_resolution(
                resolved_phase=resolved_phase_str,
                move_events=move_events,
                adjustment_events=adj_events,
                phase_state=phase_state,
                next_phase=next_phase_str,
            )
            telemetry[power] = {
                "commitments_graded": getattr(log, "commitments_graded", 0),
                "predictions_graded": getattr(log, "predictions_graded", 0),
                "beliefs_promoted": getattr(log, "beliefs_promoted", 0),
                "beliefs_retired": getattr(log, "beliefs_retired", 0),
            }
        except Exception as e:
            print(f"[grader] absorb_phase_resolution failed for {power}: {e}")
            telemetry[power] = {"_error": str(e)}

    # PRIVATE JOURNAL — one extra LLM call per agent per phase. Captures
    # the agent's inner narrative (suspicions, reactions, hypotheses) into
    # mind.private_journal. Failures are non-fatal.
    try:
        from diplomacy_private_thoughts import write_journal_entry
        what_happened = _summarize_phase_for_journal(
            pre_state, post_state, move_events, adjudication_log)
        for power, bridge in agents.items():
            if power in post_state.eliminated:
                continue
            try:
                board_summary = bridge.v2._board_summary(post_state)
                write_journal_entry(
                    bridge.v2,
                    board_summary_text=board_summary,
                    what_just_happened=what_happened,
                    phase=resolved_phase_str,
                    kind="post-resolution",
                )
            except Exception as e:
                print(f"[journal] failed for {power}: {e}")
    except Exception as e:
        print(f"[journal] module load failed: {e}")

    # DREAMING — once per game year, after FALL ADJUSTMENT. The agent
    # consolidates beliefs across the year, reviews credibility data,
    # spots cross-belief patterns. One LLM call per agent per year.
    is_year_end = (
        getattr(pre_state, "season", "") == "FALL"
        and getattr(pre_state, "phase", "") == "ADJUSTMENT"
    )
    if is_year_end:
        try:
            from diplomacy_dreaming import run_dream
            year = pre_state.year
            for power, bridge in agents.items():
                if power in post_state.eliminated:
                    continue
                try:
                    board_summary = bridge.v2._board_summary(post_state)
                    run_dream(bridge.v2, year=year,
                              board_summary_text=board_summary)
                except Exception as e:
                    print(f"[dream] failed for {power}: {e}")
        except Exception as e:
            print(f"[dream] module load failed: {e}")

    return telemetry


def _summarize_phase_for_journal(pre_state, post_state, move_events,
                                  adjudication_log) -> str:
    """Build a short factual summary of what happened this phase, for
    the journal prompt's 'WHAT JUST HAPPENED' section. Pure data — no
    LLM call here."""
    from diplomacy_engine import POWERS, supply_centers_owned
    lines = []
    # SC delta
    pre_scs = {p: len(supply_centers_owned(pre_state, p)) for p in POWERS}
    post_scs = {p: len(supply_centers_owned(post_state, p)) for p in POWERS}
    deltas = {p: post_scs[p] - pre_scs[p] for p in POWERS}
    if any(d != 0 for d in deltas.values()):
        bits = []
        for p, d in deltas.items():
            if d != 0:
                sign = "+" if d > 0 else ""
                bits.append(f"{p}{sign}{d}")
        lines.append("SC changes: " + ", ".join(bits))
    # Eliminations
    new_elim = post_state.eliminated - pre_state.eliminated
    if new_elim:
        lines.append("Eliminated: " + ", ".join(sorted(new_elim)))
    # Move highlights — successful captures and failed attacks
    successful_moves = [m for m in move_events
                        if m.order_type == "MOVE" and m.result == "success"
                        and m.target]
    bounced = [m for m in move_events
               if m.order_type == "MOVE" and m.result == "bounced"]
    if successful_moves:
        bits = [f"{m.power[:3]} {m.unit_kind}{m.origin}→{m.target}"
                for m in successful_moves[:8]]
        lines.append("Successful moves: " + ", ".join(bits))
    if bounced:
        bits = [f"{m.power[:3]} {m.unit_kind}{m.origin}→{m.target}"
                for m in bounced[:6]]
        lines.append("Bounced: " + ", ".join(bits))
    # Final adjudication log (a few lines)
    if adjudication_log:
        tail = adjudication_log[-5:]
        lines.append("Adjudication tail: " + " | ".join(tail))
    return "\n".join(lines) if lines else "(quiet phase — nothing decisive)"


# ----------------------------------------------------------------------
# Default LLM caller — Anthropic Haiku 4.5
# ----------------------------------------------------------------------


def make_default_llm_call(model: str = None, kind: str = None):
    """Returns an llm_call(prompt: str) -> str for whichever provider is selected
    or auto-detected (Ollama / Anthropic / OpenAI / Google). See llm_providers.

    Raises RuntimeError early if NO provider is available, so the session can show
    a clear error in the log instead of failing on the first turn.
    """
    caller, _kind, _model = make_default_llm_call_labeled(kind, model)
    return caller


def make_default_llm_call_labeled(kind: str = None, model: str = None):
    """Like make_default_llm_call but also returns the resolved (kind, model)
    so the session can label agents and the health log correctly.
    Returns (caller, kind, model)."""
    import llm_providers
    return llm_providers.build(kind, model)
