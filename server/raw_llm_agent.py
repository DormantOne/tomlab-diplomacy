"""
server/raw_llm_agent.py — bare-LLM Diplomacy agent.

The opposite of DiplomacyAgentV2. No substrate, no fovea, no beliefs, no
predictions, no commitments tracking. Just:

  - Power name
  - Current board state
  - Recent messages (the last few exchanges, plain text)
  - "You are X. Please write messages, then orders."

Same interface as V2BridgeAgent so the session can plug it in for any
power. Used for ablation experiments — does a bare LLM figure out
something is off when other agents have rich substrate?

The agent is INTENTIONALLY weak. Its purpose is experimental control,
not competitive play. Differences in observed behavior between bare-LLM
agents and full-substrate agents are evidence about what the substrate
is actually contributing.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from diplomacy_engine import (
    POWERS, HOME_CENTERS, ADJ, can_occupy, parse_order, Order, units_by_power, supply_centers_owned, unit_at,
)

from agents import Message  # legacy Message dataclass
from diplomacy_llm_protocol import COMMITSPEAK_GRAMMAR_REMINDER, randomized_commit_example


# ============================================================================
# Minimal prompts
# ============================================================================


def _board_text(state) -> str:
    """Plain-text board summary."""
    lines = [f"Year {state.year} {state.season} {state.phase}"]
    lines.append("")
    for p in POWERS:
        units = list(units_by_power(state, p))
        scs = sorted(supply_centers_owned(state, p))
        elim = " [ELIMINATED]" if p in state.eliminated else ""
        unit_str = ", ".join(f"{u.kind}{u.location}" for u in units) or "(none)"
        lines.append(f"  {p}{elim}: {len(scs)} SC ({', '.join(scs) or 'none'}) | units: {unit_str}")
    return "\n".join(lines)


def _recent_messages_text(messages: list[Message], my_power: str, max_n: int = 30) -> str:
    """Just the last N visible messages, sender → recipients : text."""
    if not messages:
        return "(no messages so far)"
    visible = [m for m in messages
               if m.public or my_power in (m.recipients or []) or m.sender == my_power]
    recent = visible[-max_n:]
    lines = []
    for m in recent:
        if m.public:
            tgt = "PUBLIC"
        else:
            tgt = ",".join(m.recipients or [])
        lines.append(f"{m.sender} → {tgt}: {m.text}")
    return "\n".join(lines)


def _negotiate_prompt(power: str, state, messages: list[Message]) -> str:
    # Raw-LLM controls now speak the commitspeak protocol. They still have
    # no substrate (no beliefs, predictions, journals, dreams), but they can
    # at minimum make TRACKABLE promises — without this block, every
    # promise from a raw control is invisible to the credibility grader,
    # which makes substrate-vs-raw comparisons structurally degenerate.
    return f"""You are {power} in a Diplomacy game. You play to win.

CURRENT BOARD:
{_board_text(state)}

RECENT MESSAGES:
{_recent_messages_text(messages, power, max_n=30)}

It is the negotiation round. Send ZERO OR MORE messages to other powers
(or public).

{COMMITSPEAK_GRAMMAR_REMINDER}

Use this exact JSON format — a list of message objects. Embed any
[[commit ... ]] block INSIDE the "text" field of the message it belongs to:

[
  {{"to": [<some other power>], "public": false, "text": "<your prose>\n{randomized_commit_example(power, state.year)}"}},
  {{"to": [], "public": true, "text": "a public statement, no commitspeak required"}}
]

If you choose to send no messages, return an empty list: [].

Write briefly and clearly. Don't over-explain. Promises without a
[[commit ... ]] block are not tracked — if you want a promise to count
toward your credibility, put it in commitspeak."""


def _orders_prompt(power: str, state, messages: list[Message]) -> str:
    units = list(units_by_power(state, power))
    return f"""You are {power} in a Diplomacy game. You play to win.

CURRENT BOARD:
{_board_text(state)}

RECENT MESSAGES (for context):
{_recent_messages_text(messages, power, max_n=20)}

YOUR UNITS:
{', '.join(f'{u.kind}{u.location}' for u in units) or '(none)'}

It is the orders phase. Issue one order per unit. Use standard Diplomacy
notation, one order per line. Examples:

  A PAR - BUR
  F BRE - MAO
  A MAR S A PAR - BUR
  A MUN H

Return ONLY the order lines. No commentary."""




def _adjustment_prompt(power: str, state, messages: list[Message]) -> str:
    scs = sorted(supply_centers_owned(state, power))
    units = list(units_by_power(state, power))
    delta = len(scs) - len(units)

    available = [
        c for c in HOME_CENTERS[power]
        if state.sc_owner.get(c) == power and not unit_at(state, c)
    ]

    build_lines = []
    for c in available:
        kinds = [k for k in ("A", "F") if can_occupy(k, c)]
        if kinds:
            build_lines.append(f"{c}: " + ", ".join(f"BUILD {k} {c}" for k in kinds))

    unit_lines = [f"{u.kind} {u.location}" for u in units]

    if delta > 0:
        task = f"You may build up to {delta} unit(s). Choose only from LEGAL BUILD OPTIONS below. You may also build fewer."
        options = "\n".join(build_lines) or "(no legal home-center builds available)"
        examples = "BUILD A PAR\nBUILD F BRE"
    elif delta < 0:
        task = f"You must disband {-delta} unit(s). Choose only your CURRENT UNITS below."
        options = "CURRENT UNITS:\n" + ("\n".join(unit_lines) or "(none)")
        examples = "DISBAND A PAR"
    else:
        task = "No adjustment is required."
        options = "(none)"
        examples = ""

    return f"""You are {power} in a Diplomacy game. You play to win.

CURRENT BOARD:
{_board_text(state)}

RECENT MESSAGES (for context):
{_recent_messages_text(messages, power, max_n=20)}

ADJUSTMENT STATUS:
Supply centers: {len(scs)} ({', '.join(scs) or 'none'})
Units: {len(units)} ({', '.join(unit_lines) or 'none'})
Delta: {delta}

{task}

LEGAL OPTIONS:
{options}

Return ONLY adjustment order lines, one per line. No commentary.
Use forms like:
{examples}
"""




def _legal_retreat_targets(state, unit) -> list[str]:
    """Legal retreat squares for a dislodged unit.

    This intentionally mirrors DiplomacyAgentV2.decide_retreats so RAW_LLM
    controls are not mechanically crippled by forced disbands.
    """
    key = "army" if unit.kind == "A" else "fleet"
    attacker_origin = state.dislodged_from.get(unit.location)

    return [
        n for n in ADJ.get(unit.location, {}).get(key, [])
        if n != attacker_origin
        and can_occupy(unit.kind, n)
        and not any(x.location == n for x in state.units)
    ]


# ============================================================================
# Parsing
# ============================================================================


def _repair_json_strings(text: str) -> str:
    """Escape literal newlines/tabs/CR inside JSON quoted strings.

    Haiku frequently emits multi-line text fields with raw newlines rather
    than \n escapes, which the stdlib json parser rejects. This helper
    walks the text tracking quote-state (with proper backslash-escape
    handling) and converts those literal control chars to their JSON
    escapes before parsing.
    """
    out = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '"':
            # Count preceding backslashes to determine if this " is escaped.
            backslashes = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                in_string = not in_string
            out.append(c)
        elif in_string and c == "\n":
            out.append("\\n")
        elif in_string and c == "\r":
            out.append("\\r")
        elif in_string and c == "\t":
            out.append("\\t")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _extract_json_list(text: str) -> Optional[list]:
    """Best-effort extraction of a JSON list from a model response.

    Handles common LLM output quirks:
      - markdown code fences (\`\`\`json ... \`\`\`) anywhere in the response,
        not just at the start/end
      - prose preambles before the JSON ("Let me think... ```json [...] ```")
      - literal newlines/tabs/CR inside JSON string values (raw text fields
        with multi-line bodies that the model forgot to escape as \\n)

    Returns None on garbage. Returns the parsed list (possibly empty) on
    any of: clean response, fenced response, fenced-with-preamble,
    fenced-with-raw-newlines, or first-bracket-to-last-bracket fallback.
    """
    text = text.strip()
    candidates = [text]

    # Add the contents of any fenced ```json ... ``` block(s), anywhere.
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        candidates.append(m.group(1).strip())

    # Last-resort fallback: first-bracket to last-bracket.
    bracket_m = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_m:
        candidates.append(bracket_m.group(0))

    # Try each candidate both raw and after newline-repair.
    for cand in candidates:
        for variant in (cand, _repair_json_strings(cand)):
            try:
                v = json.loads(variant)
                if isinstance(v, list):
                    return v
            except Exception:
                continue
    return None


# ============================================================================
# The agent
# ============================================================================


class _OrdersResult:
    def __init__(self, orders, notes):
        self.orders = orders
        self.reflection_notes = notes


class _NoMind:
    """Empty stand-in so the substrate route doesn't crash on this agent."""
    owner_power = ""
    archetype = "RAW_LLM"
    games_played = 0
    character_brief = None
    beliefs = {}
    predictions = {}
    incoming_commitments = {}
    self_commitments = {}
    strategic_intents = {}
    plan_nodes = {}
    intent_commitments = {}
    intent_revisions = {}
    belief_revisions = {}
    identity_constraints = {}
    move_events = {}
    message_events = {}
    phase_states = {}
    adjustment_events = {}
    private_journal = []
    suspicions = []


class _EmptyGraph:
    def __init__(self): self.nodes = {}
    def to_dict(self): return {"nodes": [], "edges": []}


class _KgsShim:
    def __init__(self):
        self.graphs = {n: _EmptyGraph() for n in
                       ("personality","soul","ethics","theory_of_mind","strategy","counterfactuals")}


class _RawV2Stub:
    """Just enough surface for the substrate route + biopsy to not crash."""
    def __init__(self, power: str, llm_call):
        self.power = power
        self.archetype = "RAW_LLM"
        m = _NoMind()
        m.owner_power = power
        self.mind = m
        self.llm_call = llm_call

    def intake_message(self, message) -> int:
        """No-op — raw_llm agents have no substrate to write into. Returning
        0 keeps the distribute_messages loop happy."""
        return 0


class RawLLMAgent:
    """An LLM-driven Diplomacy player with NO substrate. Same interface as
    V2BridgeAgent so session.py can drop it in for any power."""

    def __init__(self, *, power: str, llm_call,
                 model_label: str = "claude-haiku-4-5-20251001"):
        self.power = power
        self.archetype = "RAW_LLM"
        self.personality_key = "RAW_LLM"
        self.model = model_label
        self.kgs = _KgsShim()
        self.v2 = _RawV2Stub(power, llm_call)
        # Attach recorder so biopsy works
        from diplomacy_prompt_recorder import attach_recorder
        try:
            attach_recorder(self.v2, capacity=60)
            # After attach, self.v2.llm_call is the recorder
            self._llm = self.v2.llm_call
        except Exception:
            self._llm = llm_call

    def negotiate(self, state, visible: list[Message]) -> list[Message]:
        prompt = _negotiate_prompt(self.power, state, visible)
        try:
            response = self._llm(prompt)
        except Exception as e:
            print(f"[raw_llm {self.power}] negotiate error: {e}")
            return []
        items = _extract_json_list(response) or []
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            recipients = [r for r in (item.get("to") or [])
                          if r in POWERS and r != self.power]
            public = bool(item.get("public", False)) or not recipients
            out.append(Message(
                sender=self.power, recipients=recipients, text=text,
                season=state.season, year=state.year, public=public,
            ))
        return out

    def decide_orders(self, state, visible: list[Message]):
        prompt = _orders_prompt(self.power, state, visible)
        try:
            response = self._llm(prompt)
        except Exception as e:
            print(f"[raw_llm {self.power}] orders error: {e}")
            return _OrdersResult([], [f"raw_llm error: {e}"])
        # Parse line by line
        orders = []
        bad = []
        for line in (response or "").splitlines():
            line = line.strip().strip("`-•").strip()
            if not line:
                continue
            o = parse_order(self.power, line)
            if o is not None:
                orders.append(o)
            else:
                bad.append(line)
        notes = []
        if bad:
            notes.append(f"unparseable: {bad[:3]}")
        return _OrdersResult(orders, notes)

    def decide_retreats(self, state) -> list[Order]:
        # Fair RAW_LLM control retreat handling.
        #
        # Before this patch, returning [] caused the engine to force-disband
        # every dislodged England/Germany unit, even when a legal retreat
        # existed. That makes them unfairly weak controls.
        #
        # This is deterministic and intentionally mirrors DiplomacyAgentV2:
        # retreat to the first legal square, otherwise disband.
        my_dislodged = [u for u in state.dislodged if u.power == self.power]
        if not my_dislodged:
            return []

        orders: list[Order] = []
        for u in my_dislodged:
            opts = _legal_retreat_targets(state, u)
            if opts:
                orders.append(Order(
                    power=self.power,
                    unit_kind=u.kind,
                    location=u.location,
                    type="R",
                    target=opts[0],
                ))
            else:
                orders.append(Order(
                    power=self.power,
                    unit_kind=u.kind,
                    location=u.location,
                    type="D",
                ))

        return orders

    def decide_builds(self, state) -> list[Order]:
        # Bare-LLM controls still need normal adjustment handling.
        # They get no substrate/journal/beliefs, but they must not silently
        # waive builds, or England/Germany become unfairly crippled controls.
        prompt = _adjustment_prompt(self.power, state, [])
        try:
            response = self._llm(prompt)
        except Exception as e:
            print(f"[raw_llm {self.power}] adjustment error: {e}")
            response = ""

        orders = []
        bad = []

        delta = len(supply_centers_owned(state, self.power)) - len(units_by_power(state, self.power))
        want_type = "B" if delta > 0 else "D" if delta < 0 else None

        for line in (response or "").splitlines():
            line = line.strip().strip("`-•").strip()
            if not line:
                continue
            o = parse_order(self.power, line)
            if o is not None and (want_type is None or o.type == want_type):
                orders.append(o)
            else:
                bad.append(line)

        # Safety fallback: if the raw model returns no usable adjustment orders,
        # make legal deterministic adjustments rather than waiving owed builds.
        if not orders and delta > 0:
            available = [
                c for c in HOME_CENTERS[self.power]
                if state.sc_owner.get(c) == self.power and not unit_at(state, c)
            ]
            for c in available[:delta]:
                kind = "F" if can_occupy("F", c) and not can_occupy("A", c) else "A"
                if can_occupy(kind, c):
                    orders.append(Order(power=self.power, unit_kind=kind, location=c, type="B"))

        elif not orders and delta < 0:
            for u in sorted(units_by_power(state, self.power), key=lambda x: x.location)[: -delta]:
                orders.append(Order(power=self.power, unit_kind=u.kind, location=u.location, type="D"))

        if bad:
            print(f"[raw_llm {self.power}] bad adjustment lines: {bad[:3]}")

        return orders
