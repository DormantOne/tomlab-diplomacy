"""
diplomacy_llm_protocol.py — prompt assembly + response parsing.

Three call kinds, each a clean function:

  1. compose_negotiate_prompt + parse_negotiate_response
       Composes a negotiation prompt and parses out outgoing messages
       (each one optionally containing a commitspeak tail).

  2. compose_orders_prompt + parse_orders_response
       Composes an orders prompt with the near-term-prediction requirement
       baked in. Parser enforces the requirement; if the LLM forgot to
       emit one, it returns a structured rejection that the caller can
       use to retry-with-reminder.

  3. compose_belief_revision_prompt / compose_intent_revision_prompt +
     parse_revision_response
       Composes a focused per-belief or per-intent revision call. The
       LLM is shown the failing claim and the failing evidence and asked
       to propose a narrower successor.

DESIGN NOTES:
  - These are all PURE functions over (mind, context). No network calls.
    The actual LLM call is a callable injected by the caller (real
    Ollama, real Anthropic API, deterministic stub for testing).
  - Prompts are built telegraphically per the bench results from the
    original llm_agent.py — short prompts produce more compliant output
    from small models.
  - JSON output is parsed with the same tolerant brace-balancing logic
    as the original code (tolerant of code fences, smart quotes, trailing
    prose, commas).
  - The near-term-prediction enforcement is structural: the orders parser
    inspects emitted predictions and returns RejectedOrders if the LLM
    omitted a NEAR_TERM prediction. The caller decides whether to retry
    with a pointed reminder or synthesize a default.
"""

from __future__ import annotations

import random
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal

from diplomacy_kg_schema import (
    AgentMind, MessageEvent, PlanNode, PredictionNode, PredictionStatus,
    PredictionWindowKind, BeliefNode, BeliefRevisionProposal,
    StrategicIntentNode, StrategicIntentRevisionProposal,
    PowerName, ProvinceCode, PhaseKey, new_id,
)
from diplomacy_fovea import build_fovea, CallContext, TurnFovea
from diplomacy_commitspeak import (
    parse_message,
    commitspeak_lines_to_self_commitment_nodes,
    commitspeak_lines_to_incoming_commitment_nodes,
)


# ============================================================================
# LLM call abstraction
# ============================================================================
# We accept a callable that takes (prompt: str) -> str. This lets us
# inject real Ollama, real Anthropic, or a deterministic stub.

LLMCall = Callable[[str], str]


# ============================================================================
# JSON extraction (ported from original llm_agent.py — tolerant of fences,
# smart quotes, trailing prose, trailing commas)
# ============================================================================

def extract_json_object(text: str) -> Optional[dict]:
    """Find the first complete top-level JSON object in `text` and parse it.

    Tolerant of: leading prose, code fences, trailing commentary, smart
    quotes, trailing commas. Uses balanced-brace scanning rather than
    rfind('}'), so trailing prose containing braces won't break parsing.
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "")
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))

    candidates = []
    i = 0; n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0; j = i; in_str = False; esc = False
            while j < n:
                ch = text[j]
                if in_str:
                    if esc: esc = False
                    elif ch == "\\": esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[i:j + 1])
                            i = j; break
                j += 1
            else:
                break
        i += 1

    candidates.sort(key=len, reverse=True)
    for chunk in candidates:
        for s in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return None


# ============================================================================
# Prompt fragments shared across calls
# ============================================================================

COMMITSPEAK_GRAMMAR_REMINDER = """\
COMMITSPEAK (REQUIRED for any tracked promise — prose promises are NOT tracked):
Use FULL power names (AUSTRIA, ENGLAND, FRANCE, GERMANY, RUSSIA, TURKEY) and
FULL phase strings (e.g. 1902-FALL-MOVES, 1907-WINTER-ADJUSTMENTS).

End your message with [[commit ... ]] containing one or more lines:
  not_move_to: <prov> by <phase>
  move_to: <unit> -> <prov> by <phase>
  hold_at: <unit> by <phase>
  support: <unit> S <unit> -> <prov> by <phase>
  non_aggression: with <POWER> by <phase>
  demilitarize: <prov> with <POWER> by <phase>
  build: <kind><prov> in <phase>
One promise per line."""


# Province pool for randomized commitspeak examples — border/cross-region picks
# that don't preferentially reference any one power's home turf.
_COMMITSPEAK_EXAMPLE_PROVINCE_POOL = (
    "BUR", "GAL", "BLA", "NTH", "MAO", "ION",
    "WAL", "RUH", "SIL", "BOH", "TYR", "SER",
)


def randomized_commit_example(speaker=None, current_year: int = 1901) -> str:
    """Generate a fresh randomized commitspeak example block.

    Each call picks a random counterparty (excluding the speaker) and random
    provinces, so no single power or province is reinforced across prompts.
    """
    from diplomacy_engine import POWERS as _POWERS
    candidates = [p for p in _POWERS if p != speaker] if speaker else list(_POWERS)
    counterparty = random.choice(candidates)
    p1, p2 = random.sample(_COMMITSPEAK_EXAMPLE_PROVINCE_POOL, 2)
    near = f"{current_year}-FALL-MOVES"
    far = f"{current_year + 1}-WINTER-ADJUSTMENTS"
    return (
        f"[[commit\n"
        f"  not_move_to: {p1} by {near}\n"
        f"  non_aggression: with {counterparty} by {far}\n"
        f"  hold_at: F {p2} by {near}\n"
        f"]]"
    )


PREDICTION_REQUIREMENT_REMINDER = """\
PREDICTIONS (REQUIRED): emit at least one near-term prediction about another
power's behavior in the next phase. Format each as:
  {"about": "<power>", "type": "<event_type>", "target": "<prov>", "window": "near_term"}
Event types: move_to, attack, support, alliance, non_action, build_at, elimination_of."""


def _phase_key_to_short(phase: PhaseKey) -> str:
    """1902-FALL-MOVES -> F1902 for human-readable hint."""
    parts = phase.split("-")
    if len(parts) != 3:
        return phase
    year, season, _ = parts
    return f"{season[0]}{year}"


def _next_phase_key(phase: PhaseKey) -> PhaseKey:
    """Given current phase, return the next phase in turn order."""
    sequence = ["MOVES", "RETREATS"]
    parts = phase.split("-")
    if len(parts) != 3:
        return phase
    year, season, ph = parts
    if season == "SPRING" and ph == "MOVES":
        return f"{year}-SPRING-RETREATS"
    if season == "SPRING" and ph == "RETREATS":
        return f"{year}-FALL-MOVES"
    if season == "FALL" and ph == "MOVES":
        return f"{year}-FALL-RETREATS"
    if season == "FALL" and ph == "RETREATS":
        return f"{year}-WINTER-ADJUSTMENTS"
    if season == "WINTER" and ph == "ADJUSTMENTS":
        return f"{int(year)+1}-SPRING-MOVES"
    return phase


# ============================================================================
# 1. NEGOTIATE
# ============================================================================

@dataclass
class NegotiationOutput:
    messages: list[MessageEvent] = field(default_factory=list)
    raw_response: str = ""
    parse_notes: list[str] = field(default_factory=list)


def compose_negotiate_prompt(
    mind: AgentMind,
    *,
    fovea: TurnFovea,
    board_summary_text: str,
    recent_messages_text: str,
    addressees: list[PowerName],
    phase: PhaseKey,
) -> str:
    """Assemble the negotiation prompt.

    Telegraphic in the spirit of the original llm_agent.py. The fovea
    provides the narrow slice of mind state; the rest is board + history."""
    other_powers = ", ".join(addressees)
    short = _phase_key_to_short(phase)
    return f"""\
{fovea.render()}

BOARD ({short}):
{board_summary_text}

{recent_messages_text}

TASK: Compose 0-3 outgoing messages for {short}. Each ≤40 words natural prose.
Recipients chosen from: {other_powers}.

{COMMITSPEAK_GRAMMAR_REMINDER}

JSON output, this exact shape (do not add prose around the JSON):
{{"messages": [
  {{"to": ["FRA"], "public": false, "text": "<your prose>\\n[[commit\\n  not_move_to: MUN by {phase}\\n]]"}},
  {{"to": [], "public": true, "text": "<public statement, no commitspeak required>"}}
]}}"""


def parse_negotiate_response(
    raw: str,
    *,
    sender: PowerName,
    phase: PhaseKey,
    valid_powers: set[PowerName],
    valid_provinces: set[ProvinceCode],
) -> NegotiationOutput:
    """Parse LLM negotiation output into MessageEvents.

    Each message gets a MessageEvent. If the message contains commitspeak,
    the caller is responsible for parsing it into commitment nodes —
    we don't do that here because parsing requires the message id,
    which is generated when the MessageEvent is created.
    """
    out = NegotiationOutput(raw_response=raw)
    data = extract_json_object(raw)
    if data is None:
        out.parse_notes.append("no JSON object found in response")
        return out

    code_to_power = {p[:3]: p for p in valid_powers}

    messages_data = data.get("messages") or []
    if not isinstance(messages_data, list):
        out.parse_notes.append(f"messages field is not a list: {type(messages_data).__name__}")
        return out

    for m in messages_data[:4]:
        if not isinstance(m, dict):
            continue
        text = (m.get("text") or "").strip()
        if not text:
            continue
        raw_recipients = m.get("to") or []
        recipients: list[PowerName] = []
        for r in raw_recipients:
            ru = str(r).upper().strip()
            if ru in valid_powers:
                recipients.append(ru)
            elif ru in code_to_power:
                recipients.append(code_to_power[ru])
        is_public = bool(m.get("public")) or not recipients

        # Detect commitspeak tail (raw, not parsed yet — caller does that)
        cs_match = re.search(r"\[\[\s*commit\s*\n.*?\n\s*\]\]", text,
                             re.IGNORECASE | re.DOTALL)
        commitspeak_tail = cs_match.group(0) if cs_match else None

        msg = MessageEvent(
            id=new_id("msg"),
            phase=phase,
            sender=sender,
            recipients=recipients,
            public=is_public,
            body=text,
            commitspeak_tail=commitspeak_tail,
            sent_at=__import__("time").time(),
        )
        out.messages.append(msg)

    return out


def negotiate(
    mind: AgentMind,
    *,
    addressees: list[PowerName],
    phase: PhaseKey,
    board_summary_text: str,
    recent_messages_text: str,
    valid_powers: set[PowerName],
    valid_provinces: set[ProvinceCode],
    llm_call: LLMCall,
) -> NegotiationOutput:
    """End-to-end negotiate: build fovea, prompt, call, parse, register
    incoming/outgoing commitments. Returns the parsed messages.

    Side effects on mind:
      - registers our outgoing messages as MessageEvents
      - registers self-commitments from outgoing commitspeak
    """
    fovea = build_fovea(mind, CallContext(
        kind="negotiation",
        phase=phase,
        addressees=addressees,
        current_phase_for_relevance=phase,
    ))
    prompt = compose_negotiate_prompt(
        mind, fovea=fovea,
        board_summary_text=board_summary_text,
        recent_messages_text=recent_messages_text,
        addressees=addressees, phase=phase,
    )
    raw = llm_call(prompt)
    out = parse_negotiate_response(
        raw, sender=mind.owner_power, phase=phase,
        valid_powers=valid_powers, valid_provinces=valid_provinces,
    )

    # Side effects: store our messages, parse our own commitspeak as self-commitments
    for msg in out.messages:
        mind.message_events[msg.id] = msg
        if msg.commitspeak_tail:
            block = parse_message(
                msg.body, provinces=valid_provinces, powers=valid_powers,
            )
            if block is not None:
                self_nodes, malformed = commitspeak_lines_to_self_commitment_nodes(
                    block, source_msg_id=msg.id,
                    speaker=mind.owner_power,
                    addressees=msg.recipients,
                )
                for n in self_nodes:
                    mind.self_commitments[n.id] = n
                if malformed:
                    out.parse_notes.append(
                        f"{len(malformed)} malformed commitspeak line(s) in {msg.id}"
                    )

    return out


# ============================================================================
# 2. ORDERS — with required near-term prediction
# ============================================================================

@dataclass
class OrdersOutput:
    """Result of an orders call.

    `accepted_orders`: list of (unit_signature, order_string) tuples ready
                       for the engine's parse_order.
    `plan`: the PlanNode the LLM emitted.
    `predictions`: list of PredictionNodes (must include ≥1 near_term).
    `proto_belief`: optional new BeliefNode the LLM proposed.
    `proto_intent`: optional new StrategicIntentNode the LLM proposed.
    `near_term_satisfied`: whether the prediction requirement was met.
    `rejected_reasons`: parse failures the caller may want to retry on.
    """
    accepted_orders: list[tuple[str, str]] = field(default_factory=list)
    plan: Optional[PlanNode] = None
    predictions: list[PredictionNode] = field(default_factory=list)
    proto_belief: Optional[BeliefNode] = None
    proto_intent: Optional[StrategicIntentNode] = None
    near_term_satisfied: bool = False
    raw_response: str = ""
    parse_notes: list[str] = field(default_factory=list)


def compose_orders_prompt(
    mind: AgentMind,
    *,
    fovea: TurnFovea,
    board_summary_text: str,
    recent_messages_text: str,
    unit_options_text: str,
    phase: PhaseKey,
    pointed_reminder: Optional[str] = None,
) -> str:
    """Assemble orders prompt. If `pointed_reminder` is set, it's a
    retry — append the reminder at the top so the model sees it first."""
    short = _phase_key_to_short(phase)
    next_short = _phase_key_to_short(_next_phase_key(phase))

    reminder_block = (f"\n\n!!! RETRY !!!\n{pointed_reminder}\n!!! END RETRY !!!\n"
                      if pointed_reminder else "")

    return f"""\
{fovea.render()}{reminder_block}

BOARD ({short}):
{board_summary_text}

{recent_messages_text}

YOUR UNITS — order each one:
{unit_options_text}

ORDER FORMS:
  HOLD     :  A PAR H
  MOVE     :  A PAR - BUR
  SUPPORT  :  A MUN S A KIE - BER
              A PAR S A MAR
  CONVOY   :  F MAO C A LON - BRE

{PREDICTION_REQUIREMENT_REMINDER}

JSON output, this exact shape (no prose around the JSON):
{{
  "orders": ["A PAR - BUR", "F BRE - MAO"],
  "plan": {{
    "head": "<one-line tactical summary>",
    "body": "<reasoning, 1-3 sentences>",
    "parent_intent_id": null
  }},
  "predictions": [
    {{"about": "GERMANY", "type": "non_action", "target": "BEL",
      "window": "near_term", "rationale": "GER has no spare unit"}}
  ]
}}

You MAY also include "proto_belief" and "proto_intent" objects in the JSON
when you want to propose new ones. Both are optional. Schemas:
  proto_belief: {{"about": "<power>", "type": "<disposition|tactical_pattern|relationship|risk_assessment|credibility>", "head": "...", "body": "..."}}
  proto_intent: {{"head": "...", "body": "...", "target_powers": [...], "target_provinces": [...], "horizon": "{next_short}"}}"""


def parse_orders_response(
    raw: str,
    *,
    speaker: PowerName,
    phase: PhaseKey,
    own_unit_signatures: list[str],
    valid_powers: set[PowerName],
    valid_provinces: set[ProvinceCode],
) -> OrdersOutput:
    """Parse the LLM orders response into structured nodes.

    Enforces the per-turn near-term prediction requirement structurally:
    if no NEAR_TERM prediction is found, sets near_term_satisfied=False
    and the caller decides whether to retry-with-reminder.
    """
    out = OrdersOutput(raw_response=raw)
    data = extract_json_object(raw)
    if data is None:
        out.parse_notes.append("no JSON object found in response")
        return out

    # Orders — try several common keys the model might use
    raw_orders = (data.get("orders")
                  or data.get("moves")
                  or data.get("order_list")
                  or data.get("my_orders")
                  or [])
    if isinstance(raw_orders, list):
        seen_origins: set[str] = set()
        for o in raw_orders:
            # The model might emit orders as dicts {"unit": "A PAR", "action": "MOVE", "to": "BUR"}
            # rather than strings. Coerce to string.
            if isinstance(o, dict):
                # Try to reassemble into engine syntax
                unit = str(o.get("unit") or o.get("u") or "").strip()
                action = str(o.get("action") or o.get("type") or o.get("order") or "").strip().upper()
                target = str(o.get("to") or o.get("target") or o.get("dest") or "").strip().upper()
                if action in ("HOLD", "H") or not action:
                    if unit:
                        order_str = f"{unit} H".upper()
                    else:
                        continue
                elif action in ("MOVE", "M"):
                    if unit and target:
                        order_str = f"{unit} - {target}".upper()
                    else:
                        continue
                elif action in ("SUPPORT", "S"):
                    sup_unit = str(o.get("supports") or o.get("support") or "").strip()
                    sup_dest = str(o.get("support_dest") or o.get("support_target") or "").strip().upper()
                    if unit and sup_unit:
                        if sup_dest:
                            order_str = f"{unit} S {sup_unit} - {sup_dest}".upper()
                        else:
                            order_str = f"{unit} S {sup_unit}".upper()
                    else:
                        continue
                elif action in ("CONVOY", "C"):
                    conv_unit = str(o.get("convoys") or o.get("convoy") or "").strip()
                    conv_dest = str(o.get("convoy_dest") or "").strip().upper()
                    if unit and conv_unit and conv_dest:
                        order_str = f"{unit} C {conv_unit} - {conv_dest}".upper()
                    else:
                        continue
                else:
                    continue
            else:
                order_str = str(o).strip().upper()
            tokens = order_str.split()
            if len(tokens) < 2:
                out.parse_notes.append(f"skipped short order: {o!r}")
                continue
            origin = tokens[1] if tokens[0] in ("A", "F") else None
            if origin in seen_origins:
                out.parse_notes.append(f"duplicate order for {origin}: {o!r}")
                continue
            if origin:
                seen_origins.add(origin)
            unit_sig = " ".join(tokens[:2]) if tokens[0] in ("A", "F") else order_str
            out.accepted_orders.append((unit_sig, order_str))

    # Plan
    plan_data = data.get("plan")
    if isinstance(plan_data, dict):
        out.plan = PlanNode(
            id=new_id("plan"),
            formed_at_phase=phase,
            head=str(plan_data.get("head", "")).strip()[:200],
            body=str(plan_data.get("body", "")).strip(),
            parent_intent_id=plan_data.get("parent_intent_id") or None,
        )

    # Predictions — REQUIRED ≥1 NEAR_TERM. Try several common keys.
    next_phase = _next_phase_key(phase)
    raw_preds = (data.get("predictions")
                 or data.get("forecast")
                 or data.get("forecasts")
                 or data.get("expectations")
                 or [])
    valid_event_types = {"move_to", "attack", "support", "alliance",
                         "non_action", "build_at", "elimination_of"}
    # Type aliases — small models often paraphrase
    type_aliases = {
        "move": "move_to", "moves": "move_to", "movement": "move_to",
        "advance": "move_to", "go": "move_to", "go_to": "move_to",
        "attacks": "attack", "attacking": "attack", "aggression": "attack",
        "supports": "support", "supporting": "support",
        "ally": "alliance", "allies": "alliance", "allied": "alliance",
        "hold": "non_action", "hold_at": "non_action", "stay": "non_action",
        "stays": "non_action", "no_action": "non_action",
        "no_move": "non_action", "wont_move": "non_action",
        "build": "build_at", "builds": "build_at",
        "eliminate": "elimination_of", "elimination": "elimination_of",
        "eliminated": "elimination_of",
    }
    if isinstance(raw_preds, list):
        for pd in raw_preds:
            if not isinstance(pd, dict):
                continue
            # 'about' can also be 'power', 'who', 'subject'
            about = str(pd.get("about")
                        or pd.get("power")
                        or pd.get("who")
                        or pd.get("subject")
                        or "").upper().strip()
            if about not in valid_powers:
                out.parse_notes.append(f"prediction skipped — unknown power: {about!r}")
                continue
            ev_type = str(pd.get("type")
                          or pd.get("action")
                          or pd.get("event")
                          or pd.get("what")
                          or "").strip().lower()
            ev_type = type_aliases.get(ev_type, ev_type)
            if ev_type not in valid_event_types:
                out.parse_notes.append(f"prediction skipped — unknown type: {ev_type!r}")
                continue
            target = (pd.get("target")
                      or pd.get("to")
                      or pd.get("province")
                      or pd.get("where"))
            if target is not None:
                target = str(target).upper().strip()
                if target and target not in valid_provinces and target not in valid_powers:
                    out.parse_notes.append(
                        f"prediction skipped — unknown target: {target!r}"
                    )
                    continue
            window_kind_str = str(pd.get("window", "near_term")).lower().strip()
            if window_kind_str == "near_term":
                window_kind = PredictionWindowKind.NEAR_TERM
                window_phase = next_phase
            else:
                window_kind = PredictionWindowKind.LONG_HORIZON
                # If the model gave an explicit window string, parse it; else
                # default to 2 phases out
                explicit = pd.get("window_phase")
                window_phase = (str(explicit) if explicit else
                                _next_phase_key(_next_phase_key(phase)))

            subject_power = pd.get("subject_power")
            if subject_power is not None:
                subject_power = str(subject_power).upper().strip()
                if subject_power not in valid_powers:
                    subject_power = None

            pred_target = target if target in valid_provinces else None
            if not subject_power and target in valid_powers:
                subject_power = target
                pred_target = None

            p = PredictionNode(
                id=new_id("pred"), about_power=about,
                formed_at_phase=phase,
                predicted_event_type=ev_type,
                predicted_target=pred_target,
                predicted_subject_power=subject_power,
                prediction_window=window_phase,
                window_kind=window_kind,
                confidence=float(pd.get("confidence", 0.5)),
                rationale=str(pd.get("rationale", "")).strip()[:300],
            )
            out.predictions.append(p)

    out.near_term_satisfied = any(
        p.window_kind == PredictionWindowKind.NEAR_TERM for p in out.predictions
    )

    # Optional proto-belief
    pb = data.get("proto_belief")
    if isinstance(pb, dict):
        from diplomacy_kg_schema import BeliefStatus, BeliefType
        try:
            btype = BeliefType(str(pb.get("type", "")).strip())
        except ValueError:
            btype = None
        about = str(pb.get("about", "")).upper().strip()
        head = str(pb.get("head", "")).strip()
        if btype and about in valid_powers and head:
            out.proto_belief = BeliefNode(
                id=new_id("belief"),
                about_power=about, belief_type=btype,
                head=head[:200],
                body=str(pb.get("body", "")).strip()[:1000],
                formed_at_phase=phase, formed_in_game=0,   # caller fills this
                last_updated_phase=phase,
                status=BeliefStatus.PROTO,
            )

    # Optional proto-intent
    pi = data.get("proto_intent")
    if isinstance(pi, dict):
        from diplomacy_kg_schema import StrategicIntentStatus
        head = str(pi.get("head", "")).strip()
        if head:
            target_powers = [str(x).upper() for x in (pi.get("target_powers") or [])
                             if str(x).upper() in valid_powers]
            target_provinces = [str(x).upper() for x in (pi.get("target_provinces") or [])
                                if str(x).upper() in valid_provinces]
            horizon = str(pi.get("horizon", "")) or _next_phase_key(_next_phase_key(phase))
            out.proto_intent = StrategicIntentNode(
                id=new_id("intent"),
                head=head[:200],
                body=str(pi.get("body", "")).strip()[:1000],
                formed_at_phase=phase,
                target_powers=target_powers,
                target_provinces=target_provinces,
                horizon=horizon,
                status=StrategicIntentStatus.PROTO,
            )

    return out


def synthesize_default_near_term_prediction(
    *,
    accepted_orders: list[tuple[str, str]],
    speaker: PowerName,
    phase: PhaseKey,
    valid_powers: set[PowerName],
) -> Optional[PredictionNode]:
    """Last-resort synthesis: if the LLM failed twice to emit a near-term
    prediction, derive one so the lifecycle still gets fresh signal.

    Two heuristics, in order:
      1. If any of OUR orders moves a unit, predict our move succeeds.
         (about_power = speaker)
      2. Otherwise (we held everything, or no orders), predict that some
         other power will not enter our home centers next phase. This is
         a weak signal but reliably falsifiable.
    """
    next_phase = _next_phase_key(phase)
    # Heuristic 1: derive from a MOVE we issued
    for unit_sig, order_str in accepted_orders:
        m = re.search(r"\b([A-Z]{3})\s*(?:->|-|TO)\s*([A-Z]{3})\b", order_str)
        if m:
            return PredictionNode(
                id=new_id("pred"), about_power=speaker,
                formed_at_phase=phase,
                predicted_event_type="move_to",
                predicted_target=m.group(2),
                predicted_subject_power=None,
                prediction_window=next_phase,
                window_kind=PredictionWindowKind.NEAR_TERM,
                confidence=0.5,
                rationale="(synthesized from speaker's own MOVE order — "
                          "LLM did not emit a near-term prediction)",
            )

    # Heuristic 2: predict another power's non_action on a generic SC.
    # Pick any power that isn't us and a province they don't already own.
    others = sorted(p for p in valid_powers if p != speaker)
    if others:
        return PredictionNode(
            id=new_id("pred"), about_power=others[0],
            formed_at_phase=phase,
            predicted_event_type="non_action",
            predicted_target="STP",   # a province most powers don't touch
            predicted_subject_power=None,
            prediction_window=next_phase,
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=0.3,
            rationale="(synthesized fallback — LLM emitted no MOVE orders "
                      "and no near-term prediction)",
        )
    return None


def order_decision(
    mind: AgentMind,
    *,
    phase: PhaseKey,
    addressees: list[PowerName],
    board_summary_text: str,
    recent_messages_text: str,
    unit_options_text: str,
    own_unit_signatures: list[str],
    valid_powers: set[PowerName],
    valid_provinces: set[ProvinceCode],
    llm_call: LLMCall,
    max_retries: int = 1,
) -> OrdersOutput:
    """End-to-end orders call: build fovea, prompt, call, parse, enforce
    near-term-prediction requirement, retry with pointed reminder once,
    and synthesize a default if still absent.

    Returns the OrdersOutput. Side effects on `mind`:
      - registers the plan node
      - registers all valid predictions
      - registers proto_belief / proto_intent if emitted
    """
    fovea = build_fovea(mind, CallContext(
        kind="orders", phase=phase,
        addressees=addressees,
        current_phase_for_relevance=phase,
    ))

    prompt = compose_orders_prompt(
        mind, fovea=fovea,
        board_summary_text=board_summary_text,
        recent_messages_text=recent_messages_text,
        unit_options_text=unit_options_text,
        phase=phase,
    )
    raw = llm_call(prompt)
    out = parse_orders_response(
        raw, speaker=mind.owner_power, phase=phase,
        own_unit_signatures=own_unit_signatures,
        valid_powers=valid_powers, valid_provinces=valid_provinces,
    )

    # Retry loop for the near-term-prediction requirement
    retries = 0
    while not out.near_term_satisfied and retries < max_retries:
        retries += 1
        retry_prompt = compose_orders_prompt(
            mind, fovea=fovea,
            board_summary_text=board_summary_text,
            recent_messages_text=recent_messages_text,
            unit_options_text=unit_options_text,
            phase=phase,
            pointed_reminder=(
                "Your previous response did not include a near-term prediction. "
                "You MUST emit at least one prediction with \"window\": \"near_term\". "
                "Predict what some other power will or will NOT do in the next phase."
            ),
        )
        raw = llm_call(retry_prompt)
        retry_out = parse_orders_response(
            raw, speaker=mind.owner_power, phase=phase,
            own_unit_signatures=own_unit_signatures,
            valid_powers=valid_powers, valid_provinces=valid_provinces,
        )
        # Take whichever response had a near-term prediction; else keep the first
        if retry_out.near_term_satisfied:
            out = retry_out
            break

    # Final fallback: synthesize from orders
    if not out.near_term_satisfied:
        synth = synthesize_default_near_term_prediction(
            accepted_orders=out.accepted_orders,
            speaker=mind.owner_power, phase=phase,
            valid_powers=valid_powers,
        )
        if synth is not None:
            out.predictions.append(synth)
            out.near_term_satisfied = True
            out.parse_notes.append("synthesized default near-term prediction")

    # Register everything in the mind, with auto-linking so the lifecycle
    # has the parent/child references it needs to grade evidence rolls-up.
    #
    # Auto-linking rules:
    #   - If proto_belief was emitted in this turn, every prediction that's
    #     about the same power gets that belief in source_belief_ids.
    #     (The LLM proposed a hypothesis AND testable claims; we link them.)
    #   - Predictions also link to ACTIVE beliefs about their about_power —
    #     so existing beliefs keep accumulating evidence from new predictions.
    #   - If proto_intent was emitted, the plan node gets parent_intent_id
    #     pointing to that intent (unless the LLM already set it explicitly).
    #     Predictions also get parent_intent_id pointing to that intent.

    if out.proto_belief is not None:
        out.proto_belief.formed_in_game = mind.games_played
        mind.beliefs[out.proto_belief.id] = out.proto_belief

    if out.proto_intent is not None:
        # Set active_since_phase to None at proto stage; lifecycle sets it
        # when promoting.
        mind.strategic_intents[out.proto_intent.id] = out.proto_intent

    # Wire predictions to source beliefs and parent intents
    from diplomacy_kg_schema import BeliefStatus
    for p in out.predictions:
        # Link to proto_belief if same about_power
        if (out.proto_belief is not None
                and out.proto_belief.about_power == p.about_power):
            if out.proto_belief.id not in p.source_belief_ids:
                p.source_belief_ids.append(out.proto_belief.id)
        # ALSO link to any ACTIVE belief about the same power, so
        # existing active beliefs keep accumulating predictive evidence
        for b in mind.beliefs.values():
            if (b.about_power == p.about_power
                    and b.status in (BeliefStatus.ACTIVE, BeliefStatus.PROTO)
                    and b.id not in p.source_belief_ids):
                p.source_belief_ids.append(b.id)
        # Link prediction to proto_intent if one was emitted
        if out.proto_intent is not None and p.parent_intent_id is None:
            p.parent_intent_id = out.proto_intent.id
        mind.predictions[p.id] = p

    # Plan registration — link to proto_intent if LLM didn't already
    if out.plan is not None:
        if out.plan.parent_intent_id is None and out.proto_intent is not None:
            out.plan.parent_intent_id = out.proto_intent.id
        out.plan.emitted_prediction_ids = [p.id for p in out.predictions]
        mind.plan_nodes[out.plan.id] = out.plan

    return out


# ============================================================================
# 3. REVISION PROPOSALS
# ============================================================================

def compose_belief_revision_prompt(
    mind: AgentMind, parent_belief: BeliefNode,
    *, phase: PhaseKey,
) -> str:
    """One-belief revision call. Show the parent belief, the failing
    predictions, and ask for a narrower successor."""
    failing = [
        p for p in mind.predictions.values()
        if parent_belief.id in p.source_belief_ids
        and p.status == PredictionStatus.REFUTED
    ]
    failing_lines = "\n".join(
        f"  - {p.predicted_event_type} target={p.predicted_target or '-'} "
        f"about={p.about_power} window={p.prediction_window} -> REFUTED"
        for p in failing[:5]
    )
    return f"""\
You hold this belief about {parent_belief.about_power}:
  HEAD: {parent_belief.head}
  BODY: {parent_belief.body[:300]}

Recent predictions from this belief that REFUTED:
{failing_lines or '  (none captured)'}

Propose a NARROWER successor belief — one that fits both your past evidence
AND the recent failures. Keep it specific. Do not propose the same claim again.

JSON output:
{{"head": "<narrower head, 1-2 sentences>",
  "body": "<reasoning for narrowing>",
  "type": "{parent_belief.belief_type.value}"}}"""


def compose_intent_revision_prompt(
    mind: AgentMind, parent_intent: StrategicIntentNode,
    *, phase: PhaseKey,
) -> str:
    """One-intent revision call."""
    return f"""\
You hold this strategic intent:
  HEAD: {parent_intent.head}
  BODY: {parent_intent.body[:400]}

Stats: {parent_intent.predictions_confirmed} predictions confirmed, "
{parent_intent.predictions_refuted} refuted; sc_delta {parent_intent.sc_delta_under_intent}.

The intent's recent predictions are failing. Propose a NARROWER successor
intent — one with a tighter target or shorter horizon.

JSON output:
{{"head": "<narrower head>", "body": "<reasoning>",
  "target_powers": [...], "target_provinces": [...],
  "horizon": "<phase like 1903-WINTER-ADJUSTMENTS>"}}"""


@dataclass
class RevisionResponse:
    proposal: Optional[BeliefRevisionProposal | StrategicIntentRevisionProposal] = None
    raw_response: str = ""
    parse_notes: list[str] = field(default_factory=list)


def parse_belief_revision_response(
    raw: str, *, parent_belief: BeliefNode, phase: PhaseKey,
    failing_pred_ids: list[str],
) -> RevisionResponse:
    out = RevisionResponse(raw_response=raw)
    data = extract_json_object(raw)
    if data is None:
        out.parse_notes.append("no JSON object")
        return out
    head = str(data.get("head", "")).strip()
    if not head:
        out.parse_notes.append("missing head")
        return out
    out.proposal = BeliefRevisionProposal(
        id=new_id("brev"),
        parent_belief_id=parent_belief.id,
        proposed_head=head[:200],
        proposed_body=str(data.get("body", "")).strip()[:1000],
        proposed_belief_type=parent_belief.belief_type,
        reason="predictions failing",
        triggering_prediction_ids=failing_pred_ids,
        formed_at_phase=phase,
    )
    return out


def parse_intent_revision_response(
    raw: str, *, parent_intent: StrategicIntentNode, phase: PhaseKey,
    triggering_evidence_ids: list[str],
) -> RevisionResponse:
    out = RevisionResponse(raw_response=raw)
    data = extract_json_object(raw)
    if data is None:
        out.parse_notes.append("no JSON object")
        return out
    head = str(data.get("head", "")).strip()
    if not head:
        out.parse_notes.append("missing head")
        return out
    out.proposal = StrategicIntentRevisionProposal(
        id=new_id("irev"),
        parent_intent_id=parent_intent.id,
        proposed_head=head[:200],
        proposed_body=str(data.get("body", "")).strip()[:1000],
        reason="predictive_failure",
        triggering_evidence_ids=triggering_evidence_ids,
        formed_at_phase=phase,
    )
    return out


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    from diplomacy_kg_schema import CharacterBrief

    print("=" * 72)
    print("LLM PROTOCOL SANITY CHECK")
    print("=" * 72)

    # Minimal mind
    mind = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    mind.character_brief = CharacterBrief(
        id=new_id("brief"), archetype="MARSHAL_VEIL",
        text="I am Marshal Veil.", generated_at=__import__("time").time(),
    )

    PROVINCES = {"PAR", "MAR", "BRE", "BUR", "MUN", "BEL", "RUH", "BER", "KIE"}
    POWERS = {"AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "RUSSIA", "TURKEY"}

    # Build a fake "LLM" that returns valid JSON for an orders call
    fake_response = '''{
  "orders": ["A PAR - BUR", "F BRE - MAO", "A MAR H"],
  "plan": {
    "head": "Press east toward Burgundy.",
    "body": "Move A PAR to BUR; F BRE to MAO for flexibility; hold MAR.",
    "parent_intent_id": null
  },
  "predictions": [
    {"about": "GERMANY", "type": "non_action", "target": "BEL",
     "window": "near_term", "rationale": "GER cannot afford to commit there"}
  ]
}'''
    out = parse_orders_response(
        fake_response, speaker="FRANCE",
        phase="1902-SPRING-MOVES",
        own_unit_signatures=["A PAR", "F BRE", "A MAR"],
        valid_powers=POWERS, valid_provinces=PROVINCES,
    )
    print(f"  orders parsed: {len(out.accepted_orders)}")
    for sig, ostr in out.accepted_orders:
        print(f"    {sig:<8} :: {ostr}")
    print(f"  plan: {out.plan.head if out.plan else 'NONE'}")
    print(f"  predictions: {len(out.predictions)}")
    for p in out.predictions:
        print(f"    {p.about_power} {p.predicted_event_type} target={p.predicted_target} "
              f"window_kind={p.window_kind.value}")
    print(f"  near_term_satisfied: {out.near_term_satisfied}")
    assert len(out.accepted_orders) == 3
    assert out.plan is not None
    assert len(out.predictions) == 1
    assert out.near_term_satisfied

    # Test the rejection path
    bad_response = '{"orders": ["A PAR - BUR"], "plan": {"head": "..."}, "predictions": []}'
    out2 = parse_orders_response(
        bad_response, speaker="FRANCE",
        phase="1902-SPRING-MOVES",
        own_unit_signatures=["A PAR"],
        valid_powers=POWERS, valid_provinces=PROVINCES,
    )
    print()
    print(f"  empty-predictions case: near_term_satisfied={out2.near_term_satisfied}")
    assert out2.near_term_satisfied is False

    # Test synthesis
    synth = synthesize_default_near_term_prediction(
        accepted_orders=[("A PAR", "A PAR - BUR")],
        speaker="FRANCE", phase="1902-SPRING-MOVES",
        valid_powers=POWERS,
    )
    print(f"  synthesized prediction: target={synth.predicted_target if synth else None}")
    assert synth is not None
    assert synth.predicted_target == "BUR"
    assert synth.window_kind == PredictionWindowKind.NEAR_TERM

    print()
    print("All protocol sanity checks passed.")
