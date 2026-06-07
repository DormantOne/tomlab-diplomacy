"""
LLM agent backed by Anthropic's Messages API (Claude Haiku 4.5 by default).

The agent's job each turn:
  1. Render relevant slices of all six knowledge graphs into a system prompt.
  2. Compose a turn-context block (board state, supply centers, recent log,
     recent diplomatic messages).
  3. Ask the model for a JSON response: { messages: [...], orders: [...],
     reflections: { trust_updates: [...], counterfactuals: [...] } }
  4. Apply the reflections back into the KGs.
  5. Return the messages and orders to the game loop.

Negotiation and orders are produced in two separate calls:
  - Negotiation: outgoing private messages to specific powers + public
    statement.
  - Orders: final orders for the season.
This lets the model "speak first, then commit", which mirrors human play.

The LLM call function is named `_call_ollama` for backward compatibility
with the rest of the codebase, which checks for an "__OLLAMA_ERROR__"
sentinel string on failure. The implementation now hits Anthropic's API
instead. If ANTHROPIC_API_KEY isn't set, every call returns the error
sentinel and the agent falls back to deterministic-but-weak heuristics.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
import urllib.request
import urllib.error

from diplomacy_engine import (
    GameState, Order, POWERS, PROVINCES,
    units_by_power, supply_centers_owned, parse_order, ADJ,
    is_adjacent, can_occupy,
)
from .knowledge_graph import AgentKGBundle
from .personalities import PERSONALITIES, seed_kg_bundle


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Sentinel string returned by the LLM caller when something fails (network,
# API error, missing key). Callers grep for "__OLLAMA_ERROR__" historically;
# we keep that exact sentinel for backward compatibility even though we
# no longer call Ollama. The string just means "the LLM call failed".
LLM_ERROR_SENTINEL = "__OLLAMA_ERROR__"


@dataclass
class Message:
    sender: str            # power name or "USER"
    recipients: list[str]  # power names; empty means public
    text: str
    season: str = ""
    year: int = 0
    public: bool = False


@dataclass
class AgentTurnOutput:
    messages: list[Message] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    raw_response: str = ""
    reflection_notes: list[str] = field(default_factory=list)


class LLMAgent:
    def __init__(self, power: str, personality_key: str,
                 model: str = DEFAULT_MODEL,
                 anthropic_url: str = ANTHROPIC_URL):
        self.power = power
        self.personality_key = personality_key
        self.model = model
        self.anthropic_url = anthropic_url
        # Kept under the old attribute name for backward compatibility with
        # the legacy server's health-probe code path; treats it as "the LLM
        # endpoint we use", regardless of whether it's local or remote.
        self.ollama_url = anthropic_url
        self.kgs = AgentKGBundle(owner_power=power)
        seed_kg_bundle(self.kgs, personality_key,
                       other_powers=[p for p in POWERS if p != power])
        self.message_history: list[Message] = []
        # Cached self-brief: model-generated first-person voice block describing
        # who this agent is. Generated once at start (or first turn), reused
        # for every subsequent prompt. Far more useful to a small LLM than the
        # raw graph dump.
        self.self_brief: str = ""
        # Biopsy buffer: keeps the last N raw exchanges for inspection.
        self.biopsy_log: list[dict] = []
        self._biopsy_max = 12
        self.last_negotiation_raw: str = ""

    def ensure_self_brief(self) -> str:
        """Generate (and cache) a tight in-character voice block from the KGs.

        This bootstrap call happens once. The brief becomes the agent's
        identity in every subsequent prompt — much shorter and far more
        useful to the LLM than dumping all six graph structures.
        """
        if self.self_brief:
            return self.self_brief

        meta = PERSONALITIES[self.personality_key]
        # Render the structured graphs (used here, NOT in every prompt)
        kg_dump = (
            f"PERSONALITY\n{self.kgs.render_personality()}\n\n"
            f"SOUL\n{self.kgs.render_soul()}\n\n"
            f"ETHICS\n{self.kgs.render_ethics()}\n\n"
            f"STRATEGY\n{self.kgs.render_strategy()}\n"
        )
        bootstrap_prompt = f"""You are an actor preparing to play a character in a long political game called Diplomacy.

The character you will play is {meta['display_name']} — {meta['tagline']}.
You are playing the power {self.power}.

Below is the structured character file. It is data, not prose. Your job is to read it
carefully and write a tight first-person voice block (about 120-180 words) that you
will keep in your head every time you speak as this character. This is YOUR voice.
Speak as the character, not about the character.

Cover, in your own words:
  - Who you are and how you carry yourself (tone, mood, manner of speech)
  - What you value and what you fear
  - What you will permit yourself to do, and what you will refuse
  - How you approach this game strategically

Write in confident first person. Do not hedge. Do not list bullets. Do not mention
that you are an AI or a character or that this is a game. Just be the person.

CHARACTER FILE
{kg_dump}

Begin your voice block now. Plain prose only. About 150 words."""

        raw = self._call_ollama(bootstrap_prompt, force_json=False)
        if "__OLLAMA_ERROR__" in raw:
            # Fall back to a static synthesis from the personality metadata
            self.self_brief = self._fallback_brief(meta)
        else:
            # Strip code fences/quote wrappers the model might add
            cleaned = raw.strip()
            cleaned = re.sub(r"^```[a-z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)
            cleaned = cleaned.strip("\"'")
            # If the model produced something tiny or incoherent, fall back
            if len(cleaned) < 80:
                self.self_brief = self._fallback_brief(meta)
            else:
                self.self_brief = cleaned
        return self.self_brief

    def _fallback_brief(self, meta: dict) -> str:
        """Static brief used when Ollama is unavailable for the bootstrap call."""
        return (
            f"I am {meta['display_name']}. {meta['tagline']} "
            f"I play {self.power} in this game of Diplomacy. "
            f"I act in keeping with my nature, and I do not explain myself unless I must."
        )

    def _record_biopsy(self, *, step: str, year: int, season: str,
                       prompt: str, raw_response: str,
                       parsed_ok: bool, parsed_summary: str,
                       notes: list[str] | None = None) -> None:
        import time as _t
        self.biopsy_log.append({
            "step": step,
            "year": year,
            "season": season,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_ok": parsed_ok,
            "parsed_summary": parsed_summary,
            "notes": notes or [],
            "ts": _t.time(),
        })
        while len(self.biopsy_log) > self._biopsy_max:
            self.biopsy_log.pop(0)

    # -------------------- Prompt assembly -------------------- #

    def _board_view(self, state: GameState) -> str:
        lines = [f"Year {state.year} {state.season} {state.phase}"]
        for power in POWERS:
            scs = supply_centers_owned(state, power)
            units = units_by_power(state, power)
            unit_str = ", ".join(f"{u.kind}{u.location}" for u in units) or "(none)"
            lines.append(f"  {power}: {len(scs)} SC ({', '.join(sorted(scs)) or '-'}) | units: {unit_str}")
        return "\n".join(lines)

    def _own_unit_options(self, state: GameState) -> str:
        """Show legal one-step neighbors for each of this power's units."""
        my_units = units_by_power(state, self.power)
        lines = []
        for u in my_units:
            key = "army" if u.kind == "A" else "fleet"
            neighbors = sorted(ADJ.get(u.location, {}).get(key, []))
            valid = [n for n in neighbors if can_occupy(u.kind, n)]
            lines.append(f"  {u.kind} {u.location} can move to: {', '.join(valid) or '(nowhere — must hold)'}")
        return "\n".join(lines) if lines else "  (no units)"

    # -------------------- Telegraphic prompt assembly -------------------- #

    def _identity_tag(self) -> str:
        """Compress all six KGs into a single ~200-char telegraphic identity.

        This is the model's whole character file. Bench data showed the model
        writes BETTER (more in-character, more eloquent) with less character
        context, not more. The KGs still drive everything underneath — they
        get queried, updated, and persisted — but the *prompt* sees only a
        compressed digest.
        """
        meta = PERSONALITIES[self.personality_key]

        # Pull the 3 most-weighted personality traits
        per = self.kgs["personality"]
        traits = []
        for e in per.edges_with_relation("exhibits"):
            traits.append((e.weight, e.dst.replace("trait:", "")))
        traits.sort(reverse=True)
        top_traits = "/".join(t[1][:4] for t in traits[:3])  # 'open/extr/cons'

        # Soul: top values, fears, desires
        soul = self.kgs["soul"]
        values = [n.id.replace("value:", "") for n in soul.nodes_of_type("value")][:3]
        fears  = [n.id.replace("fear:",  "") for n in soul.nodes_of_type("fear")][:2]
        desires= [n.id.replace("desire:","") for n in soul.nodes_of_type("desire")][:2]

        # Ethics: permits and forbids
        ethics = self.kgs["ethics"]
        permits = [n.id.replace("permit:", "") for n in ethics.nodes_of_type("permit")][:3]
        forbids = [n.id.replace("forbid:", "") for n in ethics.nodes_of_type("forbid")][:2]

        # Compress with hyphens, slashes, & — proven to work in bench
        archetype = meta['display_name'].split()[0].upper()
        return (
            f"{archetype}. {meta['tagline'][:60]}. {self.power}. "
            f"traits:{top_traits}. "
            f"wants:{'/'.join(values)}. "
            f"fears:{'/'.join(fears)}. "
            f"desires:{'/'.join(desires)}. "
            f"ok:{'/'.join(permits)}. "
            f"no:{'/'.join(forbids)}."
        )

    def _ledger_tag(self) -> str:
        """Tight ledger of dynamic state: trust + intent + comm count.

        Personality is static; this is what changes turn-to-turn.
        """
        tom = self.kgs["theory_of_mind"]
        bits = []
        for nid, n in tom.nodes.items():
            if n.type != "power":
                continue
            target = nid.replace("power:", "")
            t = n.attrs.get("trust", 0.0)
            ic = n.attrs.get("communication_count", 0)
            intent = n.attrs.get("predicted_intent", "")
            piece = f"{target[:3]}={t:+.1f}"
            if ic: piece += f"/{ic}msg"
            if intent and intent != "unknown": piece += f"/{intent[:20]}"
            bits.append(piece)
        ledger = "trust:" + ",".join(bits)

        cf = self.kgs["counterfactuals"]
        cf_nodes = [n for n in cf.nodes.values() if n.type == "counterfactual"]
        if cf_nodes:
            # Last 2 counterfactuals only. DON'T truncate mid-word — the model
            # treats truncated text as an incomplete sentence to finish, which
            # corrupts its understanding of what we're asking for.
            tail = []
            for n in cf_nodes[-2:]:
                lab = n.id.split(":", 1)[-1].split(":")[0]
                premise = (n.attrs.get("premise") or "").replace("\n", " ").strip()
                expected = (n.attrs.get("expected") or "").replace("\n", " ").strip()
                tail.append(f"{lab}: if {premise} → {expected}")
            ledger += "\nLive what-ifs:\n  " + "\n  ".join(tail)

        # Active strategic plan if set beyond the seed
        strat = self.kgs["strategy"].nodes.get("plan:active")
        if strat:
            summary = strat.attrs.get("summary", "")
            if summary and "no plan yet" not in summary:
                ledger += f"\nplan:{summary[:80]}"
        return ledger

    def _system_prompt(self) -> str:
        """Tiny system prompt: identity tag + ledger only.

        Per the bench (run on gpt-oss:20b on Apple Silicon): a ~200-char
        telegraphic identity prompt produces FASTER, JSON-cleaner, AND
        more-in-character outputs than a verbose one. Verbose 2095-char
        prompts caused 100% timeouts; telegraphic 562-char prompts hit
        100% JSON success in 46s avg.
        """
        return f"{self._identity_tag()}\n\n{self._ledger_tag()}"

    # -------------------- LLM call (Anthropic) -------------------- #

    def _call_ollama(self, prompt: str, timeout: float = 180.0,
                     force_json: bool = False,
                     max_output_tokens: int = 700) -> str:
        """Call the Anthropic Messages API. Function name kept for compatibility
        with existing callers; underneath, this is now Anthropic only.

        max_output_tokens caps how much the model writes. force_json was used
        by the old Ollama path to engage Ollama's JSON-only mode; with Anthropic
        we just instruct via the system prompt and let _extract_json clean up
        the response. The boolean is accepted but otherwise ignored.

        On any failure the function returns a string starting with
        LLM_ERROR_SENTINEL ("__OLLAMA_ERROR__") so existing callers' error
        checks continue to work unchanged.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return f"{LLM_ERROR_SENTINEL}: ANTHROPIC_API_KEY not set in environment"

        # Anthropic's Messages API uses a separate `system` parameter rather
        # than embedding system instructions in the prompt. We add a short
        # system message that biases toward concise, well-formed output —
        # matches what the old Ollama bench prompt was relying on the model
        # to do implicitly.
        system_msg = (
            "You are an LLM driving one power in a Diplomacy game. "
            "Follow the instructions in the user message exactly. "
            "When asked for JSON, output JSON only (no prose, no fences)."
        )
        body = json.dumps({
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": system_msg,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.anthropic_url, data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                blocks = data.get("content", [])
                return "\n".join(
                    b.get("text", "") for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                )
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return f"{LLM_ERROR_SENTINEL}: HTTP {e.code} {err_body[:200]}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return f"{LLM_ERROR_SENTINEL}: {e}"

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Find the first complete top-level JSON object in `text` and parse it.

        Tolerant of: leading prose, code fences, trailing commentary, smart
        quotes, trailing commas. Uses balanced-brace scanning rather than
        rfind('}'), so trailing prose containing braces won't break parsing.
        """
        if not text:
            return None
        # Strip code fences
        text = re.sub(r"```(?:json)?", "", text)
        text = text.replace("```", "")
        # Smart quotes -> straight (some models emit these and they break json.loads)
        text = (text.replace("\u201c", '"').replace("\u201d", '"')
                    .replace("\u2018", "'").replace("\u2019", "'"))

        # Scan for the first balanced {...} block.
        # Skip braces that appear inside string literals.
        candidates = []
        i = 0; n = len(text)
        while i < n:
            if text[i] == "{":
                depth = 0; j = i; in_str = False; esc = False
                while j < n:
                    ch = text[j]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                candidates.append(text[i:j + 1])
                                i = j; break
                    j += 1
                else:
                    break  # ran off end with unbalanced braces
            i += 1

        # Try each candidate, prefer longer (richer payloads first)
        candidates.sort(key=len, reverse=True)
        for chunk in candidates:
            for s in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass

        # ----- Partial-recovery path -----
        # Model truncated mid-response (hit num_predict). Find the unfinished
        # top-level object and try to close it intelligently so we recover any
        # complete elements that came before the cutoff.
        first = text.find("{")
        if first == -1:
            return None
        recovered = LLMAgent._recover_truncated_json(text[first:])
        if recovered is not None:
            return recovered
        return None

    @staticmethod
    def _recover_truncated_json(text: str) -> Optional[dict]:
        """Try to salvage a JSON object that was cut off mid-generation.

        Strategy: scan to the last fully-completed element at depth 1 (so a
        full message inside a `messages` array, even if the array itself is
        unfinished), drop everything after it, and synthesize a closing
        `]` and `}` to make the object parseable.
        """
        # Walk the string keeping track of brace/bracket depth and remember
        # the last position where we were at the top-level object (depth 1)
        # AND just finished an element of an array.
        depth_obj = 0      # {} depth
        depth_arr = 0      # [] depth
        in_str = False
        esc = False
        last_safe_cut = -1   # position right after a complete element

        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth_obj += 1
            elif ch == "}":
                depth_obj -= 1
                if depth_obj == 1 and depth_arr == 1:
                    # Just finished a message object inside the messages array
                    last_safe_cut = i + 1
            elif ch == "[":
                depth_arr += 1
            elif ch == "]":
                depth_arr -= 1
            elif ch == "," and depth_obj == 1 and depth_arr == 1:
                # Comma between array elements at depth 1
                last_safe_cut = i  # cut here, drop the comma

        if last_safe_cut == -1:
            return None

        truncated = text[:last_safe_cut].rstrip().rstrip(",")
        # Try closing with ] then } until it parses
        for closing in ("]}", "}]}", "}}"):
            candidate = truncated + closing
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    # -------------------- Negotiation step -------------------- #

    def _board_tag(self, state: GameState) -> str:
        """Telegraphic board view: '1901S|AUS:3,AVIE+ABUD+FTRI|...'"""
        season_code = state.season[0]  # S/F/W
        bits = [f"{state.year}{season_code}"]
        for power in POWERS:
            if power in state.eliminated:
                continue
            scs = supply_centers_owned(state, power)
            units = units_by_power(state, power)
            unit_str = "+".join(f"{u.kind}{u.location}" for u in units) or "-"
            bits.append(f"{power[:3]}:{len(scs)},{unit_str}")
        return "|".join(bits)

    def _messages_digest(self, recent: list[Message]) -> str:
        """Last 6 messages full + 1-line summary of older ones.

        Don't truncate mid-word — that confuses the model. Bound by count, not
        character length. Six full messages average ~1000 chars, well within
        the prompt budget proven by the bench.
        """
        if not recent:
            return "Recent messages: none."
        older = recent[:-6] if len(recent) > 6 else []
        last6 = recent[-6:]
        lines = ["Recent messages:"]
        if older:
            counts = {}
            for m in older:
                counts[m.sender] = counts.get(m.sender, 0) + 1
            summary = ", ".join(f"{s}={n}" for s, n in counts.items())
            lines.append(f"  (older traffic this game: {len(older)} msgs — {summary})")
        for m in last6:
            target = "ALL" if m.public else (",".join(m.recipients) or "me")
            txt = m.text.replace("\n", " ").strip()
            lines.append(f"  {m.sender} → {target}: {txt}")
        return "\n".join(lines)

    def negotiate(self, state: GameState,
                  recent_messages: list[Message]) -> list[Message]:
        sys = self._system_prompt()
        board = self._board_tag(state)
        msgs = self._messages_digest(recent_messages)
        other_powers = [p for p in POWERS if p != self.power and p not in state.eliminated]

        # Telegraphic prompt — bench-proven on gpt-oss:20b.
        # CRITICAL: explicit length caps in instructions. Bench showed the model
        # writes 2000-4000 token responses by default, which max out num_predict
        # and produce unparseable output. Telling it "<=40 words each" + capping
        # num_predict at 700 keeps it tight.
        prompt = f"""{sys}

{board}
{msgs}

TASK=neg-1901{state.season[0]}|out=0-3-msgs|each<=40-words|stay-in-character
JSON only, this exact shape:
{{"messages":[{{"to":["FRA"],"public":false,"text":"..."}},{{"to":[],"public":true,"text":"..."}}]}}
to=array-of-power-codes (full names: {",".join(other_powers)}). public=true means to=[].
Allowed formatting in text: **bold**, [!]warning[/!], emoji."""

        raw = self._call_ollama(prompt, force_json=False, max_output_tokens=700)
        self.last_negotiation_raw = raw
        data = self._extract_json(raw) or {}
        out: list[Message] = []
        valid_powers = set(POWERS)
        # Also accept 3-letter codes the model may have used
        code_to_power = {p[:3]: p for p in POWERS}
        for m in (data.get("messages") or [])[:4]:
            txt = (m.get("text") or "").strip()
            if not txt:
                continue
            raw_recipients = m.get("to") or []
            recipients = []
            for r in raw_recipients:
                ru = str(r).upper().strip()
                if ru in valid_powers:
                    recipients.append(ru)
                elif ru in code_to_power:
                    recipients.append(code_to_power[ru])
            public = bool(m.get("public"))
            out.append(Message(
                sender=self.power,
                recipients=recipients,
                text=txt,
                season=state.season,
                year=state.year,
                public=public or not recipients,
            ))
        # Track communication counts in theory-of-mind
        for msg in out:
            if msg.public:
                continue
            for r in msg.recipients:
                node_id = f"power:{r}"
                if node_id in self.kgs["theory_of_mind"].nodes:
                    n = self.kgs["theory_of_mind"].nodes[node_id]
                    n.attrs["communication_count"] = n.attrs.get("communication_count", 0) + 1
        # Biopsy capture
        if out:
            summary = f"{len(out)} message(s): " + " | ".join(
                f"->{','.join(m.recipients) or 'PUBLIC'}: {m.text[:60]}{'…' if len(m.text)>60 else ''}"
                for m in out
            )
        elif "__OLLAMA_ERROR__" in raw:
            summary = "Ollama call failed."
        elif data == {} or not data:
            summary = "Parsed JSON but empty / no `messages` array."
        else:
            summary = f"Parsed but produced 0 valid messages. data keys: {list(data.keys())}"
        self._record_biopsy(
            step="negotiate",
            year=state.year, season=state.season,
            prompt=prompt,
            raw_response=raw,
            parsed_ok=bool(out),
            parsed_summary=summary,
        )
        return out

    def _orders_unit_block(self, state: GameState) -> str:
        """Per-unit, fill-in-the-blank example orders. Best aid to a small model."""
        my_units = units_by_power(state, self.power)
        if not my_units:
            return "(no units)"
        lines = []
        for u in my_units:
            key = "army" if u.kind == "A" else "fleet"
            neighbors = sorted(ADJ.get(u.location, {}).get(key, []))
            valid = [n for n in neighbors if can_occupy(u.kind, n)]
            if valid:
                ex_move = f"{u.kind} {u.location} - {valid[0]}"
                lines.append(
                    f"  {u.kind} {u.location}: HOLD='{u.kind} {u.location} H'  "
                    f"MOVE example='{ex_move}' (or any of: {', '.join(valid)})"
                )
            else:
                lines.append(f"  {u.kind} {u.location}: must HOLD ('{u.kind} {u.location} H')")
        return "\n".join(lines)

    # -------------------- Orders step -------------------- #

    def decide_orders(self, state: GameState,
                      recent_messages: list[Message]) -> AgentTurnOutput:
        sys = self._system_prompt()
        board = self._board_tag(state)
        msgs = self._messages_digest(recent_messages)
        unit_block = self._orders_unit_block(state)
        my_units = units_by_power(state, self.power)
        unit_sigs = [f"{u.kind} {u.location}" for u in my_units]

        # Telegraphic orders prompt. The unit_block carries filled-in examples
        # for THIS turn's units (small models follow templates better than
        # abstract syntax descriptions).
        prompt = f"""{sys}

{board}
{msgs}

TASK=orders-{state.year}{state.season[0]}|one-order-per-unit|no-prose

YOUR UNITS — order each one of: {", ".join(unit_sigs) or "(none)"}
{unit_block}

ORDER FORMS:
  HOLD     :  A PAR H
  MOVE     :  A PAR - BUR
  SUPPORT  :  A MUN S A KIE - BER     (support a move)
              A PAR S A MAR           (support a hold)
  CONVOY   :  F MAO C A LON - BRE

JSON only:
{{"orders":["A PAR - BUR","F BRE - MAO"],"reflections":{{"trust_updates":[{{"power":"FRANCE","delta":0.1,"reason":"..."}}],"counterfactuals":[{{"label":"...","premise":"...","expected":"..."}}]}}}}

reflections.* may be empty arrays. What matters: orders array (one per unit)."""

        # First pass: free-form. Retry with force_json if parse fails.
        raw = self._call_ollama(prompt, force_json=False, max_output_tokens=800)
        data = self._extract_json(raw)
        if data is None and "__OLLAMA_ERROR__" not in raw:
            raw2 = self._call_ollama(
                prompt + "\n\nReminder: respond with ONE JSON object and nothing else.",
                force_json=True,
                max_output_tokens=800,
            )
            data2 = self._extract_json(raw2)
            if data2 is not None:
                data = data2
                raw = raw2
        output = AgentTurnOutput(raw_response=raw)

        if data is None or "__OLLAMA_ERROR__" in raw:
            # fallback: hold everything
            for u in my_units:
                output.orders.append(Order(power=self.power, unit_kind=u.kind,
                                           location=u.location, type="H"))
            if "__OLLAMA_ERROR__" in raw:
                output.reflection_notes.append("(LLM unavailable — fell back to all-hold)")
                bio_summary = "Ollama call failed — held all units."
            else:
                # Include a snippet so the user can see WHAT the model said.
                snippet = (raw or "").strip().replace("\n", " ")[:400]
                output.reflection_notes.append(
                    "(LLM produced unparseable output — fell back to all-hold)"
                )
                output.reflection_notes.append(f"   raw: {snippet!r}")
                bio_summary = f"Could not parse JSON from response. Held all units."
            self._record_biopsy(
                step="orders",
                year=state.year, season=state.season,
                prompt=prompt,
                raw_response=raw,
                parsed_ok=False,
                parsed_summary=bio_summary,
                notes=list(output.reflection_notes),
            )
            return output

        # Parse orders, tracking what was rejected and why for biopsy
        seen_units = set()
        rejected_reasons: list[str] = []
        for raw_order in data.get("orders", []):
            ro_str = str(raw_order)
            o = parse_order(self.power, ro_str)
            if not o:
                rejected_reasons.append(f"unparseable: {ro_str!r}")
                continue
            if not any(u.location == o.location and u.kind == o.unit_kind
                       for u in my_units):
                rejected_reasons.append(f"no matching unit for: {ro_str!r}")
                continue
            if o.location in seen_units:
                rejected_reasons.append(f"duplicate order for {o.location}: {ro_str!r}")
                continue
            seen_units.add(o.location)
            output.orders.append(o)

        # ensure every unit has an order
        for u in my_units:
            if u.location not in seen_units:
                output.orders.append(Order(
                    power=self.power, unit_kind=u.kind,
                    location=u.location, type="H",
                ))
                output.reflection_notes.append(f"auto-hold inserted for {u.kind} {u.location}")

        # Apply reflections
        reflections = data.get("reflections") or {}
        for tu in reflections.get("trust_updates", []) or []:
            try:
                target = str(tu.get("power", "")).upper()
                delta = float(tu.get("delta", 0.0))
                reason = str(tu.get("reason", ""))
                if target in POWERS and target != self.power:
                    self.kgs.update_trust(target, delta, reason)
                    output.reflection_notes.append(
                        f"trust({target}) {'+' if delta >= 0 else ''}{delta:.2f}: {reason}"
                    )
            except (TypeError, ValueError):
                continue
        for cf in reflections.get("counterfactuals", []) or []:
            try:
                self.kgs.add_counterfactual(
                    label=str(cf.get("label", "unnamed"))[:120],
                    premise=str(cf.get("premise", "")),
                    expected=str(cf.get("expected", "")),
                )
                output.reflection_notes.append(f"counterfactual: {cf.get('label')}")
            except (TypeError, ValueError):
                continue

        # Update strategy plan summary based on most recent reflection
        plan_node = self.kgs["strategy"].nodes.get("plan:active")
        if plan_node:
            plan_node.attrs["last_updated"] = f"{state.year} {state.season}"

        # Biopsy capture on the success path
        accepted = [o.signature() for o in output.orders]
        bio_summary = (
            f"Accepted {len(accepted)} order(s): {accepted}. "
            f"Rejected {len(rejected_reasons)}: {rejected_reasons}"
            if rejected_reasons else
            f"Accepted {len(accepted)} order(s): {accepted}."
        )
        self._record_biopsy(
            step="orders",
            year=state.year, season=state.season,
            prompt=prompt,
            raw_response=raw,
            parsed_ok=True,
            parsed_summary=bio_summary,
            notes=list(output.reflection_notes),
        )

        return output

    # -------------------- Retreat / build orders -------------------- #

    def decide_retreats(self, state: GameState) -> list[Order]:
        my_dislodged = [u for u in state.dislodged if u.power == self.power]
        if not my_dislodged:
            return []
        # Simple heuristic + LLM hint
        orders: list[Order] = []
        for u in my_dislodged:
            key = "army" if u.kind == "A" else "fleet"
            attacker_origin = state.dislodged_from.get(u.location)
            options = [n for n in ADJ.get(u.location, {}).get(key, [])
                       if n != attacker_origin
                       and can_occupy(u.kind, n)
                       and not any(x.location == n for x in state.units)]
            if options:
                orders.append(Order(power=self.power, unit_kind=u.kind,
                                    location=u.location, type="R",
                                    target=options[0]))
            else:
                orders.append(Order(power=self.power, unit_kind=u.kind,
                                    location=u.location, type="D"))
        return orders

    def decide_builds(self, state: GameState) -> list[Order]:
        scs = supply_centers_owned(state, self.power)
        units = units_by_power(state, self.power)
        delta = len(scs) - len(units)
        if delta == 0:
            return []
        from diplomacy_engine import HOME_CENTERS, unit_at
        orders: list[Order] = []
        if delta > 0:
            available = [c for c in HOME_CENTERS[self.power]
                         if state.sc_owner.get(c) == self.power
                         and not unit_at(state, c)]
            for loc in available[:delta]:
                kind = "F" if PROVINCES[loc][0] in ("sea", "coast") else "A"
                # for coastal home centers we pick fleet for England, army else
                if self.power == "ENGLAND":
                    kind = "F"
                elif self.power == "RUSSIA" and loc in ("STP", "SEV"):
                    kind = "F"
                else:
                    kind = "A" if PROVINCES[loc][0] != "sea" else "F"
                orders.append(Order(power=self.power, unit_kind=kind,
                                    location=loc, type="B"))
        else:
            # disband units farthest from home (heuristic)
            home = set(HOME_CENTERS[self.power])
            def far_score(u):
                return 0 if u.location in home else 1
            sorted_units = sorted(units, key=lambda u: -far_score(u))
            for u in sorted_units[:-delta]:
                orders.append(Order(power=self.power, unit_kind=u.kind,
                                    location=u.location, type="D"))
        return orders
