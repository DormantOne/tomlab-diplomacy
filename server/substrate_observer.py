"""
substrate_observer.py — observe the legacy game and build a substrate-shaped
view of each power's mind on demand.

This module never modifies the legacy game. It reads:
  - session.messages    (full history of in-game communication)
  - session.state       (current engine state)
  - session.log         (per-phase event log, e.g. "Spring 1901: A VIE -> BOH (succeeds)")
  - session.agents[p]   (for archetype + personality info)

For each power, we send ONE Anthropic call asking the model to reconstruct
that power's substrate-shaped mental state — beliefs about other powers,
in-flight strategic intents, predictions about near-term moves, and a
credibility ledger (kept/broken commitments). The result is returned in
the same dict shape as `write_agent_snapshot`, so the existing substrate
renderer can display it.

Design choices:
  - One call per power (NOT 6 separate per-phase calls). The model gets
    full history at once so it can reason about temporal patterns.
  - JSON-structured output, validated and clamped.
  - Run in background thread; status is polled by the UI.
  - Failures are caught per-power, so one bad response doesn't kill the
    whole build.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# LLM callers — Anthropic and Ollama
#
# These are separate from the legacy Ollama agent's call path. They share the
# same Ollama URL but use a longer context window and bigger output budget
# tuned for substrate-style reconstruction prompts (which can be long).
# ============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"


def _anthropic_call(prompt: str, model: str = "claude-haiku-4-5-20251001",
                    api_key: Optional[str] = None,
                    max_tokens: int = 4096,
                    timeout: float = 120.0) -> str:
    """Call the Anthropic Messages API. Returns response text or raises."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        blocks = data.get("content", [])
        return "\n".join(
            b.get("text", "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )


def _ollama_call(prompt: str, model: str = "gpt-oss:20b",
                 url: str = OLLAMA_URL,
                 num_ctx: int = 8192,
                 num_predict: int = 2000,
                 timeout: float = 240.0) -> str:
    """Call a local Ollama server. Returns response text or raises.

    Uses a larger context window and output budget than the legacy game's
    in-game calls — substrate prompts are long (full game history) and
    the JSON response can be sizeable. Forces JSON mode for clean parsing.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")


def _llm_call(prompt: str, model: str) -> str:
    """Dispatch to Anthropic or Ollama based on the model name.

    Models starting with 'claude-' are routed to Anthropic. Anything else
    (e.g. 'gpt-oss:20b', 'llama3.2:3b') is routed to Ollama.
    """
    if model.startswith("claude-"):
        return _anthropic_call(prompt, model=model)
    return _ollama_call(prompt, model=model)


def _extract_json(text: str) -> Optional[dict]:
    """Find and parse a JSON object embedded in `text`. Returns None on fail.

    Strategy:
      1. If a ```json``` (or ```) fence is present, extract that block whole
         (no regex sub-matching — that breaks on nested braces).
      2. Otherwise, scan the text character by character with brace counting
         that ignores braces inside JSON string literals. Try parsing each
         balanced { ... } we find until one parses successfully.
      3. If everything fails, return None.

    The balanced-brace scanner is essential because Haiku/Opus often wrap
    the JSON in commentary ("Here is the analysis: { ... }. Note: ..."),
    and naive regexes either miss the body or stop at the first `}`.
    """
    if not text:
        return None

    # Step 1: Look for fenced code blocks. Take the contents whole.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # Fall through to balanced-brace scanning of the candidate
            text_to_scan = candidate
        else:
            text_to_scan = text
    else:
        text_to_scan = text

    # Step 2: Balanced-brace scan that ignores braces inside string literals.
    # Walk forward; track depth; when depth returns to 0, try parsing the slice.
    n = len(text_to_scan)
    i = 0
    while i < n:
        if text_to_scan[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        for j in range(i, n):
            ch = text_to_scan[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text_to_scan[i:j+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break  # try next opening brace
        i += 1
    return None


# ============================================================================
# Prompt template
# ============================================================================

# We keep this self-contained so we don't depend on the substrate's many
# moving parts. The LLM sees a transcript of the legacy game and is asked
# to retrofit a substrate-shaped mind.

SUBSTRATE_OBSERVER_PROMPT = """You are reconstructing the mental state of a Diplomacy player ({power}) from the game history below.

The player's archetype is **{archetype}**: {archetype_desc}

Your job: given the messages and moves that have happened so far, output a JSON object describing what {power} *probably believes, intends, predicts, and remembers* right now. Be specific and grounded in the actual events. Do not invent facts that aren't visible in the transcript.

# GAME HISTORY

{history_text}

# CURRENT BOARD

{board_text}

# OUTPUT FORMAT

Output ONLY a JSON object with this exact shape:

```json
{{
  "beliefs": [
    {{
      "id": "belief:1",
      "about": "RUSSIA",
      "type": "DISPOSITION|CREDIBILITY|TACTICAL|RELATIONSHIP|RISK",
      "status": "active|proto",
      "head": "<one sentence summary of the belief>",
      "body": "<2-3 sentences of detail and grounding evidence>",
      "hp": 0.7,
      "evidence_for_count": 2,
      "evidence_against_count": 0,
      "formed_at": "<phase like 1901-SPRING-MOVES>",
      "last_updated": "<phase>"
    }}
  ],
  "strategic_intents": [
    {{
      "id": "intent:1",
      "head": "<one sentence summary of the intent>",
      "body": "<2-3 sentence elaboration>",
      "status": "active|proto",
      "horizon": "near|medium|long",
      "target_powers": ["GERMANY"],
      "target_provinces": ["RUH", "BEL"],
      "formed_at": "<phase>",
      "hp": 0.6
    }}
  ],
  "incoming_commitments": [
    {{
      "id": "cmt:1",
      "speaker": "GERMANY",
      "type": "non_aggression|alliance_for|move_to|not_move_to|support_to|demilitarize",
      "subject_province": "BUR",
      "target_province": null,
      "deadline_phase": "<phase like 1902-SPRING-MOVES>",
      "status": "pending|kept|broken",
      "raw": "<the actual quote or paraphrase from the message>"
    }}
  ],
  "self_commitments": [
    {{
      "id": "cmt:101",
      "to": "RUSSIA",
      "type": "non_aggression|move_to|...",
      "subject_province": null,
      "target_province": "GAL",
      "deadline_phase": "<phase>",
      "status": "pending|kept|broken",
      "raw": "<paraphrase>"
    }}
  ],
  "recent_predictions": [
    {{
      "id": "pred:1",
      "about": "TURKEY",
      "type": "move_to|attack|hold|support",
      "target": "BUL",
      "window_kind": "near|medium",
      "prediction_window": "<phase>",
      "status": "open|confirmed|refuted",
      "confidence": 0.6,
      "rationale": "<1-2 sentences why this prediction>",
      "formed_at": "<phase>"
    }}
  ],
  "summary": {{
    "beliefs_active": 0,
    "beliefs_proto": 0,
    "beliefs_retired": 0,
    "intents_active": 0,
    "intents_total": 0,
    "intents_succeeded": 0,
    "incoming_commitments_kept": 0,
    "incoming_commitments_broken": 0,
    "incoming_commitments_pending": 0,
    "predictions_confirmed": 0,
    "predictions_refuted": 0,
    "predictions_open": 0
  }}
}}
```

# GUIDANCE

- **Beliefs** are durable mental models of *what kind of player* another power is. DISPOSITION beliefs are about temperament ("Russia is opportunistic"); CREDIBILITY about whether they keep their word; TACTICAL about their preferred openings; RELATIONSHIP about how they treat {power} specifically.
- **Strategic intents** are {power}'s own plans — the goals driving their orders. Should be 2-5 of them.
- **Incoming commitments** are promises others made TO {power}. Look for messages where someone said they would or wouldn't do something. Use the `status` field: KEPT if the deadline passed and they did it, BROKEN if they reneged, PENDING if still in flight.
- **Self commitments** are promises {power} made TO others. Same status rules.
- **Recent predictions** are {power}'s read on what specific powers will do in the near future. Should be 3-8.
- Fill the `summary` block with counts that match the arrays above.
- Use phase strings in this format: `1901-SPRING-MOVES`, `1901-FALL-MOVES`, etc.
- If you have very low evidence for something, use status `proto` (provisional) rather than `active` (confirmed).
- DO NOT include retired beliefs unless they were clearly disproven by events.
- Be SPECIFIC. "Russia is sneaky" is bad. "Russia talks alliance with Austria but moved A WAR-GAL on turn 1, suggesting opportunistic positioning" is good.

Output the JSON object now, with no other text before or after.
"""


ARCHETYPE_DESCRIPTIONS = {
    "MARSHAL_VEIL": "I plan in arcs. I keep my word when watched and remember when others do not.",
    "CARDINAL_FOX": "I trade in stories and what they imply. I prefer a beautiful turn to a safe one.",
    "PARSON_HAWTHORNE": "My word is given carefully and kept absolutely.",
    "BARON_KORVIN": "I trust no one before they have earned it twice.",
    "ARCHITECT_LIRA": "I look at the whole table and design the equilibrium I prefer.",
}


# ============================================================================
# History formatting
# ============================================================================

def _phase_key_from_msg(m) -> str:
    """Convert a Message's season/year/phase fields into substrate phase notation."""
    season = (m.season or "SPRING").upper()
    year = m.year or 1901
    return f"{year}-{season}-MOVES"  # only movement phases generate messages


def _format_history(messages: list, log_lines: list[str], power: str) -> str:
    """Build a chronological text rendering of messages and resolved phases."""
    sections = []

    # Group messages by phase
    by_phase = {}
    for m in messages:
        if m.public or power in m.recipients or m.sender == power:
            key = _phase_key_from_msg(m)
            by_phase.setdefault(key, []).append(m)

    # Phases that show up in either messages or logs
    log_phase_re = re.compile(r"^=== (\d+) (\w+) (\w+) ===")
    log_by_phase = {}
    current_phase = None
    for line in log_lines:
        match = log_phase_re.match(line.strip())
        if match:
            year, season, phase_kind = match.groups()
            current_phase = f"{year}-{season}-{phase_kind}"
            log_by_phase.setdefault(current_phase, []).append(line)
        elif current_phase:
            log_by_phase[current_phase].append(line)

    all_phases = sorted(set(list(by_phase.keys()) + list(log_by_phase.keys())),
                        key=_phase_sort_key)

    for phase in all_phases:
        sections.append(f"\n## {phase}")

        # Messages this phase that this power can see
        msgs = by_phase.get(phase, [])
        if msgs:
            sections.append("\n### Messages visible to " + power)
            for m in msgs:
                if m.public:
                    addr = "PUBLIC"
                elif power in m.recipients and m.sender != power:
                    addr = f"FROM {m.sender} TO " + ", ".join(m.recipients)
                elif m.sender == power:
                    addr = f"FROM YOU ({power}) TO " + ", ".join(m.recipients)
                else:
                    continue
                text = m.text.strip().replace("\n", " ")
                if len(text) > 600:
                    text = text[:600] + "…"
                sections.append(f"  - [{addr}] {text}")

        # Resolution log lines for this phase
        log_lines_p = log_by_phase.get(phase, [])
        if log_lines_p:
            sections.append("\n### What resolved this phase")
            for line in log_lines_p[:60]:  # cap noise
                sections.append(f"  {line}")

    if not sections:
        return "(No history yet — game just started.)"
    return "\n".join(sections)


def _phase_sort_key(phase: str) -> tuple:
    season_order = {"SPRING": 0, "FALL": 1, "WINTER": 2}
    kind_order = {"MOVES": 0, "MOVEMENT": 0, "RETREATS": 1, "RETREAT": 1,
                  "ADJUSTMENTS": 2, "ADJUSTMENT": 2}
    parts = phase.split("-")
    if len(parts) != 3:
        return (9999, 9, 9)
    try:
        year = int(parts[0])
    except ValueError:
        year = 9999
    return (year, season_order.get(parts[1], 9), kind_order.get(parts[2], 9))


def _format_board(state) -> str:
    """Render current board state as a brief per-power summary."""
    from diplomacy_engine import POWERS, supply_centers_owned, units_by_power
    lines = []
    for power in POWERS:
        if power in state.eliminated:
            lines.append(f"  {power}: eliminated")
            continue
        scs = supply_centers_owned(state, power)
        units = units_by_power(state, power)
        unit_str = ", ".join(f"{u.kind}{u.location}" for u in units) or "(no units)"
        lines.append(f"  {power}: {len(scs)} SC | {unit_str}")
    return "\n".join(lines)


# ============================================================================
# Substrate observer
# ============================================================================

@dataclass
class BuildStatus:
    in_progress: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    total: int = 0
    completed: int = 0
    errors: dict = field(default_factory=dict)        # power -> error string
    last_build_phase: Optional[str] = None             # phase the build was for
    last_model: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "in_progress": self.in_progress,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total": self.total,
            "completed": self.completed,
            "errors": dict(self.errors),
            "last_build_phase": self.last_build_phase,
            "last_model": self.last_model,
        }


class SubstrateObserver:
    """Builds substrate-shaped mind dicts on demand for the legacy session."""

    def __init__(self):
        self.lock = threading.Lock()
        self.minds: dict[str, dict] = {}            # power -> snapshot dict
        # Biopsies: per-power record of the last build attempt. Always populated,
        # successful or not, so the user can inspect what the LLM was given and
        # what came back when something fails.
        # Shape: power -> {prompt, raw_response, error, parsed_ok, model,
        #                  timestamp, build_phase, response_length, prompt_length}
        self.biopsies: dict[str, dict] = {}
        self.status: BuildStatus = BuildStatus()
        self.worker: Optional[threading.Thread] = None

    def trigger_build(self, session, *, model: str = "claude-haiku-4-5-20251001"
                     ) -> dict:
        """Kick off a new build in a background thread. Returns current status."""
        with self.lock:
            if self.status.in_progress:
                return self.status.to_dict()
            from diplomacy_engine import POWERS
            self.status = BuildStatus(
                in_progress=True,
                started_at=time.time(),
                total=len(POWERS),
                completed=0,
                errors={},
                last_model=model,
            )
        # Snapshot the data we need from the session (so the worker doesn't
        # race with ongoing legacy game updates).
        snap = self._capture_session_snapshot(session)

        def _worker():
            try:
                self._do_build(snap, model)
            except Exception as e:
                with self.lock:
                    self.status.errors["_global"] = str(e)
                    self.status.in_progress = False
                    self.status.finished_at = time.time()

        self.worker = threading.Thread(target=_worker, daemon=True)
        self.worker.start()
        return self.status_payload()

    def status_payload(self) -> dict:
        with self.lock:
            return self.status.to_dict()

    def get_mind(self, power: str) -> Optional[dict]:
        with self.lock:
            return self.minds.get(power)

    def all_minds(self) -> dict:
        with self.lock:
            return dict(self.minds)

    def get_biopsy(self, power: str) -> Optional[dict]:
        """Return the last build biopsy for `power` (success or failure)."""
        with self.lock:
            return self.biopsies.get(power)

    def all_biopsies(self) -> dict:
        """All biopsies keyed by power."""
        with self.lock:
            return dict(self.biopsies)

    # --- internals -----------------------------------------------------------

    def _capture_session_snapshot(self, session) -> dict:
        """Make a thread-safe copy of the bits of the session we need."""
        from diplomacy_engine import POWERS
        # Copy the messages list (shallow copy of refs is fine — Message is a
        # frozen-ish dataclass, won't be mutated)
        messages = list(session.messages)
        log = list(session.log)
        state = session.state
        # Phase string for tagging the build
        phase = f"{state.year}-{state.season.upper()}-{state.phase.upper()}"
        # Per-power metadata
        powers_meta = {}
        for p in POWERS:
            agent = session.agents.get(p)
            if agent is None:
                # User power
                powers_meta[p] = {
                    "archetype": "PLAYER_DEFAULT",
                    "is_user": True,
                }
            else:
                powers_meta[p] = {
                    "archetype": getattr(agent, "personality_key", "PLAYER_DEFAULT"),
                    "is_user": False,
                }
        return {
            "messages": messages,
            "log": log,
            "state": state,
            "phase": phase,
            "powers_meta": powers_meta,
            "powers": list(POWERS),
        }

    def _do_build(self, snap: dict, model: str) -> None:
        """Run one substrate-build LLM call per power, sequentially, with errors caught per power."""
        history_text = _format_history(snap["messages"], snap["log"], "ALL")
        # We rebuild history per power so messages visible to that power are
        # filtered correctly. (This is cheap.)
        for power in snap["powers"]:
            if snap["powers_meta"][power]["is_user"]:
                # Skip user power — no archetype, no need to reconstruct
                with self.lock:
                    self.status.completed += 1
                continue
            # Build prompt outside the try block so we can biopsy even on
            # exceptions during prompt construction.
            try:
                power_history = _format_history(snap["messages"], snap["log"], power)
                board_text = _format_board(snap["state"])
                archetype = snap["powers_meta"][power]["archetype"]
                arch_desc = ARCHETYPE_DESCRIPTIONS.get(archetype,
                                                      "An LLM-driven Diplomacy player.")
                prompt = SUBSTRATE_OBSERVER_PROMPT.format(
                    power=power, archetype=archetype,
                    archetype_desc=arch_desc,
                    history_text=power_history,
                    board_text=board_text,
                )
            except Exception as e:
                with self.lock:
                    self.status.errors[power] = f"prompt construction: {e}"
                    self.status.completed += 1
                    self.biopsies[power] = {
                        "power": power,
                        "model": model,
                        "build_phase": snap["phase"],
                        "timestamp": time.time(),
                        "prompt": "",
                        "prompt_length": 0,
                        "raw_response": "",
                        "response_length": 0,
                        "error": f"prompt construction: {e}",
                        "parsed_ok": False,
                    }
                print(f"[substrate observer] {power}: prompt construction: {e}",
                      file=sys.stderr)
                continue

            response = ""
            error = None
            parsed_ok = False
            snapshot = None
            try:
                response = _llm_call(prompt, model=model)
                parsed = _extract_json(response)
                if parsed is None:
                    error = "could not extract JSON from response"
                else:
                    snapshot = self._build_snapshot_dict(power, archetype,
                                                        snap["phase"], parsed)
                    parsed_ok = True
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"[substrate observer] {power}: {error}", file=sys.stderr)

            # Always record a biopsy and update status under the lock
            with self.lock:
                self.biopsies[power] = {
                    "power": power,
                    "model": model,
                    "build_phase": snap["phase"],
                    "timestamp": time.time(),
                    "prompt": prompt,
                    "prompt_length": len(prompt),
                    "raw_response": response,
                    "response_length": len(response),
                    "error": error,
                    "parsed_ok": parsed_ok,
                }
                if parsed_ok and snapshot is not None:
                    self.minds[power] = snapshot
                else:
                    self.status.errors[power] = error or "unknown failure"
                self.status.completed += 1

        with self.lock:
            self.status.in_progress = False
            self.status.finished_at = time.time()
            self.status.last_build_phase = snap["phase"]

    def _build_snapshot_dict(self, power: str, archetype: str, phase: str,
                             parsed: dict) -> dict:
        """Convert the LLM's parsed JSON into a snapshot dict shaped exactly like
        write_agent_snapshot's output (so the existing renderer works unchanged)."""
        # Defensively get arrays
        beliefs = self._clamp_list(parsed.get("beliefs"), 30)
        intents = self._clamp_list(parsed.get("strategic_intents"), 12)
        incoming = self._clamp_list(parsed.get("incoming_commitments"), 30)
        self_c = self._clamp_list(parsed.get("self_commitments"), 30)
        preds = self._clamp_list(parsed.get("recent_predictions"), 30)
        summary = parsed.get("summary") or {}

        # Compute summary fields that may be missing
        def _count(arr, key, val):
            return sum(1 for x in arr if (x or {}).get(key) == val)
        summary.setdefault("beliefs_total", len(beliefs))
        summary.setdefault("beliefs_active", _count(beliefs, "status", "active"))
        summary.setdefault("beliefs_proto", _count(beliefs, "status", "proto"))
        summary.setdefault("beliefs_retired", _count(beliefs, "status", "retired"))
        summary.setdefault("intents_total", len(intents))
        summary.setdefault("intents_active", _count(intents, "status", "active"))
        summary.setdefault("intents_succeeded", _count(intents, "status", "succeeded"))
        summary.setdefault("incoming_commitments_kept",
                          _count(incoming, "status", "kept"))
        summary.setdefault("incoming_commitments_broken",
                          _count(incoming, "status", "broken"))
        summary.setdefault("incoming_commitments_pending",
                          _count(incoming, "status", "pending"))
        summary.setdefault("predictions_total", len(preds))
        summary.setdefault("predictions_open", _count(preds, "status", "open"))
        summary.setdefault("predictions_confirmed",
                          _count(preds, "status", "confirmed"))
        summary.setdefault("predictions_refuted",
                          _count(preds, "status", "refuted"))
        summary.setdefault("plan_nodes_total", 0)
        summary.setdefault("intent_commitments_active", 0)

        # Re-id consistently and patch fields the renderer expects
        beliefs_out = []
        for i, b in enumerate(beliefs):
            beliefs_out.append({
                "id": b.get("id") or f"belief:{i+1}",
                "about": (b.get("about") or "").upper(),
                "type": (b.get("type") or "DISPOSITION"),
                "status": b.get("status") or "proto",
                "head": str(b.get("head") or "")[:240],
                "body": str(b.get("body") or "")[:600],
                "hp": float(b.get("hp", 0.5)),
                "critic_score": float(b.get("critic_score", 0.0)),
                "times_foveated": int(b.get("times_foveated", 0)),
                "evidence_for_count": int(b.get("evidence_for_count", 0)),
                "evidence_against_count": int(b.get("evidence_against_count", 0)),
                "retire_reason": b.get("retire_reason"),
                "formed_at": b.get("formed_at") or phase,
                "last_updated": b.get("last_updated") or phase,
                "persists_across_games": False,
            })

        intents_out = []
        for i, n in enumerate(intents):
            intents_out.append({
                "id": n.get("id") or f"intent:{i+1}",
                "head": str(n.get("head") or "")[:240],
                "body": str(n.get("body") or "")[:600],
                "status": n.get("status") or "active",
                "target_powers": [str(x).upper() for x in (n.get("target_powers") or [])],
                "target_provinces": [str(x).upper() for x in (n.get("target_provinces") or [])],
                "horizon": n.get("horizon") or "medium",
                "formed_at": n.get("formed_at") or phase,
                "active_since": n.get("active_since") or phase,
                "hp": float(n.get("hp", 0.5)),
                "critic_score": float(n.get("critic_score", 0.0)),
                "sc_delta_under_intent": int(n.get("sc_delta_under_intent", 0)),
                "predictions_confirmed": int(n.get("predictions_confirmed", 0)),
                "predictions_refuted": int(n.get("predictions_refuted", 0)),
                "supporting_plan_count": int(n.get("supporting_plan_count", 0)),
                "times_committed": int(n.get("times_committed", 0)),
                "retire_reason": n.get("retire_reason"),
            })

        incoming_out = []
        for i, c in enumerate(incoming):
            incoming_out.append({
                "id": c.get("id") or f"cmt:{i+1}",
                "speaker": (c.get("speaker") or "").upper(),
                "type": c.get("type") or "non_aggression",
                "subject_unit": c.get("subject_unit"),
                "subject_province": c.get("subject_province"),
                "target_province": c.get("target_province"),
                "counterparty": c.get("counterparty"),
                "deadline_phase": c.get("deadline_phase") or phase,
                "status": c.get("status") or "pending",
                "resolved_at_phase": c.get("resolved_at_phase"),
                "evidence_count": int(c.get("evidence_count", 0)),
                "raw": str(c.get("raw") or "")[:240],
            })

        self_out = []
        for i, c in enumerate(self_c):
            self_out.append({
                "id": c.get("id") or f"cmt:s{i+1}",
                "to": (c.get("to") or "").upper(),
                "type": c.get("type") or "non_aggression",
                "subject_province": c.get("subject_province"),
                "target_province": c.get("target_province"),
                "deadline_phase": c.get("deadline_phase") or phase,
                "status": c.get("status") or "pending",
                "raw": str(c.get("raw") or "")[:240],
            })

        preds_out = []
        for i, p in enumerate(preds):
            preds_out.append({
                "id": p.get("id") or f"pred:{i+1}",
                "about": (p.get("about") or "").upper(),
                "type": p.get("type") or "move_to",
                "target": p.get("target"),
                "subject_power": p.get("subject_power"),
                "window_kind": p.get("window_kind") or "near",
                "prediction_window": p.get("prediction_window") or phase,
                "formed_at": p.get("formed_at") or phase,
                "status": p.get("status") or "open",
                "confidence": float(p.get("confidence", 0.5)),
                "rationale": str(p.get("rationale") or "")[:400],
            })

        return {
            "owner_power": power,
            "archetype": archetype,
            "phase": phase,
            "snapshot_time": time.time(),
            "summary": summary,
            "beliefs": beliefs_out,
            "strategic_intents": intents_out,
            "intent_commitments": [],
            "incoming_commitments": incoming_out,
            "self_commitments": self_out,
            "recent_predictions": preds_out,
        }

    @staticmethod
    def _clamp_list(x, max_len: int) -> list:
        if not isinstance(x, list):
            return []
        return [item for item in x[:max_len] if isinstance(item, dict)]
