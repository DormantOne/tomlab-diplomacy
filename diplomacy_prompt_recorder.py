"""
diplomacy_prompt_recorder.py — capture LLM prompts/responses per agent.

Wraps an agent's `llm_call` callable so every (prompt, response) pair is
recorded in a bounded deque on the wrapper. Lets us inspect what the LLM
actually saw — essential for diagnosing why the substrate isn't shifting
play, or for sanity-checking that mind content is reaching the prompt.

Design
------
Single chokepoint: every LLM call in DiplomacyAgentV2 — negotiate, orders,
belief-revision proposer, intent-revision proposer, in-line grader calls —
routes through `agent.llm_call`. Wrap that one attribute and you capture
all five flavors with no other code changes.

Use
---

    from diplomacy_prompt_recorder import attach_recorder
    for agent in agents.values():
        attach_recorder(agent, capacity=20)

    # ... run the game ...

    # Later:
    rec = agent.llm_call           # IS the RecordingLLMCall
    for r in rec.history:
        print(r.prompt[:200], "→", r.response[:80])

Idempotent: calling attach_recorder twice on the same agent doesn't
double-wrap (the wrapper detects an already-wrapped llm_call and returns
the existing recorder).

This module is intentionally additive — nothing in diplomacy_agent_v2.py,
diplomacy_mute.py, or run_v2.py changes. The wrapper IS the integration.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ============================================================================
# Records
# ============================================================================


@dataclass
class PromptRecord:
    """One captured (prompt, response) pair."""
    prompt: str
    response: str
    timestamp: float
    elapsed_seconds: float
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# RecordingLLMCall — the wrapper
# ============================================================================


class RecordingLLMCall:
    """Drop-in replacement for an llm_call(prompt) -> str callable.

    Calls through to the wrapped function, captures the (prompt, response)
    pair (plus timing and error info) in a bounded deque. Re-raises the
    original exception unchanged so error semantics don't shift.
    """

    def __init__(
        self, base_call: Callable[[str], str], *, capacity: int = 20,
    ):
        self._base = base_call
        self.history: deque[PromptRecord] = deque(maxlen=capacity)
        self.total_calls = 0
        self.total_errors = 0

    def __call__(self, prompt: str) -> str:
        t0 = time.time()
        rec = PromptRecord(
            prompt=prompt, response="", timestamp=t0, elapsed_seconds=0.0,
        )
        try:
            response = self._base(prompt)
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"
            rec.elapsed_seconds = time.time() - t0
            self.history.append(rec)
            self.total_calls += 1
            self.total_errors += 1
            raise
        rec.response = response
        rec.elapsed_seconds = time.time() - t0
        self.history.append(rec)
        self.total_calls += 1
        return response

    @property
    def base(self):
        """Read-only access to the underlying llm_call (for detach_recorder)."""
        return self._base


# ============================================================================
# attach / detach
# ============================================================================


def attach_recorder(agent, *, capacity: int = 20) -> RecordingLLMCall:
    """Replace agent.llm_call with a RecordingLLMCall wrapper.

    Idempotent: if agent.llm_call is already a RecordingLLMCall, return
    the existing recorder (no double-wrapping).
    """
    if isinstance(agent.llm_call, RecordingLLMCall):
        return agent.llm_call
    rec = RecordingLLMCall(agent.llm_call, capacity=capacity)
    agent.llm_call = rec
    return rec


def detach_recorder(agent) -> None:
    """Restore the underlying llm_call. Safe to call when not attached."""
    if isinstance(agent.llm_call, RecordingLLMCall):
        agent.llm_call = agent.llm_call.base


def is_recording(agent) -> bool:
    return isinstance(getattr(agent, "llm_call", None), RecordingLLMCall)


# ============================================================================
# Serialization
# ============================================================================


def record_to_dict(rec: PromptRecord) -> dict:
    """JSON-clean view of a single record."""
    return {
        "prompt": rec.prompt,
        "response": rec.response,
        "timestamp": rec.timestamp,
        "elapsed_seconds": round(rec.elapsed_seconds, 3),
        "error": rec.error,
        "prompt_length": len(rec.prompt),
        "response_length": len(rec.response),
        "meta": dict(rec.meta),
    }


# ============================================================================
# Call-kind classifier
# ============================================================================
# Best-effort tag for "what kind of LLM call is this?" — purely for GUI
# display. Looks for distinctive markers in the prompt body. If none match,
# returns "unknown" and the user can read the prompt to figure it out.
# Wrong tagging here never gates behavior, just labels in the UI.


_KIND_MARKERS = (
    # Order matters: more-specific markers first.
    ("propose_revision",   "Propose a revised"),
    ("grade_commitments",  "Did the speaker keep"),
    ("grade_predictions",  "Did the prediction come true"),
    ("orders",             "ORDER FORMS"),
    ("orders",             "Decide your orders"),
    ("orders",             "you must issue an order"),
    ("negotiate",          "Compose 0-3 messages"),
    ("negotiate",          "outgoing messages"),
    ("negotiate",          "Compose your messages"),
)


def infer_call_kind(prompt: str) -> str:
    """Best-effort: 'negotiate' | 'orders' | 'grade_*' | 'propose_revision' | 'unknown'."""
    for kind, marker in _KIND_MARKERS:
        if marker in prompt:
            return kind
    return "unknown"


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("PROMPT RECORDER SANITY CHECK")
    print("=" * 72)

    # ---- 1. Basic call-and-record ----
    seen = []
    def stub_llm(prompt: str) -> str:
        seen.append(prompt)
        return f"resp:{prompt[:10]}"

    rec = RecordingLLMCall(stub_llm, capacity=3)
    out = rec("hello world")
    assert out == "resp:hello worl"
    assert len(rec.history) == 1
    assert rec.history[0].prompt == "hello world"
    assert rec.history[0].response.startswith("resp:")
    assert rec.history[0].error is None
    assert rec.history[0].elapsed_seconds >= 0.0
    print(f"  basic call-and-record: 1 record, response={rec.history[0].response!r}")

    # ---- 2. Capacity cap ----
    for i in range(5):
        rec(f"prompt {i}")
    assert len(rec.history) == 3, f"expected 3, got {len(rec.history)}"
    assert rec.total_calls == 6
    # Last 3 prompts should be 'prompt 2', 'prompt 3', 'prompt 4' in some order
    last_prompts = [r.prompt for r in rec.history]
    assert "prompt 4" in last_prompts and "prompt 3" in last_prompts
    print(f"  capacity cap: total_calls={rec.total_calls}, "
          f"history retains last {len(rec.history)}")

    # ---- 3. Error handling ----
    def bad_llm(prompt: str) -> str:
        raise ValueError("simulated upstream failure")
    rec2 = RecordingLLMCall(bad_llm)
    raised = False
    try:
        rec2("test")
    except ValueError as e:
        raised = True
        assert str(e) == "simulated upstream failure"
    assert raised, "exception must be re-raised"
    assert rec2.history[0].error == "ValueError: simulated upstream failure"
    assert rec2.total_errors == 1
    print(f"  error pass-through: {rec2.history[0].error}")

    # ---- 4. attach / detach with a stub agent ----
    class StubAgent:
        def __init__(self):
            self.llm_call = lambda p: f"resp:{p}"

    a = StubAgent()
    base_call = a.llm_call
    r = attach_recorder(a, capacity=5)
    assert isinstance(a.llm_call, RecordingLLMCall)
    assert a.llm_call is r
    assert is_recording(a)
    # Idempotent
    r2 = attach_recorder(a, capacity=5)
    assert r2 is r, "double-attach must not double-wrap"
    a.llm_call("greeting")
    assert len(r.history) == 1
    detach_recorder(a)
    assert a.llm_call is base_call
    assert not is_recording(a)
    # detach when not attached is a no-op
    detach_recorder(a)
    print(f"  attach/detach: idempotent + reversible")

    # ---- 5. Call-kind classifier ----
    assert infer_call_kind("Decide your orders for FRANCE") == "orders"
    assert infer_call_kind("ORDER FORMS\n A PAR HOLD") == "orders"
    assert infer_call_kind("Compose 0-3 messages to other powers") == "negotiate"
    assert infer_call_kind("Did the speaker keep their commitment?") == "grade_commitments"
    assert infer_call_kind("Did the prediction come true?") == "grade_predictions"
    assert infer_call_kind("Propose a revised belief about RUSSIA") == "propose_revision"
    assert infer_call_kind("Random unrelated text") == "unknown"
    print(f"  call-kind classifier: 6 markers + unknown fallback")

    # ---- 6. record_to_dict round-trips ----
    import json as _json
    d = record_to_dict(rec.history[0])
    s = _json.dumps(d)
    assert "prompt" in d and "response" in d and "elapsed_seconds" in d
    print(f"  record_to_dict JSON-clean: {len(s)} bytes")

    print()
    print("Prompt recorder sanity check passed.")
