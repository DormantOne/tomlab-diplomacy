# Helix-style review + changes

Two requests: (1) make the model provider selectable — local Ollama, Anthropic
Haiku, or others, auto-detected from the environment; (2) see whether the KG can
be improved the way the PubMed "helix" lab was. Below is what I found and what I
changed. **All edits are additive/surgical — the game runs exactly as before if
you set `ANTHROPIC_API_KEY` and do nothing else.**

## 1. LLM providers — now selectable + auto-detected

Before, every agent went through `make_default_llm_call()` (Anthropic-only,
hard-required `ANTHROPIC_API_KEY`). There were stray Ollama callers elsewhere
but the live game never used them.

New file **`llm_providers.py`** is the single place that builds the
`llm_call(prompt) -> str` callable for any of:

| kind | needs | default model |
|------|-------|---------------|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `google` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `gemini-1.5-flash` |
| `ollama` | local daemon at `OLLAMA_URL` (default `http://localhost:11434`) | `gpt-oss:20b` |

It's SDK-free (urllib only, like the existing code) and keeps the old error
contract: on any failure a call returns `"{}"` so the JSON parsers degrade
instead of crashing.

**Selection precedence:** explicit choice → env override → auto-detect.

- Auto-detect picks the first *available* provider in order
  anthropic → openai → google → ollama (hosted first, since local was
  historically too weak; Ollama is the fallback when it's all that's running).
- Env override (no UI needed):
  ```
  export DIPLOMACY_LLM_KIND=ollama
  export DIPLOMACY_LLM_MODEL=llama3.1        # optional; else the kind's default
  export OLLAMA_URL=http://localhost:11434   # optional
  ```
- Per-game override from the **New Game** POST: `{"llm_kind": "...", "llm_model": "..."}`.
- New read-only endpoint **`GET /api/llm/providers`** reports what's usable right
  now and what auto-detect would choose — wire it to a dropdown in the New Game
  dialog when you want a clickable selector.

The startup health line now says e.g.
`LLM ready: ollama:llama3.1 for all powers. (providers: ollama=✓ anthropic=✓ openai=— google=—)`.

### Files touched for providers
- `llm_providers.py` *(new)*
- `server/v2_bridge.py` — `make_default_llm_call` now delegates here; added
  `make_default_llm_call_labeled()` returning the resolved (kind, model).
- `server/session.py` — `GameSession` gained `llm_kind` / `llm_model`; resolves
  via the labeled builder, tags agents with the real model label, health log is
  provider-aware.
- `server/app.py` — `/new_game` accepts `llm_kind`/`llm_model`; added
  `/api/llm/providers`.

> Note: `agents/llm_agent.py` is the older Phase-1 path (its `_call_ollama` is
> Anthropic-only). The live GUI game uses the V2 substrate path, which now goes
> through `llm_providers`. Left the legacy file alone to avoid churn; point it at
> `llm_providers.build()` too if you ever revive that path.

## 3. KG — the substrate router is now WIRED INTO THE LIVE MIND

**Finding:** there are two KG layers. `diplomacy_kg_universal.py` is an advanced
"magic_kg" substrate with multi-channel weighted edges (`router[dst][channel] =
weight`), summed `edge_weight`, triple-embedding handles, and decay/compaction.
But the *live* `AgentMind` stored only typed collections; the router was never
fed. The engine was on the shelf with an empty "attach here" hook.

**Now it's connected.** Each agent's mind grows a real, persisted, **decaying
multi-channel graph** as the game plays:

- `AgentMind` gains an `augment` (`AugmentRecord`) — attached lazily at first
  sync, so no schema edit and no circular-import risk.
- `absorb_phase_resolution` (the once-per-phase resolution hook) now calls
  `sync_router_from_mind(mind)` at its end, wrapped so it can never break a turn.
- `sync_router_from_mind` folds the mind's typed state into the substrate via the
  engine's own `add_edge` / `decay_edges`, with two channel flavours:
  - **EVENT** (`promise_in/out`, `kept`, `broken`, `prediction`): folded once per
    (node, status), then decays — a betrayal or kept promise leaves a fading
    memory (verified: a broken-promise edge appears the phase it resolves and
    decays ~0.9×/phase thereafter).
  - **STATE** (`belief`×hp, `suspicion`×weight): re-asserted each phase in
    proportion to current strength, so a persistently held belief stays lit
    (steady-state weight ≈ its hp) while a fading one decays out.
- Belief nodes are also registered as substrate nodes **with embedding handles**,
  so the engine's associative retrieval has real content to match on.

The helix analytics now read this **live** graph (and fall back to on-the-fly
projection only if a mind hasn't synced yet), so `/api/kg/analysis` reflects the
agents' actual decaying memory, not a snapshot recomputed from scratch.

### Tuning knobs (top of `diplomacy_kg_analysis.py`)
`PHASE_DECAY` (0.9 — how fast event memory fades), `PHASE_EPSILON` (0.02 — prune
threshold), `STATE_REINFORCE` (0.1 — belief/suspicion steady-state level), and
`TRUST_SIGN` (how channels combine into the green/gold/red trust scalar).

### Files touched for the KG wiring
- `diplomacy_kg_analysis.py` *(new)* — projection + analytics **+ live sync**.
- `diplomacy_agent_v2.py` — one best-effort `sync_router_from_mind` call at the
  end of `absorb_phase_resolution`.
- `server/app.py` — `/api/kg/analysis` (reads the live graph).

## What the substrate already had (for context)
`diplomacy_kg_universal.py` is, in places, ahead of where the PubMed helix
started — multi-channel edges, decay, embedding handles, preconditions. The only
thing missing was the feed from the live mind, which is what §3 adds.

## Recommended next steps (optional)
1. **Wire `/api/kg/analysis` into the 🕸 Graph lens** so edge colour = composite
   trust and node size = degree (the helix bridge view in the UI).
2. **Learn the channel weights** (`TRUST_SIGN`, `PHASE_*`) from outcomes via the
   prediction-grader — which channel actually predicted betrayal — instead of the
   hand-set priors.
3. **Use the populated router in retrieval/fovea** so the agent *thinks with* the
   decaying trust graph (e.g. bias attention toward high-suspicion powers), not
   just inspects it.
