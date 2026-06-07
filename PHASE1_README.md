# Phase 1 — Substrate Inspection + Theory of Mind Lens

## What this delivers

Three new files. Together they let the GUI see what each AI agent's
substrate `AgentMind` actually contains, with a flagship **Theory of Mind**
view that pivots the data by target power instead of by record type.

| File | Where it goes | What it is |
|---|---|---|
| `diplomacy_inspection.py` | project root, next to `diplomacy_persistence.py` | Full read-only serializer for `AgentMind`. Covers every record kind. Includes the `theory_of_mind_view` aggregator that builds per-target cards. |
| `diplomacy_kg_universal.py` | project root | Stand-alone primitives module mirroring `magic_kg`'s architecture: hashed-BoW embeddings (placeholder), cosine similarity, multi-channel weighted edges, structured precondition predicates, `AugmentRecord` side-car. **Not yet integrated into the typed records** — Phase 2 wires it up. |
| `server/substrate_lenses.py` | `server/` folder, next to `substrate_routes.py` | Flask blueprint with 9 read-only routes under `/api/kg/...`. Reads from the live `session.agents[<power>].mind`; returns shaped JSON for each lens. |

All three files have a `__main__` sanity check that exercises their public
surface end-to-end against synthetic data. Verified passing.

## Routes added

```
GET /api/kg/lenses                  → catalog of available lenses
GET /api/kg/<power>                  → full inspection dict (debug fallback)
GET /api/kg/<power>/beliefs          → grouped by status and by type
GET /api/kg/<power>/predictions      → grouped by status, sorted by confidence
GET /api/kg/<power>/commitments      → incoming + self, grouped by status
GET /api/kg/<power>/intents          → intents + plans + commitments + revisions
GET /api/kg/<power>/identity         → character_brief + identity_constraints
GET /api/kg/<power>/lifecycle        → per-phase telemetry from agent.phase_logs
GET /api/kg/<power>/tom              → FLAGSHIP: per-target ToM cards
```

URL prefix `/api/kg/...` is intentionally distinct from the existing
`/api/substrate/...` routes (which serve the LLM-reconstruction observer)
and from the legacy `/kg/<power>/<graph>` route (which serves the legacy
six-graph KG). All three coexist; nothing was changed.

## How to register the blueprint

One-line edit to `server/app.py`:

```python
from .substrate_lenses import bp_lenses
app.register_blueprint(bp_lenses)
```

That's it. No other source files need to change for Phase 1.

## What Phase 1 deliberately does NOT do

- **Does not modify `diplomacy_kg_schema.py`.** No schema migration, no
  risk to existing saves, no behavior change in any agent.
- **Does not modify `diplomacy_agent_v2.py`.** Agent behavior is identical
  pre- and post-Phase-1.
- **Does not modify `server/session.py`.** The session-migration work
  (legacy `LLMAgent` → `DiplomacyAgentV2`) from the previous handoff is
  separate and still pending. Until that lands, the routes return
  HTTP 409 with `error: no_substrate` — clean and informative, not a
  crash.
- **Does not change the front-end.** No HTML/JS edits. The routes are
  hit-with-curl-able and ready for a JS renderer when you want one. The
  legacy `/kg/<power>/<graph>` route still works, so the existing UI
  keeps rendering as before.
- **Does not wire universal primitives into the typed records.** That's
  Phase 2: attach `handles` / `router` / `precondition` to each
  AgentMind via the `AugmentRecord` side-car, then teach
  `build_fovea` to use `score_node` for retrieval.

## Theory of Mind card shape

For each non-self power, `theory_of_mind_view(mind, valid_powers)` returns:

```jsonc
{
  "viewer": "FRANCE",
  "archetype": "MARSHAL_VEIL",
  "by_target": {
    "RUSSIA": {
      "trust": 0.5,                          // kept / (kept + broken), or null
      "ledger": { "kept": 1, "broken": 1, "pending": 0, "irrelevant": 0 },
      "credibility_belief": { /* the dedicated CREDIBILITY-type belief, if any */ },
      "beliefs": [ /* all beliefs about this target, sorted by hp */ ],
      "beliefs_by_type": {
        "disposition":     [...],
        "tactical_pattern":[...],
        "relationship":    [...],
        "risk_assessment": [...],
        "credibility":     [...]
      },
      "predictions": {
        "open":      [...],   // sorted by confidence desc
        "confirmed": [...],
        "refuted":   [...],
        "partial":   [...]
      },
      "commitments_from_them": [...],   // incoming where speaker == target
      "commitments_to_them":   [...],   // self where target_power == target
      "my_intents_targeting_them": [...],
      "recent_messages_from_them": [...], // last 5
      "recent_messages_to_them":   [...]  // last 5
    },
    "GERMANY": { ... },
    ...
  }
}
```

This is the data the eventual flagship-lens renderer will read. One card
per other power, with a trust gauge, ledger strip, beliefs sorted by
strength, predictions with confirm/refute icons, and the promise ledger
in two columns. The legacy UI's existing CSS variables work for it.

## How to verify Phase 1 in isolation (no game required)

```
cd /path/to/diplomacy
python3 diplomacy_inspection.py        # exercises every per-record helper + ToM view
python3 diplomacy_kg_universal.py      # exercises embeddings, edges, preconditions, AugmentRecord
python3 -m server.substrate_lenses     # spins up a stub Flask app + hits all 9 routes
```

All three should print "passed" and exit 0.

## How to verify after session migration

Once `server/session.py` is migrated to use `DiplomacyAgentV2`:

```
curl http://localhost:5000/api/kg/lenses
curl http://localhost:5000/api/kg/FRANCE/tom | jq '.by_target.RUSSIA.trust'
curl http://localhost:5000/api/kg/FRANCE/lifecycle | jq '.totals'
```

Before migration, those return `409 no_substrate` with a clear message
explaining what's needed.

## What comes next

**Phase 1.5 (the gate, ~80 lines)** — port the magic_kg muted control:
add a per-power KG-advisory mute switch on `GameSession`, branch the
prompt builder on it. Run a 4-game eval comparing full-substrate vs.
muted to validate the substrate is actually shifting play. Without this,
later phases are speculation.

**Phase 2 (the magic_kg core, ~400 lines)** — wire `AugmentRecord` into
each `AgentMind` so every typed record gains universal fields. Implement
`retrieve` / `traverse` / `dream`. Swap the placeholder `embed()` for
`sentence-transformers/all-MiniLM-L6-v2`. Add the memory-selection cap
per game (the anchoring fix). At this point the substrate is doing
magic_kg-shaped work end-to-end.

**Phase 3 (new node kinds, ~300 lines + prompt iteration)** —
`meta_belief` for recursive ToM, `goal_other` and `goal_meta` for
modeled-other goals, `chronology` accumulator, seed `constitution` and
`tool` nodes.

**Phase 4 (cross-game, ~100 lines)** — extend `mind_to_dict` with
`include_in_game=True` for full snapshot save/load, wire `load_mind` into
session start so each archetype's persistent slice loads automatically.
