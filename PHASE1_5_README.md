# Phase 1.5 — Muted-Control Ablation Harness

## Purpose

The gate. Phase 1.5 measures whether the substrate's KG advisory actually
shifts play. If full-substrate games produce indistinguishable metrics from
all-muted games (across SC counts, kept-commitment rates, prediction
confirm rates), the substrate isn't paying off and Phases 2+ are
speculation. **Stop and rethink before building further.**

If the gap exists, every later phase has a measurable target.

## What this delivers

| File | Where | What |
|---|---|---|
| `diplomacy_mute.py` | project root, next to `diplomacy_agent_v2.py` | `MutableAgent` subclass with `kg_advisory_mute` switch; helpers to make muted minds, transfer writes, build muted-aware agent dicts, and extract eval metrics. |
| `eval_ablation.py` | project root, next to `run_v2.py` | CLI harness: runs N games per condition, aggregates, prints a comparison table, optionally writes per-game JSON. |

Both files have a `__main__` sanity check (or `--self-test` flag) that runs
end-to-end against the stub LLM in under a second. Verified passing.

## How muting works (brief recap)

A muted agent runs the same LLM call cadence as a normal agent — same
`negotiate()` and `decide_orders()` cycle. The only difference: the prompts
for those calls are built against an **empty `AgentMind`** preserving only
the identity layer (`character_brief`, `identity_constraints`,
`games_played`).

Writes still happen. The protocol's `negotiate()` writes
`message_events` and `self_commitments`; `order_decision()` writes
`proto_belief`, `proto_intent`, `predictions`, and `plan_nodes`. We let
those writes land in the muted scratch mind, then forward them back to the
real mind. The agent's KG keeps growing — we just deny it READ access
during prompt construction.

`intake_message()` and `absorb_phase_resolution()` are NOT muted — they're
pure-write paths that always run on the real mind. `decide_retreats()` and
`decide_builds()` are heuristic (no LLM call) and need no muting.

This is the magic_kg pattern (`kg_advisory_mute` per-player, line 36-48 of
`magic_kg/app.py`). The diff between muted and full play is exactly the
effect of KG advisory on each individual move — same prompt structure,
same call cadence, same write paths.

## Conditions

```
full        all 6 powers run with full KG advisory
all_muted   all 6 powers muted (control: substrate OFF, identity-only)
mute_K      K random powers muted, rest full (the gradient; K ∈ [0..6])
```

For `mute_K`, the K muted powers are sampled with the seed you pass
(`--seed`, default `0xD1`) so runs are reproducible.

The within-condition comparison (e.g. `mute_2`: muted vs unmuted bucket)
gives you a same-game contrast — same board state, different KG access. The
across-condition comparison (`full.unmuted` vs `all_muted.muted`) gives
the same-condition contrast — every power has the same KG access, gap is
between games. Both are useful; the first is more controlled, the second
is what you'd report as a headline.

## Metrics extracted per agent

```
sc_count              final supply-center count (proxy for "won the game")
eliminated            bool
kept_rate             self-commitments kept / (kept + broken)         ← reputation signal
confirm_rate          predictions confirmed / (confirmed + refuted)   ← model-quality signal
beliefs               raw count of belief nodes at end of game
predictions           raw count of prediction nodes
self_commitments      raw count of own promises
strategic_intents     raw count
```

`kept_rate` is the headline reputation signal — does having a KG help an
agent keep its word? `confirm_rate` is the headline model-quality signal —
does having a KG help an agent predict opponents better? `sc_count` is the
ultimate scorecard but very high-variance per game.

## Usage

### Cheap stub-LLM smoke test (no API key required, ~2 sec)

```
python eval_ablation.py --games 2 --phases 4 --conditions full,all_muted
```

Stub will produce identical numbers in every bucket because it doesn't
read mind content — that's correct, and confirms the harness machinery
doesn't accidentally distort outcomes.

### Real experiment with Anthropic Haiku

```
export ANTHROPIC_API_KEY=...
python eval_ablation.py --llm anthropic \
    --games 6 --phases 12 \
    --conditions full,all_muted \
    --out gate_results.json
```

Cost estimate: ~6 powers × ~12 phases × ~2 LLM calls per phase per power
× ~$0.0008 per Haiku call ≈ **~$0.70 per game**, ~$8 for the full 12-game
gate run. Wall time at Haiku speeds: roughly 2 minutes per game, so ~24
minutes total.

If you want a bigger N, double `--games`. If you want longer games (more
opportunity for promises and predictions to resolve), bump `--phases` to
16 or 20.

### Including the gradient

```
python eval_ablation.py --llm anthropic \
    --games 4 --phases 12 \
    --conditions full,mute_2,all_muted \
    --out gradient_results.json
```

Three conditions × 4 games × 6 powers = 72 metric rows; enough to see a
trend if one exists.

## How to read the headline result

After running with anthropic, look at the table. The two rows that
matter most:

```
condition       bucket    mean_sc   kept_rt   conf_rt
------------------------------------------------------
full           unmuted     X.XX     0.YY      0.ZZ
all_muted      muted       X.XX     0.YY      0.ZZ
```

If `full.unmuted.kept_rt` is meaningfully higher than `all_muted.muted.kept_rt`
(say, >0.05 absolute, with 24+ rows of data each), the substrate is helping
agents honor their word. Same logic for `confirm_rt`. SC counts are noisier;
expect to need 8+ games per condition to see a real signal there.

If neither metric moves: that's the signal that something's broken
upstream — either the prompt isn't actually surfacing KG content
effectively, or the KG content isn't yet rich enough to matter. Investigate
before Phase 2.

## What's NOT done in Phase 1.5

- **No prompt changes.** The protocol still builds prompts from whatever
  mind it's handed. We mute by handing it an empty mind, not by editing
  the prompt template.
- **No statistical testing.** The aggregator reports means; significance
  testing is on you (or the next phase's tooling).
- **No session.py integration.** This harness drives games via
  `run_v2.run_one_phase` directly — it doesn't touch `server/session.py`.
  Adding a per-power mute toggle to the live GUI is straightforward later
  (one bool field on `GameSession`, one route to set it).
- **No persistent metric tracking across runs.** Each invocation is
  standalone. Pass `--out` to save raw results for later comparison if
  you're doing a series.

## Wiring for the live GUI (later, ~30 lines, not Phase 1.5)

When you want a per-power mute switch in the running game:

1. Add `kg_advisory_mute: set[str] = field(default_factory=set)` to
   `GameSession`.
2. In `__post_init__`, call `make_mutable_agents(...)` instead of the
   current factory.
3. Add a Flask route `POST /api/kg/<power>/mute` and `DELETE
   /api/kg/<power>/mute` that calls `set_mute(session.agents, ...)`.
4. Optional: tiny JS toggle in the UI that calls the route.

Total: ~30 lines, no change to anything in Phase 1.5 itself.

## Verification

Both files have built-in tests:

```
python diplomacy_mute.py             # 5 sanity tests, ~1 sec
python eval_ablation.py --self-test  # 2-condition × 1-game × 2-phase smoke, ~1 sec
```

Both should print "passed" and exit 0.
