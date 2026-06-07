# Diplomacy Lab — Dreaming + Force Graph + Biopsy on Disk

This drop adds:

1. **Per-phase biopsy saved to disk.** Every snapshot folder now also contains `biopsy-NNN-PHASE.json` files with the full prompt history for every agent at that phase. So even if the GUI biopsy modal misbehaves, the data is on disk and can be reviewed (or sent to me for review).
2. **Download JSON button on biopsy modal.** Right next to the power picker. Click it, get the full biopsy bundle as a downloaded JSON. Works regardless of how the rest of the modal renders.
3. **Dreaming.** After every FALL ADJUSTMENT (end of game year), each non-eliminated agent gets one extra LLM call where it consolidates the year — reviews accumulated beliefs, reviews credibility data, looks for patterns, plans for next year. Outputs a long journal entry plus optional belief HP adjustments. Stored under `private_journal` with `kind: "dream-consolidation"`.
4. **🕸 Graph view lens.** New 7th lens in the inspector. Renders the current power's mind as a force-directed graph: that power at center, other powers around the ring, edges colored by promise-keeping (green/gold/red), edge thickness proportional to volume. Click any node to see top beliefs.
5. **💭 Dreams lens.** New 8th lens. Shows year-end consolidations as full-text cards, plus the running journal entries, plus the current suspicions list with weights.

## Run

```
export ANTHROPIC_API_KEY=sk-ant-...
python run.py
```

## Cost notes

- Journal call per agent per phase (already in place): +$0.05/12-year game.
- Dream call per agent per year (new): +$0.05/12-year game.
- Total added beyond baseline ~$0.10/12-year game on Haiku.

## What you'll find on disk after a game

```
~/.diplomacy_llm/snapshots/<game-time>/
  000-1901-SPRING-MOVEMENT_after-movement.json    ← full snapshot (mind + biopsy + log + messages)
  biopsy-000-1901-SPRING-MOVEMENT_after-movement.json    ← biopsy alone (smaller, easier to review)
  001-1901-FALL-MOVEMENT_after-movement.json
  biopsy-001-1901-FALL-MOVEMENT_after-movement.json
  ...
```

The snapshot files have `biopsy` as a key inside, alongside `minds`, `messages`, etc. The standalone `biopsy-*.json` files are just the biopsy data, easier to grep through.

## Diagnosing biopsy issues

If the biopsy modal in the GUI shows nothing or behaves strangely:

1. Click the **⬇ Download JSON** button. If it downloads a file with real content, the data IS there — only the rendering is broken. Send me the JSON and I'll see what's there.
2. Check `~/.diplomacy_llm/snapshots/<game>/` for the `biopsy-*.json` files. They have the same data, persisted per phase.
3. Open browser DevTools → Network tab → click around in the biopsy modal. Look for any `/api/biopsy/*` calls that return non-200 status.

## Specific things to watch in the next game

- **Dreams lens** — open the inspector, switch to 💭 Dreams, then click each AI in turn. After year 1 ends (FALL ADJUSTMENT), each substrate agent should have a year-end consolidation entry. The raw_llm agents (England, Germany if you're using the same setup) won't have any — they have no substrate, so no dreaming.
- **Graph lens** — same drill, switch to 🕸 Graph view. Edges between the inspected agent and the others. Edge color shows trust based on actual kept-vs-broken counts. After a few years you should see real differentiation: substrate agents probably get green edges to each other, raw_llm agents may get red or gold.
- **Per-row 🔬 buttons** — these are SUPPOSED to open the biopsy modal pre-selected on that power. If clicking them doesn't open the modal, that's a real bug — please send me the browser console output (Cmd+Opt+I → Console tab).

## What this drop does NOT fix

If the GUI biopsy modal is genuinely broken (the JS errors out before showing anything), I can't see that without the browser console output. The Download button is my workaround — even if the modal fails to render, you can still get the data.

If something else 500s, send me the terminal traceback.
