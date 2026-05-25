"""
eval_ablation.py — Phase 1.5 ablation harness.

Runs N games per condition (full / all_muted / mute_K) and reports per-power
metrics, splitting MUTED from UNMUTED so the experiment table reads cleanly.

The headline question this answers:
    Does the substrate's KG advisory measurably shift play?

If the answer is no — full and all_muted produce indistinguishable SC counts,
kept-rates, and confirm-rates — then phases 2+ don't pay off and we should
rethink before further investment. This is the GATE.

Usage
-----

    # Cheap stub-LLM smoke test (no API key needed). 2 games per condition.
    python eval_ablation.py --games 2 --phases 4 --conditions full,all_muted

    # Real experiment with Anthropic Haiku. ~12 phases is one or two
    # game-years; ~6 games per condition is a reasonable starting bar.
    export ANTHROPIC_API_KEY=...
    python eval_ablation.py --llm anthropic --games 6 --phases 12 \\
        --conditions full,all_muted,mute_2 \\
        --out results.json

Conditions
----------
    full        — all 6 powers run with full KG advisory (control: substrate ON)
    all_muted   — all 6 powers muted (control: substrate OFF, identity-only)
    mute_K      — K random powers muted, rest full (the gradient)

Within each condition, every power produces an independent metrics row.
The `kg_advisory_mute` flag in each row tells you whether that row was a
MUTED player or an UNMUTED player. The aggregate report splits them so you
can read both the within-condition and across-condition deltas.

Metrics per agent
-----------------
    sc_count                 final supply-center count
    eliminated               bool
    kept_rate                self-commitments kept / (kept + broken)
    confirm_rate             predictions confirmed / (confirmed + refuted)
    beliefs / preds / etc.   raw KG-state counts at end of game

Aggregates
----------
    For each condition and each {muted, unmuted} bucket:
        n_rows, mean_sc, mean_kept_rate, mean_confirm_rate, mean_beliefs.

Output
------
    Pretty-printed table + (optional) per-game JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from diplomacy_engine import POWERS, initial_state
from diplomacy_mute import (
    make_mutable_agents, set_mute, per_agent_metrics,
)


# ============================================================================
# LLM caller selection
# ============================================================================


def build_llm_call(kind: str, anthropic_model: str, ollama_url: str,
                   ollama_model: str, debug_log: Optional[str]):
    """Return a callable(prompt) -> str for the chosen backend.

    Reuses run_v2's wrappers when possible. Stub is the default for cheap
    runs; anthropic and ollama require their respective configuration.
    """
    if kind == "stub":
        from run_v2 import StubLLM
        return StubLLM()
    if kind == "anthropic":
        from run_v2 import make_anthropic_caller
        return make_anthropic_caller(anthropic_model, debug_log_path=debug_log)
    if kind == "ollama":
        from run_v2 import make_ollama_caller
        return make_ollama_caller(ollama_url, ollama_model, debug_log_path=debug_log)
    raise ValueError(f"unknown LLM kind: {kind}")


# ============================================================================
# Condition → muted-set
# ============================================================================


def muted_set_for_condition(
    condition: str, powers: list[str], rng: random.Random,
) -> set[str]:
    """Translate a condition name to the set of powers that should be muted."""
    if condition == "full":
        return set()
    if condition == "all_muted":
        return set(powers)
    if condition.startswith("mute_"):
        try:
            n = int(condition.split("_", 1)[1])
        except (IndexError, ValueError):
            raise ValueError(
                f"condition '{condition}' must be 'mute_<int>'"
            )
        if n < 0 or n > len(powers):
            raise ValueError(
                f"mute count {n} out of range [0, {len(powers)}]"
            )
        return set(rng.sample(powers, k=n))
    raise ValueError(f"unknown condition: {condition!r}")


# ============================================================================
# Run one game
# ============================================================================


def run_one_game(
    *,
    llm_call,
    archetype_assignment: dict[str, str],
    muted_powers: set[str],
    n_phases: int,
    verbose: bool = False,
) -> dict:
    """Build agents, run n_phases via run_v2.run_one_phase, return final metrics."""
    from run_v2 import run_one_phase

    agents = make_mutable_agents(
        llm_call=llm_call,
        archetype_assignment=archetype_assignment,
        muted_powers=muted_powers,
    )
    state = initial_state()
    message_log: list = []
    phases_completed = 0
    aborted_with: Optional[str] = None
    started_at = time.time()

    for phase_idx in range(n_phases):
        try:
            state = run_one_phase(
                state, agents, message_log, verbose=verbose,
            )
            phases_completed += 1
        except Exception as e:
            aborted_with = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"    [phase {phase_idx + 1} aborted] {aborted_with}")
            break

    metrics = per_agent_metrics(agents, state)
    return {
        "muted_powers": sorted(muted_powers),
        "n_phases_requested": n_phases,
        "n_phases_completed": phases_completed,
        "aborted_with": aborted_with,
        "wall_seconds": round(time.time() - started_at, 2),
        "final_year": getattr(state, "year", None),
        "final_season": getattr(state, "season", None),
        "metrics": metrics,
    }


# ============================================================================
# Aggregation + reporting
# ============================================================================


def _safe_mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return round(sum(xs) / len(xs), 4)


def aggregate_bucket(rows: list[dict]) -> dict:
    """Roll up a list of per-agent metric rows into bucket-level numbers."""
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_sc": _safe_mean([r["sc_count"] for r in rows]),
        "n_eliminated": sum(1 for r in rows if r["eliminated"]),
        "mean_kept_rate": _safe_mean([r["kept_rate"] for r in rows]),
        "rows_with_kept_data": sum(
            1 for r in rows if r["kept_rate"] is not None
        ),
        "mean_confirm_rate": _safe_mean([r["confirm_rate"] for r in rows]),
        "rows_with_confirm_data": sum(
            1 for r in rows if r["confirm_rate"] is not None
        ),
        "mean_beliefs": _safe_mean([r["beliefs"] for r in rows]),
        "mean_predictions": _safe_mean([r["predictions"] for r in rows]),
        "mean_self_commitments": _safe_mean(
            [r["self_commitments"] for r in rows]
        ),
        "mean_strategic_intents": _safe_mean(
            [r["strategic_intents"] for r in rows]
        ),
    }


def aggregate_condition(games: list[dict]) -> dict:
    """Aggregate a condition's games, splitting muted vs unmuted rows."""
    muted_rows = []
    unmuted_rows = []
    for g in games:
        for power, m in g["metrics"].items():
            row = dict(m)
            row["power"] = power
            (muted_rows if m["kg_advisory_mute"] else unmuted_rows).append(row)
    aborted = sum(1 for g in games if g["aborted_with"] is not None)
    completed_phases = sum(g["n_phases_completed"] for g in games)
    requested_phases = sum(g["n_phases_requested"] for g in games)
    return {
        "n_games": len(games),
        "n_aborted": aborted,
        "phases_completed_total": completed_phases,
        "phases_requested_total": requested_phases,
        "phase_completion_rate": (
            round(completed_phases / requested_phases, 4)
            if requested_phases else None
        ),
        "muted": aggregate_bucket(muted_rows),
        "unmuted": aggregate_bucket(unmuted_rows),
    }


def _fmt(v):
    if v is None:
        return "  —  "
    if isinstance(v, float):
        return f"{v:>5.2f}"
    return f"{v:>5}"


def print_report(aggregated: dict[str, dict]) -> None:
    """Pretty-print the aggregated results."""
    print()
    print("=" * 78)
    print("ABLATION RESULTS")
    print("=" * 78)
    headers = ["condition", "bucket", "n", "mean_sc", "elim", "kept_rt",
               "conf_rt", "beliefs", "preds", "ints", "promises"]
    print(
        f"  {'condition':<14}{'bucket':<10}{'n':>5}{'mean_sc':>10}"
        f"{'elim':>6}{'kept_rt':>10}{'conf_rt':>10}{'beliefs':>10}"
        f"{'preds':>8}{'ints':>7}{'promises':>10}"
    )
    print("  " + "-" * 100)
    for cond_name, cond_data in aggregated.items():
        for bucket_name in ("muted", "unmuted"):
            b = cond_data[bucket_name]
            if b["n"] == 0:
                continue
            print(
                f"  {cond_name:<14}{bucket_name:<10}"
                f"{b['n']:>5}"
                f"{_fmt(b['mean_sc']):>10}"
                f"{b['n_eliminated']:>6}"
                f"{_fmt(b['mean_kept_rate']):>10}"
                f"{_fmt(b['mean_confirm_rate']):>10}"
                f"{_fmt(b['mean_beliefs']):>10}"
                f"{_fmt(b['mean_predictions']):>8}"
                f"{_fmt(b['mean_strategic_intents']):>7}"
                f"{_fmt(b['mean_self_commitments']):>10}"
            )
        ag = cond_data["n_aborted"]
        if ag:
            print(
                f"    (note: {ag}/{cond_data['n_games']} games aborted "
                f"before completing all phases)"
            )
    print()
    print("Reading the table:")
    print("  - kept_rt: self-commitments kept / (kept + broken). Null = none resolved.")
    print("  - conf_rt: predictions confirmed / (confirmed + refuted). Null = none resolved.")
    print("  - The MUTED-vs-UNMUTED gap *within* a mixed condition (e.g. mute_2)")
    print("    is the within-game contrast — same board, different KG access.")
    print("  - The 'full' UNMUTED row vs 'all_muted' MUTED row is the across-game")
    print("    contrast — same condition, different KG availability for everyone.")


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 1.5 ablation harness — measures KG advisory effect.",
    )
    ap.add_argument("--games", type=int, default=2,
                    help="games per condition (default 2)")
    ap.add_argument("--phases", type=int, default=4,
                    help="phases per game (default 4 — one full year)")
    ap.add_argument("--conditions", default="full,all_muted",
                    help="comma-separated. Options: full, all_muted, "
                         "mute_<K>. Default: 'full,all_muted'.")
    ap.add_argument("--llm", choices=["stub", "anthropic", "ollama"],
                    default="stub",
                    help="LLM backend (default 'stub')")
    ap.add_argument("--anthropic-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    ap.add_argument("--ollama-model", default="gpt-oss:20b")
    ap.add_argument("--debug-llm", default=None,
                    help="path to append every prompt/response pair (debug)")
    ap.add_argument("--seed", type=int, default=0xD1,
                    help="rng seed for mute-set selection (default 0xD1)")
    ap.add_argument("--out", default=None,
                    help="optional JSON file to write per-game raw results")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Default archetype assignment matches run_v2.py
    archetype_assignment = {
        "AUSTRIA": "MARSHAL_VEIL",
        "ENGLAND": "PARSON_HAWTHORNE",
        "FRANCE":  "ARCHITECT_LIRA",
        "GERMANY": "BARON_KORVIN",
        "RUSSIA":  "CARDINAL_FOX",
        "TURKEY":  "PLAYER_DEFAULT",
    }
    powers = list(archetype_assignment.keys())

    try:
        llm_call = build_llm_call(
            args.llm, args.anthropic_model,
            args.ollama_url, args.ollama_model, args.debug_llm,
        )
    except RuntimeError as e:
        print(f"error setting up LLM caller: {e}", file=sys.stderr)
        return 2

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"[plan] {len(conditions)} condition(s), "
          f"{args.games} game(s) each, {args.phases} phase(s) per game.")
    print(f"       LLM = {args.llm}; "
          f"total games = {len(conditions) * args.games}")

    raw_results: dict[str, list[dict]] = {}
    aggregated: dict[str, dict] = {}

    for cond in conditions:
        try:
            muted = muted_set_for_condition(cond, powers, rng)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"\n[condition] {cond}: muted = {sorted(muted) or '(none)'}")
        games_for_condition: list[dict] = []
        for gi in range(args.games):
            print(f"  game {gi + 1}/{args.games}…", end=" ", flush=True)
            t0 = time.time()
            game_result = run_one_game(
                llm_call=llm_call,
                archetype_assignment=archetype_assignment,
                muted_powers=muted,
                n_phases=args.phases,
                verbose=args.verbose,
            )
            dt = time.time() - t0
            ph = game_result["n_phases_completed"]
            ab = game_result["aborted_with"]
            print(
                f"completed {ph}/{args.phases} phases "
                f"in {dt:.1f}s"
                f"{' (ABORTED: ' + str(ab) + ')' if ab else ''}"
            )
            games_for_condition.append(game_result)
        raw_results[cond] = games_for_condition
        aggregated[cond] = aggregate_condition(games_for_condition)

    print_report(aggregated)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps({
            "config": {
                "games": args.games, "phases": args.phases,
                "conditions": conditions, "llm": args.llm,
                "anthropic_model": args.anthropic_model,
                "seed": args.seed,
            },
            "raw": raw_results,
            "aggregated": aggregated,
        }, indent=2, default=str))
        print(f"\nwrote: {out_path}")

    return 0


# ============================================================================
# Self-test (cheap, with stub LLM)
# ============================================================================
# Verifies the harness wires up end-to-end against the stub. Runs 1 game per
# condition for 2 phases — about 1 second wall time.

def _self_test():
    print("=" * 72)
    print("EVAL ABLATION HARNESS SELF-TEST (stub LLM, 1 game × 2 phases)")
    print("=" * 72)
    sys.argv = [
        "eval_ablation.py",
        "--games", "1", "--phases", "2",
        "--conditions", "full,all_muted",
        "--llm", "stub",
    ]
    rc = main()
    if rc == 0:
        print()
        print("Self-test passed.")
    else:
        print(f"Self-test FAILED (rc={rc})")
    return rc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
