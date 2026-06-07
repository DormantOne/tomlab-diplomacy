"""
dump_prompts.py — run a short Diplomacy game with prompt recording, then
dump the latest prompt+response of each kind from each agent to disk.

Lets you read what the LLM actually saw, offline, without needing the
live GUI. The output directory ends up with files like

    AUSTRIA_FULL_negotiate.txt
    AUSTRIA_FULL_orders.txt
    AUSTRIA_FULL_grade_commitments.txt
    FRANCE_MUTED_negotiate.txt
    ...

Each file is ~3-5KB of plain text: the raw prompt followed by the raw
response. Open with `less` and read top to bottom.

Usage
-----

    # Cheap stub run — verify the wiring works without API calls
    python dump_prompts.py --llm stub --phases 4 --out prompts_stub/

    # Real run with Anthropic Haiku, mute_3
    export ANTHROPIC_API_KEY=...
    python dump_prompts.py --llm anthropic --phases 6 \\
        --muted AUSTRIA,GERMANY,RUSSIA --out prompts_mute3/

    # All-full Sonnet run for a higher-quality reference
    python dump_prompts.py --llm anthropic \\
        --anthropic-model claude-sonnet-4-6 \\
        --phases 6 --out prompts_sonnet/

Cost: 6 phases × 6 agents × 2 calls/phase ≈ 72 calls × ~$0.006 = ~$0.43
with Haiku. Sonnet ~3x that. Stub is free.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from diplomacy_engine import initial_state
from diplomacy_mute import make_mutable_agents
from diplomacy_prompt_recorder import (
    attach_recorder, infer_call_kind, is_recording,
)


# Default archetype assignment matches eval_ablation.py and run_v2's docs.
ARCHETYPES = {
    "AUSTRIA": "MARSHAL_VEIL",
    "ENGLAND": "PARSON_HAWTHORNE",
    "FRANCE":  "ARCHITECT_LIRA",
    "GERMANY": "BARON_KORVIN",
    "RUSSIA":  "CARDINAL_FOX",
    "TURKEY":  "PLAYER_DEFAULT",
}


def build_llm_call(kind: str, anthropic_model: str,
                   ollama_url: str, ollama_model: str):
    if kind == "stub":
        from run_v2 import StubLLM
        return StubLLM()
    if kind == "anthropic":
        from run_v2 import make_anthropic_caller
        return make_anthropic_caller(anthropic_model)
    if kind == "ollama":
        from run_v2 import make_ollama_caller
        return make_ollama_caller(ollama_url, ollama_model)
    raise ValueError(f"unknown LLM kind: {kind}")


def write_prompt_file(out_dir: Path, power: str, muted_marker: str,
                      kind: str, record) -> Path:
    """Write one (prompt, response) pair to a labelled file."""
    fname = f"{power}_{muted_marker}_{kind}.txt"
    fpath = out_dir / fname
    parts = [
        f"=== {power} ({muted_marker}) — call kind: {kind} ===",
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.timestamp))}",
        f"elapsed: {record.elapsed_seconds:.2f}s",
        f"prompt_length: {len(record.prompt):,} chars (~{len(record.prompt)//4:,} tokens)",
        f"response_length: {len(record.response):,} chars",
    ]
    if record.error:
        parts.append(f"ERROR: {record.error}")
    parts.append("")
    parts.append("--- PROMPT ---")
    parts.append(record.prompt)
    parts.append("")
    parts.append("--- RESPONSE ---")
    parts.append(record.response)
    parts.append("")
    fpath.write_text("\n".join(parts))
    return fpath


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a short recorded game and dump prompts for inspection.",
    )
    ap.add_argument("--phases", type=int, default=6,
                    help="phases to run (default 6 — far enough for some "
                         "promises and predictions to resolve)")
    ap.add_argument("--out", default="prompts",
                    help="output directory (default 'prompts/')")
    ap.add_argument("--muted", default="",
                    help="comma-separated powers to mute "
                         "(e.g. 'AUSTRIA,GERMANY,RUSSIA')")
    ap.add_argument("--llm", choices=["stub", "anthropic", "ollama"],
                    default="stub")
    ap.add_argument("--anthropic-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    ap.add_argument("--ollama-model", default="gpt-oss:20b")
    ap.add_argument("--capacity", type=int, default=50,
                    help="max history per agent (default 50)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    muted = {s.strip().upper() for s in args.muted.split(",") if s.strip()}
    invalid = muted - set(ARCHETYPES.keys())
    if invalid:
        print(f"error: unknown power(s) in --muted: {sorted(invalid)}",
              file=sys.stderr)
        return 2

    try:
        llm_call = build_llm_call(
            args.llm, args.anthropic_model,
            args.ollama_url, args.ollama_model,
        )
    except RuntimeError as e:
        print(f"error setting up LLM caller: {e}", file=sys.stderr)
        return 2

    print(f"[plan] {args.phases} phases, llm={args.llm}, "
          f"muted={sorted(muted) or '(none)'}")

    # Build agents and attach recorder.
    agents = make_mutable_agents(
        llm_call=llm_call,
        archetype_assignment=ARCHETYPES,
        muted_powers=muted,
    )
    for agent in agents.values():
        attach_recorder(agent, capacity=args.capacity)
    assert all(is_recording(a) for a in agents.values())

    # Drive the game.
    from run_v2 import run_one_phase
    state = initial_state()
    message_log: list = []
    t0 = time.time()
    completed = 0
    for i in range(args.phases):
        try:
            state = run_one_phase(state, agents, message_log,
                                  verbose=args.verbose)
            completed += 1
            print(f"  phase {i + 1}/{args.phases} ok")
        except Exception as e:
            print(f"  phase {i + 1} aborted: {type(e).__name__}: {e}")
            break
    dt = time.time() - t0
    print(f"\n[run] {completed}/{args.phases} phases in {dt:.1f}s")

    # Dump: for each agent, take the latest record of each call kind.
    print(f"\n[dump] writing to {out_dir}/")
    written = 0
    total_recorded = 0
    for power, agent in agents.items():
        muted_marker = "MUTED" if power in muted else "FULL"
        rec = agent.llm_call
        total_recorded += rec.total_calls
        # Latest record per kind. deque is oldest-first, so iterating in
        # order and overwriting yields newest-of-kind at the end.
        latest_by_kind = {}
        for r in rec.history:
            kind = infer_call_kind(r.prompt)
            latest_by_kind[kind] = r
        for kind, r in latest_by_kind.items():
            fpath = write_prompt_file(out_dir, power, muted_marker, kind, r)
            print(f"  {fpath.name}  ({len(r.prompt):,} chars)")
            written += 1

    print(f"\n[done] {written} files written; "
          f"{total_recorded} total LLM calls recorded across all agents")
    print(f"\nRead with: less {out_dir}/<file>.txt")
    print(f"Compare archetypes: diff {out_dir}/AUSTRIA_*_negotiate.txt "
          f"{out_dir}/FRANCE_*_negotiate.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
