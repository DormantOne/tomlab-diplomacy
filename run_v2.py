#!/usr/bin/env python3
"""
run_v2.py — minimal runnable driver for the new architecture.

Stands up 6 agents (one per power, no Italy in this 6-power variant — but
we run all 7 here to exercise the full engine), runs N phases against the
existing diplomacy_engine, and prints per-phase telemetry.

Two LLM modes:

  --stub   (default) — deterministic stub LLM. Always returns valid JSON,
           always emits commitspeak when negotiating, always emits a near-term
           prediction for orders. Use this to verify the wiring end-to-end
           without needing Ollama loaded.

  --ollama OLLAMA_URL  — real Ollama. Requires ollama serve running with
           the configured model pulled. Will be slower; expect 30-90s per
           agent per phase per the original code's bench numbers.

  --anthropic MODEL  — real Anthropic API. (Sketched; needs ANTHROPIC_API_KEY
           in env. Slower per call but more reliable JSON output.)

Usage:
  python run_v2.py                    # 4 phases with stub LLM
  python run_v2.py --phases 8         # 8 phases with stub
  python run_v2.py --ollama --phases 4 --model gpt-oss:20b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from typing import Optional, Callable

# Engine
from diplomacy_engine import (
    POWERS, PROVINCES, initial_state,
    adjudicate_movement, adjudicate_retreats, adjudicate_adjustments,
    update_supply_centers, advance_phase, parse_order,
    units_by_power, supply_centers_owned,
)

# New mind layer
from diplomacy_kg_schema import MessageEvent, new_id
from diplomacy_agent_v2 import DiplomacyAgentV2, schema_phase_key
from diplomacy_engine_glue import (
    capture_move_events, capture_adjustment_events, capture_phase_state,
    distribute_messages,
)
from diplomacy_biopsy import (
    make_run_dir, write_agent_snapshot, append_board_state,
    append_messages, write_run_summary,
)

# ============================================================================
# Stub LLM — for testing without a real model
# ============================================================================

class StubLLM:
    """A deterministic LLM that emits valid JSON for any prompt.

    Each agent calling the same StubLLM with the same prompt gets the same
    response (good for reproducibility). We vary by hashing the prompt's
    first 200 chars so different agents/phases get different content.
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        # Detect call kind from the prompt
        is_orders = "ORDER FORMS" in prompt or "near_term" in prompt.lower()
        is_negotiate = ("Compose 0-3" in prompt or
                        "messages" in prompt.lower() and "outgoing" in prompt.lower())
        is_revision = ("Propose a NARROWER" in prompt or
                       "revision" in prompt.lower())

        # Extract the speaker if we can
        speaker = self._extract_speaker(prompt)
        phase = self._extract_phase(prompt) or "1901-SPRING-MOVES"

        if is_orders:
            return self._fake_orders(prompt, speaker, phase)
        if is_revision:
            return self._fake_revision(prompt)
        if is_negotiate:
            return self._fake_negotiate(prompt, speaker, phase)
        return "{}"

    def _extract_speaker(self, prompt: str) -> str:
        # The agent's character_brief usually starts the fovea render
        for power in POWERS:
            if power in prompt[:500]:
                return power
        return "FRANCE"

    def _extract_phase(self, prompt: str) -> Optional[str]:
        # Look for a phase like "1902-SPRING-MOVES" or "S1902"
        m = re.search(r"\b(\d{4})-(SPRING|FALL|WINTER)-(MOVES|RETREATS|ADJUSTMENTS)\b",
                      prompt)
        if m:
            return m.group(0)
        m = re.search(r"\b([SFW])(\d{4})\b", prompt)
        if m:
            season = {"S": "SPRING", "F": "FALL", "W": "WINTER"}[m.group(1)]
            phase_kind = "ADJUSTMENTS" if season == "WINTER" else "MOVES"
            return f"{m.group(2)}-{season}-{phase_kind}"
        return None

    def _fake_orders(self, prompt: str, speaker: str, phase: str) -> str:
        # Parse "YOUR UNITS — order each one:" block to find this agent's units
        m = re.search(r"YOUR UNITS — order each one:\s*\n((?:.*\n)+?)\n", prompt)
        unit_block = m.group(1) if m else ""
        unit_orders = []
        for line in unit_block.split("\n"):
            # Lines look like "  A PAR: HOLD or MOVE to one of: BUR, GAS, PIC"
            ln = line.strip()
            if not ln:
                continue
            mu = re.match(r"([AF])\s+([A-Z]{3}):\s*HOLD or MOVE to one of:\s*(.+)", ln)
            if mu:
                kind = mu.group(1); origin = mu.group(2)
                options = [x.strip() for x in mu.group(3).split(",")]
                # Stub heuristic: prefer first option
                target = options[0] if options else None
                if target:
                    unit_orders.append(f"{kind} {origin} - {target}")
                else:
                    unit_orders.append(f"{kind} {origin} H")
            else:
                mh = re.match(r"([AF])\s+([A-Z]{3}):\s*must HOLD", ln)
                if mh:
                    unit_orders.append(f"{mh.group(1)} {mh.group(2)} H")

        # Pick a target power that isn't us for the prediction
        other = "GERMANY" if speaker != "GERMANY" else "FRANCE"
        # Predict non_action on a province that's not our own
        return json.dumps({
            "orders": unit_orders or ["A PAR H"],
            "plan": {
                "head": f"{speaker} stub plan for {phase}",
                "body": "Stub orders proceed conservatively.",
                "parent_intent_id": None,
            },
            "predictions": [{
                "about": other,
                "type": "non_action",
                "target": "BEL",
                "window": "near_term",
                "rationale": "stub heuristic"
            }]
        })

    def _fake_negotiate(self, prompt: str, speaker: str, phase: str) -> str:
        # Pick first addressee and emit one message with a clean commitspeak
        m = re.search(r"Recipients chosen from:\s*([A-Z, ]+)", prompt)
        addressees = []
        if m:
            addressees = [x.strip() for x in m.group(1).split(",")]
        target = addressees[0] if addressees else "GERMANY"
        return json.dumps({
            "messages": [{
                "to": [target],
                "public": False,
                "text": (f"Routine coordination from {speaker[:3]}.\n\n"
                         f"[[commit\n  not_move_to: BEL by {phase}\n]]"),
            }]
        })

    def _fake_revision(self, prompt: str) -> str:
        return json.dumps({
            "head": "Narrower successor (stub).",
            "body": "Stub revision — narrowing the parent claim by one step.",
            "type": "tactical_pattern",
        })


# ============================================================================
# Real LLM wrappers
# ============================================================================

def make_ollama_caller(url: str, model: str, timeout: float = 300.0,
                       debug_log_path: Optional[str] = None) -> Callable:
    """Returns a callable(prompt) -> str using Ollama's /api/generate.

    If `debug_log_path` is set, every prompt+response pair is appended to
    that file for debugging. Useful for the first few real-LLM runs to see
    what the model is actually emitting.
    """
    def _call(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.6, "top_p": 0.9,
                "num_predict": 800, "num_ctx": 4096,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response = data.get("response", "")
                if debug_log_path:
                    _append_debug_log(debug_log_path, prompt, response)
                return response
        except Exception as e:
            print(f"  [ollama error] {e}", file=sys.stderr)
            if debug_log_path:
                _append_debug_log(debug_log_path, prompt, f"[ERROR: {e}]")
            return "{}"
    return _call


def _append_debug_log(path: str, prompt: str, response: str) -> None:
    """Append a prompt/response pair to the debug log."""
    try:
        with open(path, "a") as f:
            f.write("=" * 72 + "\n")
            f.write(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("--- PROMPT ---\n")
            f.write(prompt[:3000])
            if len(prompt) > 3000:
                f.write(f"\n[... {len(prompt) - 3000} more chars ...]\n")
            f.write("\n--- RESPONSE ---\n")
            f.write(response[:3000])
            if len(response) > 3000:
                f.write(f"\n[... {len(response) - 3000} more chars ...]\n")
            f.write("\n\n")
    except Exception as e:
        print(f"  [debug log error] {e}", file=sys.stderr)


def make_anthropic_caller(model: str,
                          debug_log_path: Optional[str] = None) -> Callable:
    """Returns a callable using the Anthropic Messages API.

    Requires ANTHROPIC_API_KEY in environment. Optional debug_log_path
    will append every prompt+response pair to that file.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    def _call(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "max_tokens": 1500,
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
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content_blocks = data.get("content", [])
                response = "\n".join(
                    b.get("text", "") for b in content_blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                if debug_log_path:
                    _append_debug_log(debug_log_path, prompt, response)
                return response
        except Exception as e:
            print(f"  [anthropic error] {e}", file=sys.stderr)
            if debug_log_path:
                _append_debug_log(debug_log_path, prompt, f"[ERROR: {e}]")
            return "{}"
    return _call


# ============================================================================
# Brief generators (one-shot, at agent construction)
# ============================================================================

DEFAULT_BRIEFS = {
    "MARSHAL_VEIL": ("I am Marshal Veil. I plan in arcs. I keep my word "
                     "when watched and remember when others do not."),
    "CARDINAL_FOX": ("I am Cardinal Fox. I trade in stories and what they "
                     "imply. I prefer a beautiful turn to a safe one."),
    "PARSON_HAWTHORNE": ("I am Parson Hawthorne. My word is given carefully "
                        "and kept absolutely."),
    "BARON_KORVIN": ("I am Baron Korvin. I trust no one before they have "
                    "earned it twice."),
    "ARCHITECT_LIRA": ("I am Architect Lira. I look at the whole table "
                      "and design the equilibrium I prefer."),
    "PLAYER_DEFAULT": "An LLM-driven Diplomacy player.",
}


def make_agents(
    *, llm_call: Callable, archetype_assignment: dict[str, str],
) -> dict[str, DiplomacyAgentV2]:
    """Build one agent per power with assigned archetypes."""
    agents = {}
    for power, archetype in archetype_assignment.items():
        agents[power] = DiplomacyAgentV2(
            power=power, archetype=archetype,
            character_brief_text=DEFAULT_BRIEFS.get(archetype,
                                                   DEFAULT_BRIEFS["PLAYER_DEFAULT"]),
            llm_call=llm_call,
            valid_powers=set(POWERS),
            valid_provinces=set(PROVINCES.keys()),
        )
    return agents


# ============================================================================
# Phase orchestrator
# ============================================================================

def run_one_phase(state, agents: dict[str, DiplomacyAgentV2],
                  message_log: list, *, verbose: bool = True,
                  biopsy: Optional[dict] = None):
    """Run one full phase: negotiate (1 round) → orders → adjudicate → absorb.

    Returns the new GameState and any per-agent telemetry.

    `biopsy`, if provided, must be a dict with:
      - 'opts': dict of toggle bools
      - 'run_dir': directory path
    Used to write per-phase snapshots / message log / board log.
    """
    pre_state = state
    schema_phase = schema_phase_key(state.year, state.season, state.phase)

    if state.phase == "MOVEMENT":
        # 1. One round of negotiation
        if verbose:
            print(f"  -- negotiation round --")
        all_messages = []
        for power, agent in agents.items():
            if power in state.eliminated:
                continue
            messages = agent.negotiate(state, message_log)
            all_messages.extend(messages)
            if verbose:
                for m in messages:
                    target = "ALL" if m.public else ",".join(m.recipients)
                    has_cs = "[cs]" if m.commitspeak_tail else ""
                    print(f"    {power[:3]} -> {target}: "
                          f"{m.body.split(chr(10))[0][:60]} {has_cs}")

        # Distribute and record
        distribute_counts = distribute_messages(all_messages, agents)
        message_log.extend(all_messages)
        if verbose:
            tot_cs = sum(1 for m in all_messages if m.commitspeak_tail)
            print(f"  {len(all_messages)} messages sent ({tot_cs} with commitspeak); "
                  f"new_commitments_per_recipient={distribute_counts}")

        # 2. Orders
        if verbose:
            print(f"  -- orders --")
        all_orders = []
        for power, agent in agents.items():
            if power in state.eliminated:
                continue
            order_strings, out = agent.decide_orders(state, message_log)
            for line in order_strings:
                o = parse_order(power, line)
                if o is not None:
                    all_orders.append(o)
            if verbose:
                preds = len(out.predictions)
                synth = " (synth)" if any("synth" in n for n in out.parse_notes) else ""
                print(f"    {power[:3]}: {len(order_strings)} orders, "
                      f"{preds} predictions{synth}")

        # 3. Adjudicate
        new_state, adjudication_log = adjudicate_movement(state, all_orders)
        if verbose:
            print(f"  -- adjudication: {len(adjudication_log)} log lines --")

        # 4. Capture events; absorb
        move_events = capture_move_events(
            orders=all_orders, pre_state=pre_state,
            post_state=new_state, adjudication_log=adjudication_log,
        )
        adj_events = capture_adjustment_events(
            pre_state=pre_state, post_state=new_state,
        )
        phase_state = capture_phase_state(new_state)

        # Determine next phase
        # advance_phase returns the next phase; we project the schema key
        peek = advance_phase(_clone_state(new_state))
        next_schema = schema_phase_key(peek.year, peek.season, peek.phase)

        for power, agent in agents.items():
            if power in state.eliminated:
                continue
            log = agent.absorb_phase_resolution(
                resolved_phase=schema_phase,
                move_events=move_events,
                adjustment_events=adj_events,
                phase_state=phase_state,
                next_phase=next_schema,
            )
            if verbose:
                bits = []
                if log.commitments_graded: bits.append(f"cmt_graded={log.commitments_graded}")
                if log.predictions_graded: bits.append(f"pred_graded={log.predictions_graded}")
                if log.beliefs_promoted: bits.append(f"belief+={log.beliefs_promoted}")
                if log.beliefs_retired: bits.append(f"belief-={log.beliefs_retired}")
                if log.intents_promoted: bits.append(f"intent+={log.intents_promoted}")
                if log.intents_retired: bits.append(f"intent-={log.intents_retired}")
                if log.revision_proposals_made: bits.append(f"rev={log.revision_proposals_made}")
                if log.near_term_synthesized: bits.append("synth")
                if bits:
                    print(f"    {power[:3]} lifecycle: {', '.join(bits)}")

        # 5. Biopsy writes (after lifecycle has run)
        if biopsy:
            opts = biopsy["opts"]
            run_dir = biopsy["run_dir"]
            phase_label = f"{state.year}_{state.season}_{state.phase}"
            if opts.get("messages_log"):
                append_messages(all_messages, run_dir, phase_label=phase_label)
            if opts.get("board_log"):
                append_board_state(new_state, run_dir, phase_label=phase_label)
            if opts.get("kg_snapshots"):
                for power, agent in agents.items():
                    if power in state.eliminated:
                        continue
                    write_agent_snapshot(agent.mind, schema_phase, run_dir)

        # 6. Advance the engine
        state = advance_phase(new_state)
        return state

    elif state.phase == "RETREAT":
        # Retreats — heuristic, no LLM call.
        all_orders = []
        for power, agent in agents.items():
            for line in agent.decide_retreats(state):
                o = parse_order(power, line)
                if o is not None:
                    all_orders.append(o)
        new_state, _retreat_log = adjudicate_retreats(state, all_orders)
        adj_events = capture_adjustment_events(
            pre_state=pre_state, post_state=new_state,
        )
        phase_state = capture_phase_state(new_state)
        peek = advance_phase(_clone_state(new_state))
        next_schema = schema_phase_key(peek.year, peek.season, peek.phase)
        for power, agent in agents.items():
            if power in state.eliminated:
                continue
            agent.absorb_phase_resolution(
                resolved_phase=schema_phase,
                move_events=[],
                adjustment_events=adj_events,
                phase_state=phase_state,
                next_phase=next_schema,
            )
        if verbose:
            print(f"  retreats resolved (heuristic, no LLM calls)")
        if biopsy and biopsy["opts"].get("board_log"):
            phase_label = f"{state.year}_{state.season}_{state.phase}"
            append_board_state(new_state, biopsy["run_dir"],
                               phase_label=phase_label)
        return advance_phase(new_state)

    elif state.phase == "ADJUSTMENT":
        # Builds/disbands at end of fall — heuristic.
        # First update SC ownership from the fall position (mutates state)
        update_supply_centers(state)
        new_state = state
        # Decide builds per power
        all_orders = []
        for power, agent in agents.items():
            if power in new_state.eliminated:
                continue
            for line in agent.decide_builds(new_state):
                o = parse_order(power, line)
                if o is not None:
                    all_orders.append(o)
        new_state, _adj_log = adjudicate_adjustments(new_state, all_orders)
        adj_events = capture_adjustment_events(
            pre_state=pre_state, post_state=new_state,
        )
        phase_state = capture_phase_state(new_state)
        peek = advance_phase(_clone_state(new_state))
        next_schema = schema_phase_key(peek.year, peek.season, peek.phase)
        for power, agent in agents.items():
            if power in state.eliminated:
                continue
            agent.absorb_phase_resolution(
                resolved_phase=schema_phase,
                move_events=[],
                adjustment_events=adj_events,
                phase_state=phase_state,
                next_phase=next_schema,
            )
        if verbose:
            print(f"  adjustments resolved: {len(adj_events)} build/disband events")
        if biopsy:
            opts = biopsy["opts"]
            run_dir = biopsy["run_dir"]
            phase_label = f"{state.year}_{state.season}_{state.phase}"
            if opts.get("board_log"):
                append_board_state(new_state, run_dir, phase_label=phase_label)
            if opts.get("kg_snapshots"):
                # End-of-year is a natural KG snapshot point
                for power, agent in agents.items():
                    if power in state.eliminated:
                        continue
                    write_agent_snapshot(agent.mind, schema_phase, run_dir)
        return advance_phase(new_state)

    return state


def _clone_state(state):
    """Shallow clone — just enough to peek at advance_phase without mutating."""
    import copy
    return copy.copy(state)


# ============================================================================
# Main
# ============================================================================

def pick_llm_interactive() -> tuple[str, dict]:
    """Interactive backend picker. Returns (kind, options).

    `kind` ∈ {"stub", "ollama", "anthropic"}.
    `options` is a dict of any per-backend overrides the user chose
    (e.g. {"anthropic_model": "claude-haiku-4-5-20251001"}).
    """
    print()
    print("=" * 64)
    print("  Pick an LLM backend:")
    print("=" * 64)
    print("    [1] Anthropic Haiku 4.5      (fast, cheap, JSON-clean)")
    print("    [2] Anthropic Opus 4.7       (slower, smartest, $$$)")
    print("    [3] Ollama gpt-oss:20b       (local, may time out)")
    print("    [4] Ollama llama3.2:3b       (local, faster + smaller)")
    print("    [5] Stub LLM                  (deterministic, no network)")
    print("    [q] Quit")
    print()
    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            sys.exit(0)
        if choice in ("q", "quit", "exit"):
            print("  Cancelled.")
            sys.exit(0)
        if choice in ("1", ""):
            return "anthropic", {"model": "claude-haiku-4-5-20251001"}
        if choice == "2":
            return "anthropic", {"model": "claude-opus-4-7"}
        if choice == "3":
            return "ollama", {"model": "gpt-oss:20b"}
        if choice == "4":
            return "ollama", {"model": "llama3.2:3b"}
        if choice == "5":
            return "stub", {}
        print("  (pick 1-5, or q to quit)")


def pick_biopsy_options() -> dict:
    """Interactive biopsy options menu.

    Returns a dict with 5 booleans:
      debug_llm:        save every prompt/response pair
      kg_snapshots:     save per-agent KG snapshots after each phase
      messages_log:     save chronological message log
      board_log:        save phase-by-phase board state
      summary:          save final run summary

    Default: everything except the heavyweight ones is on. User can flip
    individual toggles or pick a preset.
    """
    print()
    print("=" * 64)
    print("  What to save? (will create a timestamped biopsy_/ folder)")
    print("=" * 64)

    # Default toggles — cheap stuff on, heavy stuff off
    opts = {
        "debug_llm": True,
        "kg_snapshots": True,
        "messages_log": True,
        "board_log": True,
        "summary": True,
    }

    while True:
        print()
        print(f"    [1] LLM debug log (every prompt/response)   "
              f"{'[on]' if opts['debug_llm'] else '[off]'}")
        print(f"    [2] KG snapshots (per agent per phase)       "
              f"{'[on]' if opts['kg_snapshots'] else '[off]'}")
        print(f"    [3] Message log                              "
              f"{'[on]' if opts['messages_log'] else '[off]'}")
        print(f"    [4] Board state log                          "
              f"{'[on]' if opts['board_log'] else '[off]'}")
        print(f"    [5] Run summary                              "
              f"{'[on]' if opts['summary'] else '[off]'}")
        print("    [a] All on")
        print("    [n] None — skip biopsy entirely")
        print("    [<enter>] continue with these settings")
        print()
        try:
            choice = input("  toggle> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            sys.exit(0)
        if choice == "":
            return opts
        if choice == "a":
            for k in opts:
                opts[k] = True
        elif choice == "n":
            for k in opts:
                opts[k] = False
            return opts
        elif choice == "1":
            opts["debug_llm"] = not opts["debug_llm"]
        elif choice == "2":
            opts["kg_snapshots"] = not opts["kg_snapshots"]
        elif choice == "3":
            opts["messages_log"] = not opts["messages_log"]
        elif choice == "4":
            opts["board_log"] = not opts["board_log"]
        elif choice == "5":
            opts["summary"] = not opts["summary"]
        else:
            print("  (1-5 to toggle, a=all, n=none, enter to continue)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=4,
                    help="how many phases to run (default 4)")
    ap.add_argument("--llm", choices=["stub", "ollama", "anthropic"],
                    default=None,
                    help="LLM backend (omit for interactive picker)")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    ap.add_argument("--ollama-model", default="gpt-oss:20b")
    ap.add_argument("--ollama-timeout", type=float, default=300.0,
                    help="seconds before Ollama call times out (default 300)")
    ap.add_argument("--anthropic-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--debug-llm", default=None,
                    help="path to append every LLM prompt/response pair (for debugging)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # If --llm wasn't specified, pop the interactive picker
    if args.llm is None:
        kind, opts = pick_llm_interactive()
        args.llm = kind
        if kind == "anthropic" and opts.get("model"):
            args.anthropic_model = opts["model"]
        elif kind == "ollama" and opts.get("model"):
            args.ollama_model = opts["model"]
        # And ask which biopsy artifacts to save
        biopsy_opts = pick_biopsy_options()
    else:
        # CLI invocation — default to no biopsy unless --debug-llm was passed
        biopsy_opts = {
            "debug_llm": bool(args.debug_llm),
            "kg_snapshots": False,
            "messages_log": False,
            "board_log": False,
            "summary": False,
        }

    # Set up biopsy run dir if any artifact requested
    biopsy = None
    if any(biopsy_opts.values()):
        run_dir = make_run_dir(".")
        print(f"\n  Biopsy folder: {run_dir}")
        biopsy = {"opts": biopsy_opts, "run_dir": run_dir}
        # If debug_llm toggle is on but no path was given via CLI, drop it
        # in the run dir so it stays grouped with the other artifacts
        if biopsy_opts["debug_llm"] and not args.debug_llm:
            args.debug_llm = os.path.join(run_dir, "llm_debug.log")

    # Build LLM caller
    if args.llm == "stub":
        llm_call = StubLLM()
        print(f"Using stub LLM (deterministic).")
    elif args.llm == "ollama":
        llm_call = make_ollama_caller(args.ollama_url, args.ollama_model,
                                      timeout=args.ollama_timeout,
                                      debug_log_path=args.debug_llm)
        print(f"Using Ollama at {args.ollama_url} with model {args.ollama_model}.")
        if args.debug_llm:
            print(f"  Debug log: {args.debug_llm}")
    elif args.llm == "anthropic":
        llm_call = make_anthropic_caller(args.anthropic_model,
                                         debug_log_path=args.debug_llm)
        print(f"Using Anthropic API with model {args.anthropic_model}.")
        if args.debug_llm:
            print(f"  Debug log: {args.debug_llm}")

    # Default archetype assignment — match the engine's POWERS list (this
    # codebase is a 6-power variant; Italy is not in play). Each power
    # gets a distinct archetype so personalities don't collide in
    # self-introductions.
    DEFAULT_ARCHETYPES = {
        "AUSTRIA": "MARSHAL_VEIL",
        "ENGLAND": "PARSON_HAWTHORNE",
        "FRANCE":  "ARCHITECT_LIRA",
        "GERMANY": "BARON_KORVIN",
        "RUSSIA":  "CARDINAL_FOX",
        "TURKEY":  "PLAYER_DEFAULT",
    }
    archetype_assignment = {
        power: DEFAULT_ARCHETYPES[power]
        for power in POWERS
        if power in DEFAULT_ARCHETYPES
    }
    agents = make_agents(llm_call=llm_call,
                         archetype_assignment=archetype_assignment)

    state = initial_state()
    message_log: list = []

    for phase_idx in range(args.phases):
        if not args.quiet:
            print()
            print("=" * 72)
            print(f"PHASE {phase_idx + 1}: {state.year} {state.season} {state.phase}")
            print("=" * 72)
            for power in POWERS:
                if power in state.eliminated:
                    continue
                scs = supply_centers_owned(state, power)
                print(f"  {power[:3]}: {len(scs)} SC")
        state = run_one_phase(state, agents, message_log,
                              verbose=not args.quiet, biopsy=biopsy)

    # Summary
    print()
    print("=" * 72)
    print("FINAL STATE")
    print("=" * 72)
    for power in POWERS:
        if power in state.eliminated:
            print(f"  {power}: eliminated")
            continue
        scs = supply_centers_owned(state, power)
        agent = agents.get(power)
        mind = agent.mind if agent else None
        if mind:
            print(f"  {power}: {len(scs)} SC, "
                  f"beliefs={len(mind.beliefs)}, "
                  f"intents={len(mind.strategic_intents)}, "
                  f"open_commitments_in={sum(1 for c in mind.incoming_commitments.values() if c.status.value=='pending')}, "
                  f"messages={len(mind.message_events)}")
        else:
            print(f"  {power}: {len(scs)} SC")

    if args.llm == "stub":
        print()
        print(f"Stub LLM: {llm_call.call_count} total calls.")

    # Final biopsy: run summary
    if biopsy and biopsy["opts"].get("summary"):
        model_str = (args.anthropic_model if args.llm == "anthropic"
                     else args.ollama_model if args.llm == "ollama"
                     else "stub")
        summary_path = write_run_summary(
            agents, state, biopsy["run_dir"],
            llm_kind=args.llm, model=model_str, phases_run=args.phases,
        )
        print(f"\n  Summary written: {summary_path}")
    if biopsy:
        print(f"  All biopsy artifacts: {biopsy['run_dir']}")


if __name__ == "__main__":
    main()
