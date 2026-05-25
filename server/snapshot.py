"""
server/snapshot.py — auto-save per-phase snapshots of a game in progress.

Writes one JSON file per phase to:
    ~/.diplomacy_llm/snapshots/<game-start-isodate>/<phase-index>-<phase-key>.json

Each snapshot contains:
    - phase, year, season, season_phase
    - sc_count_by_power and unit_count_by_power
    - all 5 AI minds (full mind_to_inspection_dict)
    - the running message log (all messages so far, not just this phase)
    - the running event log (all log lines so far)
    - a concise `summary` block: paragraph-style narration of what's
      changed since the previous snapshot, plus headline metrics

The summary is generated WITHOUT an LLM call — it's pure data
transformation. So snapshotting is fast and free.

Use
---
The session calls SnapshotWriter.snapshot(session, phase_label) once at
each phase boundary. The writer maintains its own state to compute
deltas.

Resilience
----------
A snapshot failure never aborts the game. We log and continue.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from diplomacy_engine import POWERS, supply_centers_owned, units_by_power
from diplomacy_inspection import mind_to_inspection_dict


SNAPSHOT_ROOT = Path.home() / ".diplomacy_llm" / "snapshots"


def _safe_phase_label(state) -> str:
    """e.g. 1902-FALL-MOVEMENT — safe for filenames."""
    return f"{state.year}-{state.season}-{state.phase}"


def _isodate(t: float) -> str:
    """e.g. 2026-05-06_14-32-08 — safe for directory names."""
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H-%M-%S")


# ----------------------------------------------------------------------
# Summary-builder (pure functions, no LLM)
# ----------------------------------------------------------------------


def _per_power_stats(state) -> dict[str, dict]:
    out = {}
    for p in POWERS:
        scs = supply_centers_owned(state, p)
        units = list(units_by_power(state, p))
        out[p] = {
            "sc_count": len(scs),
            "supply_centers": sorted(scs),
            "unit_count": len(units),
            "units": [f"{u.kind} {u.location}" for u in units],
            "eliminated": p in state.eliminated,
        }
    return out


def _aggregate_mind_metrics(minds: dict[str, dict]) -> dict:
    """Sum/avg over all minds — light counts only, doesn't need much."""
    n_beliefs = sum(len(m.get("beliefs", [])) for m in minds.values())
    n_predictions = sum(len(m.get("predictions", [])) for m in minds.values())
    n_intents = sum(len(m.get("strategic_intents", [])) for m in minds.values())
    n_self_commits = sum(
        len(m.get("self_commitments", [])) for m in minds.values())
    n_incoming = sum(
        len(m.get("incoming_commitments", [])) for m in minds.values())
    return {
        "total_beliefs": n_beliefs,
        "total_predictions": n_predictions,
        "total_strategic_intents": n_intents,
        "total_self_commitments": n_self_commits,
        "total_incoming_commitments": n_incoming,
    }


def _delta_paragraph(prev: Optional[dict], curr: dict) -> str:
    """One-paragraph natural-language delta. No LLM."""
    parts = []
    cstats = curr["per_power"]
    if prev is None:
        parts.append(f"Game opens at {curr['phase']}.")
        leaders = sorted(cstats.items(), key=lambda kv: -kv[1]["sc_count"])[:3]
        names = ", ".join(f"{p} {s['sc_count']}sc" for p, s in leaders)
        parts.append(f"Standings: {names}.")
        return " ".join(parts)
    pstats = prev["per_power"]
    parts.append(f"Phase {curr['phase']}.")
    # SC changes
    gainers = []
    losers = []
    for p in POWERS:
        delta = cstats[p]["sc_count"] - pstats[p]["sc_count"]
        if delta > 0:
            gainers.append((p, delta))
        elif delta < 0:
            losers.append((p, delta))
    gainers.sort(key=lambda x: -x[1])
    losers.sort(key=lambda x: x[1])
    if gainers:
        parts.append("Gained: " + ", ".join(f"{p} +{d}" for p, d in gainers) + ".")
    if losers:
        parts.append("Lost: " + ", ".join(f"{p} {d}" for p, d in losers) + ".")
    if not gainers and not losers:
        parts.append("No SC changes.")
    # Eliminations
    new_elim = [p for p in POWERS
                if cstats[p]["eliminated"] and not pstats[p]["eliminated"]]
    if new_elim:
        parts.append("Eliminated: " + ", ".join(new_elim) + ".")
    # Mind growth
    pmm = prev["mind_metrics"]; cmm = curr["mind_metrics"]
    if cmm["total_beliefs"] > pmm["total_beliefs"]:
        parts.append(
            f"Beliefs across all minds grew "
            f"{pmm['total_beliefs']}→{cmm['total_beliefs']}.")
    if cmm["total_self_commitments"] > pmm["total_self_commitments"]:
        parts.append(
            f"Promises made: {pmm['total_self_commitments']}"
            f"→{cmm['total_self_commitments']}.")
    return " ".join(parts)


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------


class SnapshotWriter:
    """Per-game snapshot writer. One per GameSession.

    Lifecycle:
      writer = SnapshotWriter()           # opens a fresh dir on disk
      writer.snapshot(session, label)     # call after each phase
      writer.dir_path                     # snapshots written here
    """

    def __init__(self, game_label: Optional[str] = None):
        SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
        self.game_label = game_label or _isodate(time.time())
        self.dir_path = SNAPSHOT_ROOT / self.game_label
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._index = 0
        self._prev_summary: Optional[dict] = None

    def snapshot(self, session, label: Optional[str] = None) -> Optional[Path]:
        """Write one snapshot. Returns path on success, None on failure."""
        try:
            phase_label = label or _safe_phase_label(session.state)
            # Build mind dicts via the real inspection helper
            minds = {}
            biopsy = {}
            for power, agent in session.agents.items():
                # Bridge agents expose .v2; raw V2 agents have .mind directly
                v2 = getattr(agent, "v2", None) or agent
                if hasattr(v2, "mind"):
                    minds[power] = mind_to_inspection_dict(
                        v2.mind, include_events=False, message_limit=20)
                else:
                    minds[power] = {"_error": "no .mind on agent"}
                # Capture biopsy (prompt history) — full content per power
                rec = getattr(v2, "llm_call", None)
                if rec is not None and hasattr(rec, "history"):
                    try:
                        from diplomacy_prompt_recorder import infer_call_kind
                        biopsy[power] = {
                            "archetype": getattr(agent, "archetype", "?"),
                            "total_calls": getattr(rec, "total_calls", 0),
                            "total_errors": getattr(rec, "total_errors", 0),
                            "records": [
                                {
                                    "kind": infer_call_kind(r.prompt),
                                    "timestamp": r.timestamp,
                                    "elapsed_seconds": r.elapsed_seconds,
                                    "prompt": r.prompt,
                                    "response": r.response,
                                    "error": r.error,
                                }
                                for r in list(rec.history)
                            ],
                        }
                    except Exception as e:
                        biopsy[power] = {"_error": str(e)}
                else:
                    biopsy[power] = {"_note": "no recorder attached"}
            per_power = _per_power_stats(session.state)
            mm = _aggregate_mind_metrics(minds)
            curr = {
                "phase": phase_label,
                "per_power": per_power,
                "mind_metrics": mm,
            }
            summary_paragraph = _delta_paragraph(self._prev_summary, curr)
            self._prev_summary = curr

            # Serialize message log — convert legacy Message dataclass
            messages = []
            for m in session.messages:
                messages.append({
                    "sender": getattr(m, "sender", None),
                    "recipients": list(getattr(m, "recipients", []) or []),
                    "text": getattr(m, "text", None) or getattr(m, "body", None),
                    "season": getattr(m, "season", None),
                    "year": getattr(m, "year", None),
                    "public": bool(getattr(m, "public", False)),
                })

            payload = {
                "game_label": self.game_label,
                "snapshot_index": self._index,
                "snapshot_time": time.time(),
                "phase": phase_label,
                "year": session.state.year,
                "season": session.state.season,
                "season_phase": session.state.phase,
                "summary": summary_paragraph,
                "headline": {
                    "leaders": sorted(
                        [(p, s["sc_count"]) for p, s in per_power.items()],
                        key=lambda kv: -kv[1]),
                    "n_messages_so_far": len(messages),
                    "n_log_lines_so_far": len(session.log),
                    "mind_metrics": mm,
                },
                "per_power": per_power,
                "minds": minds,
                "biopsy": biopsy,
                "messages": messages,
                "log": list(session.log),
            }
            fname = f"{self._index:03d}-{phase_label}.json"
            path = self.dir_path / fname
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)

            # Also write a STANDALONE biopsy file alongside, so user can
            # find/inspect prompts without parsing the full snapshot.
            try:
                bio_path = self.dir_path / f"biopsy-{self._index:03d}-{phase_label}.json"
                with open(bio_path, "w") as f:
                    json.dump({
                        "phase": phase_label,
                        "snapshot_time": time.time(),
                        "biopsy": biopsy,
                    }, f, indent=2, default=str)
            except Exception as e:
                print(f"[snapshot] biopsy file write failed: {e}")

            self._index += 1
            return path
        except Exception as e:
            try:
                err_path = self.dir_path / f"_snapshot_error_{int(time.time())}.txt"
                err_path.write_text(
                    f"Snapshot failed: {type(e).__name__}: {e}\n\n" +
                    traceback.format_exc())
            except Exception:
                pass
            return None

    def list_snapshots(self) -> list[Path]:
        return sorted(self.dir_path.glob("*.json"))


# ----------------------------------------------------------------------
# Read-side helpers — for the route
# ----------------------------------------------------------------------


def list_all_games() -> list[dict]:
    """Each known game directory with file count and last-modified time."""
    if not SNAPSHOT_ROOT.exists():
        return []
    out = []
    for p in sorted(SNAPSHOT_ROOT.iterdir()):
        if not p.is_dir():
            continue
        files = sorted(p.glob("*.json"))
        if not files:
            continue
        latest = max(files, key=lambda f: f.stat().st_mtime)
        out.append({
            "label": p.name,
            "n_snapshots": len(files),
            "last_modified": latest.stat().st_mtime,
            "path": str(p),
        })
    out.sort(key=lambda d: -d["last_modified"])
    return out
