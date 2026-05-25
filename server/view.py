"""
view.py — Flask blueprint for the biopsy mind-viewer.

Serves /view and a small JSON API that the frontend uses to load all the
biopsy artifacts produced by run_v2.py. Discoverable folders live at the
project root and match the pattern `biopsy_*`.

Endpoints:
  GET /view                       → list of biopsy folders + redirect helper
  GET /view/<folder>              → render view.html for that folder
  GET /api/view/folders           → JSON list of available biopsy folders
  GET /api/view/<folder>/bundle   → ALL data for that folder in one payload:
                                    summary, snapshots, messages, board states
                                    (the frontend caches this in memory)

Design choice: a single bundled payload, not per-resource endpoints.
Biopsy folders are small (≤ a few MB) and the frontend wants everything
loaded before scrubbing the phase slider feels instant. One round trip,
no waterfall.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from flask import Blueprint, abort, jsonify, render_template, redirect, url_for


bp = Blueprint("view", __name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root = parent of the server/ package.
_SERVER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVER_DIR.parent


def _biopsy_root() -> Path:
    return _PROJECT_ROOT


def _list_folders() -> list[str]:
    root = _biopsy_root()
    if not root.exists():
        return []
    out = sorted(
        (p.name for p in root.iterdir()
         if p.is_dir() and p.name.startswith("biopsy_")),
        reverse=True,  # most recent first
    )
    return out


def _folder_path(name: str) -> Path:
    # Defensive: strict allowlist on folder names. No path traversal.
    if not re.fullmatch(r"biopsy_[A-Za-z0-9_\-]+", name):
        abort(400, "bad folder name")
    p = _biopsy_root() / name
    if not p.is_dir():
        abort(404, f"no such biopsy folder: {name}")
    return p


# ---------------------------------------------------------------------------
# Parsers for the text artifacts
# ---------------------------------------------------------------------------

# messages.txt section header: "=== 1901_SPRING_MOVEMENT ==="
_PHASE_RE = re.compile(r"^===\s+([A-Z0-9_]+)\s+===\s*$")

# Within a phase: "--- AUSTRIA -> RUSSIA ---" or "--- AUSTRIA -> ENGLAND,FRANCE ---"
_MSG_HEADER_RE = re.compile(
    r"^---\s+([A-Z]+)\s+->\s+([A-Z, ]+)\s+---\s*$"
)


def _normalize_phase(label: str) -> str:
    """Convert 'messages.txt' phase format ('1901_SPRING_MOVEMENT') to the
    schema phase format ('1901-SPRING-MOVES'). Also accept already-normalized."""
    s = label.replace("_", "-").upper()
    s = s.replace("MOVEMENT", "MOVES")
    s = s.replace("RETREAT", "RETREATS")
    s = s.replace("ADJUSTMENT", "ADJUSTMENTS")
    return s


def _parse_messages(path: Path) -> dict:
    """Return { phase_key: [ {from, to: [..], text, has_commitspeak} ] }.

    Phase keys are normalized to schema form (`1901-SPRING-MOVES`) so they
    join cleanly to snapshot phases.
    """
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out

    current_phase: Optional[str] = None
    current_msg: Optional[dict] = None
    text_lines: list[str] = []

    def _flush():
        nonlocal current_msg, text_lines
        if current_msg is None or current_phase is None:
            text_lines = []
            return
        text = "\n".join(text_lines).rstrip()
        current_msg["text"] = text
        current_msg["has_commitspeak"] = "[[commit" in text
        out.setdefault(current_phase, []).append(current_msg)
        current_msg = None
        text_lines = []

    for raw in path.read_text(errors="replace").splitlines():
        if (m := _PHASE_RE.match(raw)):
            _flush()
            current_phase = _normalize_phase(m.group(1))
            continue

        if (m := _MSG_HEADER_RE.match(raw)):
            _flush()
            sender = m.group(1).strip()
            recipients_raw = m.group(2)
            recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
            current_msg = {"from": sender, "to": recipients}
            text_lines = []
            continue

        if current_msg is not None:
            text_lines.append(raw)

    _flush()
    return out


# board_log.txt format:
#   === 1901_SPRING_MOVEMENT ===
#     AUS: 3 SC | ASER, FALB, ATRI
#     ENG: 3 SC | FNTH, FNWG, AYOR
#     ...
_BOARD_POWER_RE = re.compile(
    r"^\s+([A-Z]{3}):\s+(\d+)\s+SC\s*\|?\s*(.*)$"
)

_POWER_FROM_ABBREV = {
    "AUS": "AUSTRIA", "ENG": "ENGLAND", "FRA": "FRANCE",
    "GER": "GERMANY", "RUS": "RUSSIA", "TUR": "TURKEY",
    "ITA": "ITALY",
}


def _parse_unit_token(tok: str) -> dict | None:
    """Convert tokens like 'ASER' or 'FALB' to {kind, prov}."""
    tok = tok.strip()
    if len(tok) < 4:
        return None
    kind = tok[0]
    prov = tok[1:].strip().upper()
    if kind not in ("A", "F"):
        return None
    return {"kind": kind, "prov": prov}


def _parse_board(path: Path) -> dict:
    """Return { phase_key: { POWER: { sc_count, units: [ {kind, prov} ] } } }."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out

    current_phase: Optional[str] = None

    for raw in path.read_text(errors="replace").splitlines():
        if (m := _PHASE_RE.match(raw)):
            current_phase = _normalize_phase(m.group(1))
            out.setdefault(current_phase, {})
            continue

        if current_phase is None:
            continue

        if (m := _BOARD_POWER_RE.match(raw)):
            abbr = m.group(1)
            power = _POWER_FROM_ABBREV.get(abbr, abbr)
            sc_count = int(m.group(2))
            units_raw = m.group(3).strip()
            units = []
            if units_raw:
                for tok in units_raw.split(","):
                    u = _parse_unit_token(tok)
                    if u:
                        units.append(u)
            out[current_phase][power] = {"sc_count": sc_count, "units": units}

    return out


def _load_snapshots(folder: Path) -> dict:
    """Return { phase_key: { POWER: snapshot_dict } }."""
    snap: dict[str, dict] = {}
    for f in folder.glob("*.json"):
        if f.name == "summary.json":
            continue
        # Filename is like AUSTRIA_1901-FALL-MOVES.json
        stem = f.stem
        if "_" not in stem:
            continue
        power, _, phase = stem.partition("_")
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        snap.setdefault(phase, {})[power] = data
    return snap


def _load_summary(folder: Path) -> dict:
    p = folder / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/view")
def view_index():
    """Folder picker. If only one folder exists, jump straight to it."""
    folders = _list_folders()
    if not folders:
        return render_template("view.html",
                               folders=[],
                               folder=None,
                               error="No biopsy folders found at project root.")
    if len(folders) == 1:
        return redirect(url_for("view.view_folder", folder=folders[0]))
    return render_template("view.html", folders=folders, folder=None, error=None)


@bp.route("/view/<folder>")
def view_folder(folder: str):
    _folder_path(folder)  # validates
    folders = _list_folders()
    return render_template("view.html",
                           folders=folders,
                           folder=folder,
                           error=None)


@bp.route("/api/view/folders")
def api_folders():
    return jsonify({"folders": _list_folders()})


@bp.route("/api/view/<folder>/bundle")
def api_bundle(folder: str):
    fp = _folder_path(folder)
    payload = {
        "folder": folder,
        "summary": _load_summary(fp),
        "snapshots": _load_snapshots(fp),
        "messages": _parse_messages(fp / "messages.txt"),
        "board": _parse_board(fp / "board_log.txt"),
    }
    return jsonify(payload)
