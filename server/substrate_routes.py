"""
substrate_routes.py — Flask blueprint that exposes the LIVE V2 minds
(no LLM reconstruction needed).

Routes (same names as the previous observer-based implementation, so the
GUI doesn't change):
  POST /api/substrate/build       → no-op (kept for UI compat); just confirms
  GET  /api/substrate/status      → quick status payload
  GET  /api/substrate/<power>     → live mind for one power
  GET  /api/substrate/all         → all live minds + status
  GET  /api/substrate/snapshots   → list of saved snapshot bundles
  GET  /api/substrate/snapshot/<game>/<filename>  → one snapshot

Data shape is identical to what diplomacy_inspection.mind_to_inspection_dict
returns. Field renames for the inspector:
  predictions  →  recent_predictions   (the JS expects this name)
"""

from __future__ import annotations

import time
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from diplomacy_inspection import mind_to_inspection_dict
from .snapshot import list_all_games, SNAPSHOT_ROOT


bp_substrate = Blueprint("substrate", __name__)


def _get_session():
    """Reach into app module for the live session singleton."""
    from . import app as _app
    return getattr(_app, "session", None)


def _mind_payload(agent) -> dict:
    """Build inspector-shaped dict from a live agent. Handles both raw V2
    agents and bridged agents (which expose .v2)."""
    v2 = getattr(agent, "v2", None) or agent
    if not hasattr(v2, "mind"):
        return {"_error": "agent has no .mind", "owner_power": getattr(agent, "power", "?")}
    d = mind_to_inspection_dict(v2.mind, include_events=False, message_limit=20)
    # The JS expects the key `recent_predictions`; inspection emits `predictions`.
    if "predictions" in d and "recent_predictions" not in d:
        d["recent_predictions"] = d.pop("predictions")
    # Add a `summary` field if a character_brief exists, for the Identity lens
    if d.get("character_brief"):
        d["summary"] = d["character_brief"].get("text", "")
    # Add the current phase string (matches what observer used to provide)
    sess = _get_session()
    if sess is not None and hasattr(sess, "state"):
        d["phase"] = (
            f"{sess.state.year}-{sess.state.season}-{sess.state.phase}")
    return d


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


@bp_substrate.route("/api/substrate/build", methods=["POST"])
def api_build():
    """Kept as a no-op for compatibility with the inspector's Build/Refresh
    button. Live minds are always current — no LLM call is made."""
    sess = _get_session()
    if sess is None:
        return jsonify({"error": "no game session — start a game first"}), 400
    return jsonify({
        "ok": True,
        "note": "minds are live; no rebuild needed",
        "state": "live",
    })


@bp_substrate.route("/api/substrate/status")
def api_status():
    sess = _get_session()
    if sess is None or not sess.agents:
        return jsonify({
            "state": "no_session",
            "last_built_at": None,
            "last_built_phase": None,
            "completed_powers": 0,
        })
    return jsonify({
        "state": "live",
        "last_built_at": time.time(),
        "last_built_phase": (
            f"{sess.state.year}-{sess.state.season}-{sess.state.phase}"),
        "completed_powers": len(sess.agents),
    })


@bp_substrate.route("/api/substrate/all")
def api_all():
    sess = _get_session()
    if sess is None or not sess.agents:
        return jsonify({
            "status": {
                "state": "no_session",
                "last_built_at": None,
                "last_built_phase": None,
                "completed_powers": 0,
            },
            "minds": {},
        })
    minds = {power: _mind_payload(agent) for power, agent in sess.agents.items()}
    return jsonify({
        "status": {
            "state": "live",
            "last_built_at": time.time(),
            "last_built_phase": (
                f"{sess.state.year}-{sess.state.season}-{sess.state.phase}"),
            "completed_powers": len(minds),
        },
        "minds": minds,
    })


@bp_substrate.route("/api/substrate/<power>")
def api_power(power):
    sess = _get_session()
    if sess is None or not sess.agents:
        return jsonify({"error": "no game session"}), 400
    p = power.upper()
    agent = sess.agents.get(p)
    if agent is None:
        return jsonify({"error": f"no agent for {p}"}), 404
    return jsonify(_mind_payload(agent))


# ---- snapshot read API ---------------------------------------------


@bp_substrate.route("/api/substrate/snapshots")
def api_snapshots():
    """List saved snapshot bundles (one per game)."""
    return jsonify({"games": list_all_games()})


@bp_substrate.route("/api/substrate/snapshot/<game>/<fname>")
def api_snapshot_one(game, fname):
    """Return a single snapshot JSON by name."""
    if "/" in game or ".." in game or "/" in fname or ".." in fname:
        return jsonify({"error": "bad path"}), 400
    p = SNAPSHOT_ROOT / game / fname
    if not p.exists() or not p.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(str(p), mimetype="application/json")


# ---- BIOPSY ROUTES — raw prompts, responses, parse failures --------
# No hiding. Whatever the LLM saw, whatever it returned, whatever the
# parser made of it. This is the diagnostic surface.


@bp_substrate.route("/api/biopsy/prompts/<power>")
def api_biopsy_prompts(power):
    """Return the recent (prompt, response) pairs for one power's agent."""
    sess = _get_session()
    if sess is None or not sess.agents:
        return jsonify({"error": "no game session"}), 400
    p = power.upper()
    agent = sess.agents.get(p)
    if agent is None:
        return jsonify({"error": f"no agent for {p}"}), 404
    v2 = getattr(agent, "v2", None) or agent
    rec = getattr(v2, "llm_call", None)
    if rec is None or not hasattr(rec, "history"):
        return jsonify({"error": "agent has no recording wrapper"}), 500
    from diplomacy_prompt_recorder import infer_call_kind
    out = []
    for r in rec.history:
        out.append({
            "timestamp": r.timestamp,
            "elapsed_seconds": r.elapsed_seconds,
            "kind": infer_call_kind(r.prompt),
            "prompt": r.prompt,
            "response": r.response,
            "error": r.error,
            "prompt_chars": len(r.prompt),
            "response_chars": len(r.response or ""),
        })
    return jsonify({
        "power": p,
        "archetype": agent.archetype,
        "total_calls": getattr(rec, "total_calls", len(out)),
        "records": out,
    })


@bp_substrate.route("/api/biopsy/all")
def api_biopsy_all():
    """Summary across all agents — counts and latest call kind per power."""
    sess = _get_session()
    if sess is None or not sess.agents:
        return jsonify({"error": "no game session"}), 400
    from diplomacy_prompt_recorder import infer_call_kind
    out = {}
    for power, agent in sess.agents.items():
        v2 = getattr(agent, "v2", None) or agent
        rec = getattr(v2, "llm_call", None)
        if rec is None or not hasattr(rec, "history"):
            out[power] = {"_error": "no recorder"}
            continue
        history = list(rec.history)
        latest = history[-1] if history else None
        out[power] = {
            "archetype": agent.archetype,
            "total_calls": getattr(rec, "total_calls", len(history)),
            "n_in_buffer": len(history),
            "latest_kind": (infer_call_kind(latest.prompt) if latest else None),
            "latest_at": (latest.timestamp if latest else None),
        }
    return jsonify({"powers": out})


@bp_substrate.route("/api/biopsy/agent/<power>/full")
def api_biopsy_agent_full(power):
    """Full mind dump (no truncation) + recent prompts in one payload."""
    sess = _get_session()
    if sess is None or not sess.agents:
        return jsonify({"error": "no game session"}), 400
    p = power.upper()
    agent = sess.agents.get(p)
    if agent is None:
        return jsonify({"error": f"no agent for {p}"}), 404
    v2 = getattr(agent, "v2", None) or agent
    from diplomacy_inspection import mind_to_inspection_dict
    mind = mind_to_inspection_dict(v2.mind, include_events=True, message_limit=200)
    rec = getattr(v2, "llm_call", None)
    prompts = []
    if rec is not None and hasattr(rec, "history"):
        from diplomacy_prompt_recorder import infer_call_kind
        for r in rec.history:
            prompts.append({
                "timestamp": r.timestamp,
                "kind": infer_call_kind(r.prompt),
                "prompt": r.prompt,
                "response": r.response,
                "error": r.error,
            })
    return jsonify({
        "power": p, "archetype": agent.archetype,
        "phase": (f"{sess.state.year}-{sess.state.season}-{sess.state.phase}"
                  if sess and hasattr(sess, "state") else None),
        "mind": mind,
        "prompts": prompts,
    })
