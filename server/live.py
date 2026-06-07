"""
live.py — Flask blueprint for the live (substrate-driven) game UI.

Pages:
  GET /             → live game page (was the legacy index)
  GET /play         → same live game page

API:
  GET  /api/live/state           → public state (started, completed_phases, etc.)
  POST /api/live/start           → initialize game with chosen LLM backend
  POST /api/live/run_next_phase  → run one phase synchronously (blocks)
  POST /api/live/auto_play       → start background loop running phases
  POST /api/live/auto_pause      → stop background loop
  GET  /api/live/snapshots       → all snapshots accumulated so far
  GET  /api/live/board           → all board states
  GET  /api/live/messages        → all messages by phase
  GET  /api/live/log             → phase log entries (what happened each phase)
  GET  /api/live/summary         → run summary in the same shape as viewer
  GET  /api/live/bundle          → all four (snapshots+board+messages+log+summary)
                                    in one payload, matching the viewer's bundle
                                    shape so live.js can reuse view.js patterns
"""

from __future__ import annotations

import threading
from flask import Blueprint, jsonify, render_template, request


bp_live = Blueprint("live", __name__)

# Single shared session — the app is single-game for now.
_session = None
_session_init_lock = threading.Lock()


def _get_session():
    global _session
    with _session_init_lock:
        if _session is None:
            from .live_session import LiveSession
            _session = LiveSession()
        return _session


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@bp_live.route("/play")
def play_page():
    return render_template("live.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@bp_live.route("/api/live/state")
def api_state():
    return jsonify(_get_session().public_state())


@bp_live.route("/api/live/start", methods=["POST"])
def api_start():
    body = request.get_json(silent=True) or {}
    llm_kind = body.get("llm_kind", "stub")
    llm_options = body.get("llm_options", {})
    archetype_assignment = body.get("archetype_assignment")
    max_phases = int(body.get("max_phases", 24))

    session = _get_session()
    try:
        state = session.start(
            llm_kind=llm_kind,
            llm_options=llm_options,
            archetype_assignment=archetype_assignment,
            max_phases=max_phases,
        )
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp_live.route("/api/live/run_next_phase", methods=["POST"])
def api_run_next_phase():
    session = _get_session()
    if not session.started:
        return jsonify({"error": "not started"}), 400
    if session.is_running_phase:
        return jsonify(session.public_state())
    try:
        # Run synchronously in the request thread. Caller (the JS) sees the
        # response only after the phase completes. For long phases the JS
        # also has its poll loop running, so it'll show "thinking".
        # To avoid blocking the rest of the app, we run it in a thread and
        # immediately return state showing "running"; the UI polls until done.
        def _bg():
            try:
                session.run_next_phase()
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()
        # Give the worker a moment to flip is_running_phase
        import time as _t
        for _ in range(20):
            if session.is_running_phase:
                break
            _t.sleep(0.05)
        return jsonify(session.public_state())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp_live.route("/api/live/auto_play", methods=["POST"])
def api_auto_play():
    session = _get_session()
    if not session.started:
        return jsonify({"error": "not started"}), 400
    return jsonify(session.start_auto_play())


@bp_live.route("/api/live/auto_pause", methods=["POST"])
def api_auto_pause():
    return jsonify(_get_session().pause_auto_play())


@bp_live.route("/api/live/snapshots")
def api_snapshots():
    return jsonify(_get_session().snapshots_payload())


@bp_live.route("/api/live/board")
def api_board():
    return jsonify(_get_session().board_payload())


@bp_live.route("/api/live/messages")
def api_messages():
    return jsonify(_get_session().messages_payload())


@bp_live.route("/api/live/log")
def api_log():
    return jsonify(_get_session().log_payload())


@bp_live.route("/api/live/summary")
def api_summary():
    return jsonify(_get_session().summary_payload())


@bp_live.route("/api/live/bundle")
def api_bundle():
    """Single-call payload mirroring the viewer's /api/view/<folder>/bundle.

    This lets live.js reuse the rendering patterns from view.js with minimal
    glue code. The shape is identical:
        {summary, snapshots, messages, board, log, state}
    """
    session = _get_session()
    return jsonify({
        "state":     session.public_state(),
        "summary":   session.summary_payload(),
        "snapshots": session.snapshots_payload()["snapshots"],
        "messages":  session.messages_payload()["messages"],
        "board":     session.board_payload()["board"],
        "log":       session.log_payload()["log"],
    })
