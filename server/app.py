"""
Flask web server: serves the UI and exposes endpoints to drive the session.
"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory

from agents import PERSONALITIES
from diplomacy_engine import (
    POWERS, GEOMETRY, VIEW_W, VIEW_H, PROVINCES, HOME_CENTERS, ALL_SUPPLY_CENTERS,
)
from .session import (
    GameSession, load_profile, save_profile, load_archetype_kg,
    reset_archetype_memories, load_history,
)
from .view import bp as view_bp
from .live import bp_live
from .substrate_routes import bp_substrate


app = Flask(__name__,
            static_folder="static",
            template_folder="templates")
app.register_blueprint(view_bp)
app.register_blueprint(bp_live)
app.register_blueprint(bp_substrate)

# Calibration assets (custom map background + province coords)
STATIC_DIR = Path(__file__).parent / "static"
CALIBRATION_PATH = STATIC_DIR / "map_calibration.json"
ALLOWED_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "svg"}

session: GameSession | None = None
session_lock = threading.Lock()


def _bg_image_url() -> str | None:
    """Return /static/map_bg.{ext} if a custom bg image is on disk, else None."""
    if not STATIC_DIR.exists():
        return None
    for ext in ALLOWED_EXTS:
        f = STATIC_DIR / f"map_bg.{ext}"
        if f.exists():
            return f"/static/{f.name}"
    return None


def _load_calibration() -> dict | None:
    if CALIBRATION_PATH.exists():
        try:
            return json.loads(CALIBRATION_PATH.read_text())
        except Exception:
            return None
    return None


# ============================================================================
# UI
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================================
# Static-ish data the frontend needs once
# ============================================================================

@app.route("/personalities")
def list_personalities():
    return jsonify({
        k: {"display_name": v["display_name"], "tagline": v["tagline"]}
        for k, v in PERSONALITIES.items()
    })


@app.route("/map")
def map_data():
    """Return polygon geometry for SVG rendering, plus optional custom calibration."""
    out = {}
    for code, geo in GEOMETRY.items():
        logical = "MAO" if code == "MAO_S" else code
        info = PROVINCES.get(logical, ("land", logical, False))
        out[code] = {
            "polygon": geo["polygon"],
            "label": geo.get("label"),
            "sc": geo.get("sc"),
            "kind": info[0],            # land / sea / coast
            "name": info[1],
            "is_sc": info[2],
            "logical": logical,
        }
    custom = _load_calibration()
    return jsonify({
        "view_w": VIEW_W, "view_h": VIEW_H,
        "provinces": out,
        "home_centers": HOME_CENTERS,
        "all_sc": ALL_SUPPLY_CENTERS,
        "powers": POWERS,
        # Custom map (optional). When present, frontend renders the image as the
        # background and uses these coords for province boxes instead of the
        # parchment polygon labels.
        "custom_image_url": _bg_image_url(),
        "custom_coords": (custom or {}).get("provinces") or None,
        "custom_image_size": (custom or {}).get("image_size") or [VIEW_W, VIEW_H],
    })


# ============================================================================
# Map calibration (custom image + province center coordinates)
# ============================================================================

@app.route("/calibrate")
def calibrate_page():
    return render_template("calibrate.html")


@app.route("/upload_map_bg", methods=["POST"])
def upload_map_bg():
    """Accept a base64-encoded image, write to server/static/map_bg.<ext>."""
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    ext = (data.get("ext") or "png").lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported extension '{ext}'"}), 400
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400
    STATIC_DIR.mkdir(exist_ok=True)
    # Remove any prior bg in any extension so we never serve the wrong one
    for old_ext in ALLOWED_EXTS:
        old = STATIC_DIR / f"map_bg.{old_ext}"
        if old.exists():
            old.unlink()
    out = STATIC_DIR / f"map_bg.{ext}"
    out.write_bytes(img_bytes)
    return jsonify({"ok": True, "url": f"/static/map_bg.{ext}", "bytes": len(img_bytes)})


@app.route("/save_calibration", methods=["POST"])
def save_calibration():
    """Save {provinces: {CODE: [x,y]...}, image_size: [w,h]} to disk."""
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "expected JSON object"}), 400
    provs = data.get("provinces") or {}
    if not isinstance(provs, dict):
        return jsonify({"error": "provinces must be a dict"}), 400
    # sanitize: only known province codes, only [int,int] coords
    cleaned = {}
    for code, xy in provs.items():
        code_u = str(code).upper()
        if code_u not in PROVINCES and code_u != "MAO_S":
            continue
        if not (isinstance(xy, list) and len(xy) == 2):
            continue
        try:
            cleaned[code_u] = [int(xy[0]), int(xy[1])]
        except (TypeError, ValueError):
            continue
    size = data.get("image_size") or [VIEW_W, VIEW_H]
    try:
        size = [int(size[0]), int(size[1])]
    except Exception:
        size = [VIEW_W, VIEW_H]
    STATIC_DIR.mkdir(exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps({
        "provinces": cleaned,
        "image_size": size,
    }, indent=2))
    return jsonify({"ok": True, "saved": len(cleaned)})


@app.route("/load_calibration")
def load_calibration_route():
    cal = _load_calibration() or {"provinces": {}, "image_size": [VIEW_W, VIEW_H]}
    cal["image_url"] = _bg_image_url()
    cal["all_provinces"] = sorted(PROVINCES.keys())
    cal["province_kinds"] = {k: v[0] for k, v in PROVINCES.items()}  # land/sea/coast
    cal["home_centers"] = HOME_CENTERS
    return jsonify(cal)


@app.route("/clear_calibration", methods=["POST"])
def clear_calibration():
    n = 0
    if CALIBRATION_PATH.exists():
        CALIBRATION_PATH.unlink(); n += 1
    if STATIC_DIR.exists():
        for ext in ALLOWED_EXTS:
            f = STATIC_DIR / f"map_bg.{ext}"
            if f.exists():
                f.unlink(); n += 1
    return jsonify({"ok": True, "removed": n})


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        data = request.get_json(force=True)
        prof = load_profile()
        if "player_name" in data:
            prof["player_name"] = (data.get("player_name") or "").strip()[:60] or None
        save_profile(prof)
        return jsonify(prof)
    return jsonify(load_profile())


@app.route("/history")
def history():
    return jsonify({"history": load_history()})


@app.route("/reset_memories", methods=["POST"])
def reset_memories():
    n = reset_archetype_memories()
    return jsonify({"ok": True, "deleted": n})


# ============================================================================
# Game lifecycle
# ============================================================================

@app.route("/start", methods=["POST"])
def start():
    global session
    data = request.get_json(force=True)
    power = data.get("power", "FRANCE").upper()
    hide_personalities = bool(data.get("hide_personalities", False))
    hide_ai_chatter = bool(data.get("hide_ai_chatter", True))
    tutorial = bool(data.get("tutorial", False))
    spectator = bool(data.get("spectator", False))
    player_name = (data.get("player_name") or "").strip()[:60]
    # Per-power agent mode for ablation experiments. Map power → mode where
    # mode is "full" | "muted" | "raw_llm". Powers not listed default to "full".
    agent_modes = data.get("agent_modes") or {}
    if not isinstance(agent_modes, dict):
        agent_modes = {}
    # Normalize keys to upper
    agent_modes = {k.upper(): v for k, v in agent_modes.items()
                   if isinstance(v, str) and v in ("full", "muted", "raw_llm")}

    # In tutorial, force user to play France (the script targets France).
    if tutorial:
        power = "FRANCE"
    # In spectator, no human seat — power becomes irrelevant.
    if spectator:
        tutorial = False
        # In spectator, AI chatter must be visible (whole point).
        hide_ai_chatter = False

    with session_lock:
        session = GameSession(
            user_power=("" if spectator else power),
            hide_personalities=hide_personalities,
            hide_ai_chatter=hide_ai_chatter,
            tutorial_mode=tutorial,
            spectator_mode=spectator,
            player_name=player_name,
            agent_modes=agent_modes,
        )
        # Save name to profile for next time
        if player_name:
            prof = load_profile()
            prof["player_name"] = player_name
            save_profile(prof)
    return jsonify({"ok": True, "user_power": power, "tutorial": tutorial,
                    "spectator": spectator, "agent_modes": agent_modes})


@app.route("/end_game", methods=["POST"])
def end_game():
    if session is None:
        return jsonify({"error": "no session"}), 400
    session.end_game_now("resigned")
    return jsonify({"ok": True})


@app.route("/end_reveal")
def end_reveal():
    if session is None:
        return jsonify({"error": "no session"}), 400
    return jsonify(session.end_game_reveal())


# ============================================================================
# State / messages
# ============================================================================

@app.route("/state")
def state():
    if session is None:
        return jsonify({"started": False})
    return jsonify({
        "started": True,
        "board": session.board_summary(),
        "log_tail": session.log[-200:],
        "awaiting": session.awaiting,
        "ai_busy": session.ai_busy,
    })


@app.route("/messages")
def messages():
    if session is None:
        return jsonify({"messages": []})
    return jsonify({
        "messages": [
            {"sender": m.sender, "recipients": list(m.recipients),
             "text": m.text, "season": m.season, "year": m.year,
             "public": m.public}
            for m in session.visible_messages_for_user()
        ]
    })


@app.route("/send_message", methods=["POST"])
def send_message():
    if session is None:
        return jsonify({"error": "no session"}), 400
    data = request.get_json(force=True)
    recipients = [r.upper() for r in data.get("recipients", [])]
    public = bool(data.get("public", False))
    text = data.get("text", "")
    session.user_send_message(recipients, text, public)
    return jsonify({"ok": True})


@app.route("/run_negotiation", methods=["POST"])
def run_negotiation():
    if session is None:
        return jsonify({"error": "no session"}), 400
    threading.Thread(target=session.run_ai_negotiation, daemon=True).start()
    return jsonify({"ok": True, "running": True})


# ============================================================================
# Click-to-order helpers
# ============================================================================

@app.route("/legal/<location>")
def legal(location):
    if session is None:
        return jsonify({"error": "no session"}), 400
    return jsonify(session.legal_destinations(location.upper()))


@app.route("/build_options")
def build_options():
    if session is None:
        return jsonify({"error": "no session"}), 400
    return jsonify(session.legal_build_options())


# ============================================================================
# Order submission and adjudication
# ============================================================================

@app.route("/submit_orders", methods=["POST"])
def submit_orders():
    if session is None:
        return jsonify({"error": "no session"}), 400
    data = request.get_json(force=True)
    lines = data.get("orders", [])
    if isinstance(lines, str):
        lines = [l for l in lines.splitlines() if l.strip()]
    errors = session.submit_user_orders(lines)
    return jsonify({"ok": True, "errors": errors,
                    "staged": [o.signature() for o in session.pending_user_orders]})


@app.route("/run_movement", methods=["POST"])
def run_movement():
    if session is None:
        return jsonify({"error": "no session"}), 400
    threading.Thread(target=session.run_movement_phase, daemon=True).start()
    return jsonify({"ok": True, "running": True})


@app.route("/submit_retreats", methods=["POST"])
def submit_retreats():
    if session is None:
        return jsonify({"error": "no session"}), 400
    data = request.get_json(force=True)
    lines = data.get("orders", [])
    if isinstance(lines, str):
        lines = [l for l in lines.splitlines() if l.strip()]
    errors = session.submit_user_retreats(lines)
    return jsonify({"ok": True, "errors": errors})


@app.route("/run_retreats", methods=["POST"])
def run_retreats():
    if session is None:
        return jsonify({"error": "no session"}), 400
    threading.Thread(target=session.run_retreat_phase, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/submit_builds", methods=["POST"])
def submit_builds():
    if session is None:
        return jsonify({"error": "no session"}), 400
    data = request.get_json(force=True)
    lines = data.get("orders", [])
    if isinstance(lines, str):
        lines = [l for l in lines.splitlines() if l.strip()]
    errors = session.submit_user_builds(lines)
    return jsonify({"ok": True, "errors": errors})


@app.route("/run_adjustment", methods=["POST"])
def run_adjustment():
    if session is None:
        return jsonify({"error": "no session"}), 400
    threading.Thread(target=session.run_adjustment_phase, daemon=True).start()
    return jsonify({"ok": True})


# ============================================================================
# Notes
# ============================================================================

@app.route("/notes", methods=["GET", "POST"])
def notes():
    if session is None:
        return jsonify({"error": "no session"}), 400
    if request.method == "POST":
        data = request.get_json(force=True)
        for power, text in (data or {}).items():
            session.set_user_note(power.upper(), str(text))
        return jsonify({"ok": True})
    return jsonify(session.get_user_notes())


# ============================================================================
# KG inspection
# ============================================================================

@app.route("/kg/<power>/<graph>")
def kg_dump(power, graph):
    if session is None:
        return jsonify({"error": "no session"}), 400
    return jsonify(session.get_kg_dump(power.upper(), graph))


@app.route("/biopsy/<power>")
def biopsy(power):
    """Return the captured prompt/response pairs for an AI agent."""
    if session is None:
        return jsonify({"error": "no session"}), 400
    p = power.upper()
    if p not in session.agents:
        return jsonify({"error": f"no agent for power {p}"}), 404
    agent = session.agents[p]
    archetype = agent.personality_key
    return jsonify({
        "power": p,
        "archetype": archetype if not session.hide_personalities else "Hidden",
        "model": agent.model,
        "biopsy_log": list(agent.biopsy_log),
    })


# ============================================================================
# Tutorial step advancement (UI tracks step number, we just store/expose it)
# ============================================================================

@app.route("/tutorial/advance", methods=["POST"])
def tutorial_advance():
    if session is None:
        return jsonify({"error": "no session"}), 400
    session.advance_tutorial_step()
    return jsonify({"ok": True, "step": session.tutorial_step})


# ============================================================================
# Spectator auto-play
# ============================================================================

@app.route("/auto_play", methods=["POST"])
def auto_play():
    if session is None:
        return jsonify({"error": "no session"}), 400
    if not session.spectator_mode:
        return jsonify({"error": "not in spectator mode"}), 400
    data = request.get_json(silent=True) or {}
    if "speed" in data:
        try:
            session.auto_speed = max(1.0, min(20.0, float(data["speed"])))
        except (TypeError, ValueError):
            pass
    if not session.auto_playing:
        threading.Thread(target=session.run_auto_loop, daemon=True).start()
    return jsonify({"ok": True, "auto_playing": True, "speed": session.auto_speed})


@app.route("/auto_pause", methods=["POST"])
def auto_pause():
    if session is None:
        return jsonify({"error": "no session"}), 400
    if session.spectator_mode:
        session.pause_auto()
    return jsonify({"ok": True, "auto_playing": False})


@app.route("/auto_step", methods=["POST"])
def auto_step():
    if session is None:
        return jsonify({"error": "no session"}), 400
    if not session.spectator_mode:
        return jsonify({"error": "not in spectator mode"}), 400
    session.step_once()
    return jsonify({"ok": True})


# ============================================================================

def main():
    print("Starting server on http://localhost:5050")
    print("Make sure your Anthropic API key is set:")
    print("    export ANTHROPIC_API_KEY=sk-ant-...")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
