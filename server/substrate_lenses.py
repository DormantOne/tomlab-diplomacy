"""
substrate_lenses.py — Flask blueprint exposing the substrate's mind to the GUI.

This is the Phase 1 user-facing deliverable. Seven read-only routes pull
data DIRECTLY from each agent's live AgentMind via diplomacy_inspection,
no LLM call, no game mutation:

  GET /api/kg/lenses               → list of available lenses (introspection)
  GET /api/kg/<power>              → full inspection dict (catalog)
  GET /api/kg/<power>/beliefs      → beliefs + belief_revisions
  GET /api/kg/<power>/predictions  → predictions, partitioned by status
  GET /api/kg/<power>/commitments  → incoming + self, partitioned by status
  GET /api/kg/<power>/intents      → strategic_intents + plans + intent_commitments
  GET /api/kg/<power>/identity     → character_brief + identity_constraints
  GET /api/kg/<power>/lifecycle    → per-phase telemetry from agent.phase_logs
  GET /api/kg/<power>/tom          → Theory of Mind, pivoted by target power (FLAGSHIP)

URL prefix `/api/kg/...` is distinct from the existing
`/api/substrate/...` (observer-based reconstruction) and from the legacy
`/kg/<power>/<graph>` (legacy six-graph KG). Three coexist; the substrate
lenses are the new home.

This module does NOT depend on session.py being migrated to
DiplomacyAgentV2 yet. If the live session is still on the legacy LLMAgent,
each route returns a 409 with a clear migration-required message rather
than crashing. Once session.py is migrated, the routes light up
automatically — no further changes here.

To register: in server/app.py add
    from .substrate_lenses import bp_lenses
    app.register_blueprint(bp_lenses)
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from diplomacy_engine import POWERS
from diplomacy_inspection import (
    mind_to_inspection_dict, theory_of_mind_view, lifecycle_view,
)


bp_lenses = Blueprint("kg_lenses", __name__)


# Catalog of available lenses. The front-end can hit /api/kg/lenses to get
# this list rather than hard-coding it.
LENSES = [
    {
        "id": "tom",
        "title": "Theory of Mind",
        "subtitle": "What this agent thinks about each other power",
        "flagship": True,
    },
    {
        "id": "beliefs",
        "title": "Beliefs",
        "subtitle": "Structured claims about other powers",
    },
    {
        "id": "predictions",
        "title": "Predictions",
        "subtitle": "Forecasted moves and their confirm/refute status",
    },
    {
        "id": "commitments",
        "title": "Commitments",
        "subtitle": "Promises made and received, with kept/broken status",
    },
    {
        "id": "intents",
        "title": "Intents",
        "subtitle": "Strategic plans, supporting plans, and intent commitments",
    },
    {
        "id": "identity",
        "title": "Identity",
        "subtitle": "Character brief and seeded constraints",
    },
    {
        "id": "lifecycle",
        "title": "Lifecycle",
        "subtitle": "Per-phase substrate telemetry",
    },
]


# ============================================================================
# Helpers
# ============================================================================


def _get_session():
    """Reach into app module for the singleton GameSession."""
    from . import app as _app
    return getattr(_app, "session", None)


def _resolve_agent(power_raw: str):
    """Return (agent, error_response_or_None).

    Translates URL `power` to upper-case, looks it up in the session,
    and verifies the agent has a substrate `mind` attribute. If any of
    those checks fail, returns (None, response) where response is a
    Flask jsonify result with appropriate HTTP status.
    """
    session = _get_session()
    if session is None:
        return None, (jsonify({
            "error": "no_session",
            "message": "No game session — start a game first.",
        }), 400)

    power = power_raw.upper()
    if power not in POWERS:
        return None, (jsonify({
            "error": "unknown_power",
            "message": f"Power '{power}' is not in {sorted(POWERS)}.",
        }), 404)

    agent = session.agents.get(power)
    if agent is None:
        return None, (jsonify({
            "error": "no_agent",
            "message": (
                f"No AI agent for {power}. "
                f"This typically means {power} is the human player."
            ),
        }), 404)

    if not hasattr(agent, "mind"):
        return None, (jsonify({
            "error": "no_substrate",
            "message": (
                "The active session is using the legacy LLMAgent, which "
                "does not expose a substrate mind. Migrate session.py to "
                "DiplomacyAgentV2 to enable substrate lenses."
            ),
            "available_attrs": sorted(
                a for a in dir(agent) if not a.startswith("_")
            ),
        }), 409)

    return agent, None


def _valid_powers_for_session() -> set[str]:
    """All non-eliminated powers in the live session, used by the ToM lens.

    Falls back to the engine's full POWERS list if the session isn't loaded
    or eliminations can't be read, since the ToM view filters out the
    viewer itself anyway.
    """
    session = _get_session()
    if session is None or not hasattr(session, "state"):
        return set(POWERS)
    eliminated = set(getattr(session.state, "eliminated", set()))
    return set(POWERS) - eliminated


# ============================================================================
# Routes
# ============================================================================


@bp_lenses.route("/api/kg/lenses")
def list_lenses():
    """Introspection endpoint — what lenses are available."""
    return jsonify({"lenses": LENSES})


@bp_lenses.route("/api/kg/<power>")
def catalog(power):
    """The full inspection dict — useful for debugging and as a fallback
    if a specific lens is unavailable."""
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    return jsonify(mind_to_inspection_dict(agent.mind))


@bp_lenses.route("/api/kg/<power>/beliefs")
def beliefs(power):
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    full = mind_to_inspection_dict(agent.mind, include_events=False)
    # Group beliefs by status for renderer convenience.
    by_status = {"proto": [], "active": [], "retired": [], "revised": []}
    for b in full["beliefs"]:
        by_status.setdefault(b["status"], []).append(b)
    # Group by belief type as well.
    by_type = {}
    for b in full["beliefs"]:
        by_type.setdefault(b["belief_type"], []).append(b)
    return jsonify({
        "viewer": full["owner_power"],
        "items": full["beliefs"],
        "by_status": by_status,
        "by_type": by_type,
        "revisions": full["belief_revisions"],
        "count": len(full["beliefs"]),
    })


@bp_lenses.route("/api/kg/<power>/predictions")
def predictions(power):
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    full = mind_to_inspection_dict(agent.mind, include_events=False)
    by_status = {"open": [], "confirmed": [], "refuted": [],
                 "partial": [], "superseded": []}
    for p in full["predictions"]:
        by_status.setdefault(p["status"], []).append(p)
    # Sort each bucket by confidence desc.
    for k in by_status:
        by_status[k].sort(key=lambda p: -p["confidence"])
    return jsonify({
        "viewer": full["owner_power"],
        "items": full["predictions"],
        "by_status": by_status,
        "count": len(full["predictions"]),
    })


@bp_lenses.route("/api/kg/<power>/commitments")
def commitments(power):
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    full = mind_to_inspection_dict(agent.mind, include_events=False)

    def _partition(rows):
        out = {"pending": [], "kept": [], "broken": [],
               "irrelevant": [], "unparseable": []}
        for c in rows:
            out.setdefault(c["status"], []).append(c)
        return out

    return jsonify({
        "viewer": full["owner_power"],
        "incoming": {
            "items": full["incoming_commitments"],
            "by_status": _partition(full["incoming_commitments"]),
            "count": len(full["incoming_commitments"]),
        },
        "self": {
            "items": full["self_commitments"],
            "by_status": _partition(full["self_commitments"]),
            "count": len(full["self_commitments"]),
        },
    })


@bp_lenses.route("/api/kg/<power>/intents")
def intents(power):
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    full = mind_to_inspection_dict(agent.mind, include_events=False)
    by_status = {}
    for i in full["strategic_intents"]:
        by_status.setdefault(i["status"], []).append(i)
    return jsonify({
        "viewer": full["owner_power"],
        "intents": {
            "items": full["strategic_intents"],
            "by_status": by_status,
            "count": len(full["strategic_intents"]),
        },
        "plans": {
            "items": full["plan_nodes"],
            "count": len(full["plan_nodes"]),
        },
        "intent_commitments": {
            "items": full["intent_commitments"],
            "count": len(full["intent_commitments"]),
        },
        "revisions": {
            "items": full["intent_revisions"],
            "count": len(full["intent_revisions"]),
        },
    })


@bp_lenses.route("/api/kg/<power>/identity")
def identity(power):
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    full = mind_to_inspection_dict(agent.mind, include_events=False)
    return jsonify({
        "viewer": full["owner_power"],
        "archetype": full["archetype"],
        "games_played": full["games_played"],
        "character_brief": full["character_brief"],
        "identity_constraints": full["identity_constraints"],
    })


@bp_lenses.route("/api/kg/<power>/lifecycle")
def lifecycle(power):
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    phase_logs = getattr(agent, "phase_logs", []) or []
    items = lifecycle_view(phase_logs)
    return jsonify({
        "viewer": agent.mind.owner_power,
        "items": items,
        "count": len(items),
        "totals": {
            "negotiate_calls": sum(x["negotiate_calls"] for x in items),
            "commitments_graded": sum(x["commitments_graded"] for x in items),
            "predictions_graded": sum(x["predictions_graded"] for x in items),
            "beliefs_promoted": sum(x["beliefs_promoted"] for x in items),
            "beliefs_retired": sum(x["beliefs_retired"] for x in items),
            "intents_promoted": sum(x["intents_promoted"] for x in items),
            "intents_retired": sum(x["intents_retired"] for x in items),
            "revisions_proposed": sum(x["revision_proposals_made"] for x in items),
        },
    })


@bp_lenses.route("/api/kg/<power>/tom")
def theory_of_mind(power):
    """Flagship lens — the cross-cutting Theory of Mind view.

    Pivots the agent's mind by TARGET POWER, not by record type. Each
    other-power gets a card aggregating beliefs about them, predictions
    of their moves, commitment ledger (kept/broken), our intents
    targeting them, and recent message exchanges.

    This is the lens that demonstrates the substrate is paying off —
    everything the agent has learned about Russia, on one card.
    """
    agent, err = _resolve_agent(power)
    if err is not None:
        return err
    valid = _valid_powers_for_session()
    return jsonify(theory_of_mind_view(agent.mind, valid))


# ============================================================================
# Sanity check
# ============================================================================
# The blueprint can be smoke-tested without spinning up the full Flask app
# by mounting it on a stub.

if __name__ == "__main__":
    import sys
    from flask import Flask
    from diplomacy_kg_schema import (
        AgentMind, CharacterBrief, BeliefNode, BeliefType, BeliefStatus,
        new_id,
    )
    import time as _t

    print("=" * 72)
    print("SUBSTRATE LENSES BLUEPRINT SANITY CHECK")
    print("=" * 72)

    # Build a stub session with one substrate-bearing agent.
    class StubAgent:
        def __init__(self, mind):
            self.mind = mind
            self.phase_logs = []

    class StubState:
        eliminated = set()

    class StubSession:
        def __init__(self):
            mind = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
            mind.character_brief = CharacterBrief(
                id=new_id("brief"), archetype="MARSHAL_VEIL",
                text="I am Marshal Veil.", generated_at=_t.time(),
            )
            b = BeliefNode(
                id=new_id("belief"), about_power="RUSSIA",
                belief_type=BeliefType.CREDIBILITY,
                head="Russia keeps tactical promises.",
                body="(elided)",
                formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
                last_updated_phase="1901-FALL-MOVES",
                status=BeliefStatus.ACTIVE, persists_across_games=True,
            )
            mind.beliefs[b.id] = b
            self.agents = {"FRANCE": StubAgent(mind)}
            self.state = StubState()

    # Inject a stub `app` module so _get_session can find a session.
    import types
    fake_app_pkg = types.ModuleType("__main___app")
    fake_app_pkg.session = StubSession()
    # Patch the lookup to use our stub instead of the real .app import.
    def _fake_get_session():
        return fake_app_pkg.session
    sys.modules[__name__]._get_session = _fake_get_session

    # Mount blueprint on a bare Flask app for testing.
    app = Flask(__name__)
    app.register_blueprint(bp_lenses)
    client = app.test_client()

    def _check(path, expected_keys, status=200):
        r = client.get(path)
        assert r.status_code == status, f"{path}: status {r.status_code} != {status}"
        if status == 200:
            data = r.get_json()
            for k in expected_keys:
                assert k in data, f"{path}: missing key '{k}' in {sorted(data.keys())}"
            print(f"  {path:42s} → 200, keys: {sorted(data.keys())[:6]}{'...' if len(data) > 6 else ''}")
        else:
            print(f"  {path:42s} → {status}")

    _check("/api/kg/lenses", ["lenses"])
    _check("/api/kg/FRANCE", ["owner_power", "beliefs", "counts"])
    _check("/api/kg/FRANCE/beliefs", ["items", "by_status", "by_type"])
    _check("/api/kg/FRANCE/predictions", ["items", "by_status"])
    _check("/api/kg/FRANCE/commitments", ["incoming", "self"])
    _check("/api/kg/FRANCE/intents", ["intents", "plans", "intent_commitments"])
    _check("/api/kg/FRANCE/identity", ["archetype", "character_brief"])
    _check("/api/kg/FRANCE/lifecycle", ["items", "totals"])
    _check("/api/kg/FRANCE/tom", ["viewer", "by_target"])

    # Error paths
    _check("/api/kg/ATLANTIS", [], status=404)   # unknown power
    _check("/api/kg/ENGLAND", [], status=404)    # power not in agents (no agent)

    # Verify ToM payload shape
    r = client.get("/api/kg/FRANCE/tom")
    tom = r.get_json()
    assert tom["viewer"] == "FRANCE"
    assert "RUSSIA" in tom["by_target"]
    russia = tom["by_target"]["RUSSIA"]
    assert russia["credibility_belief"] is not None
    assert russia["ledger"] == {"kept": 0, "broken": 0, "pending": 0, "irrelevant": 0}
    assert russia["trust"] is None  # no resolved promises yet
    print(f"\n  ToM payload looks right: trust={russia['trust']}, "
          f"credibility_belief='{russia['credibility_belief']['head'][:40]}...'")

    print()
    print("Substrate lenses blueprint sanity check passed.")
