"""
diplomacy_persistence.py — JSON save/load for AgentMind.

Cross-game rule (per design):
  - DISPOSITION and CREDIBILITY beliefs persist across games (saved + loaded
    in their original status).
  - TACTICAL_PATTERN, RELATIONSHIP, RISK_ASSESSMENT beliefs reset:
    saved as RETIRED with reason=game_ended_resettable. They remain in the
    file (you can inspect what the agent thought) but they don't re-enter
    play in the next game.
  - All StrategicIntents reset (game-specific).
  - Layer 0 events (move/message/phase) are NOT persisted across games —
    they're game-specific evidence and would dilute the credibility ledger
    if carried forward.
  - Identity constraints (personality seed) and character_brief are saved
    so the same agent reloads with the same voice.

The format is JSON-with-enums-as-strings. The dataclasses use Enum types
that don't serialize natively, so we walk the structure manually.

Public functions:
  save_mind(mind, path)   — write to a JSON file
  load_mind(power, archetype, path) — load and return AgentMind
  reset_for_new_game(mind) — apply the cross-game rule in-place
"""

from __future__ import annotations

import json
from typing import Optional

from diplomacy_kg_schema import (
    AgentMind, CharacterBrief,
    BeliefNode, BeliefType, BeliefStatus,
    IdentityConstraintNode,
    StrategicIntentStatus,
)


# ============================================================================
# Helpers — convert dataclass instances to/from dicts
# ============================================================================

def _enum_value(x):
    """Get .value for Enum, str for str, return x for None."""
    if x is None:
        return None
    if hasattr(x, "value"):
        return x.value
    return x


def _belief_to_dict(b: BeliefNode) -> dict:
    return {
        "id": b.id,
        "about_power": b.about_power,
        "belief_type": b.belief_type.value,
        "head": b.head,
        "body": b.body,
        "formed_at_phase": b.formed_at_phase,
        "formed_in_game": b.formed_in_game,
        "last_updated_phase": b.last_updated_phase,
        "evidence_for": list(b.evidence_for),
        "evidence_against": list(b.evidence_against),
        "hp": b.hp,
        "critic_score": b.critic_score,
        "success": b.success,
        "times_foveated": b.times_foveated,
        "times_inspected": b.times_inspected,
        "status": b.status.value,
        "retire_reason": b.retire_reason,
        "superseded_by": b.superseded_by,
        "persists_across_games": b.persists_across_games,
    }


def _belief_from_dict(d: dict) -> BeliefNode:
    return BeliefNode(
        id=d["id"], about_power=d["about_power"],
        belief_type=BeliefType(d["belief_type"]),
        head=d["head"], body=d["body"],
        formed_at_phase=d["formed_at_phase"],
        formed_in_game=d["formed_in_game"],
        last_updated_phase=d["last_updated_phase"],
        evidence_for=list(d.get("evidence_for", [])),
        evidence_against=list(d.get("evidence_against", [])),
        hp=d.get("hp", 1.0),
        critic_score=d.get("critic_score", 0.5),
        success=d.get("success", 0.0),
        times_foveated=d.get("times_foveated", 0),
        times_inspected=d.get("times_inspected", 0),
        status=BeliefStatus(d["status"]),
        retire_reason=d.get("retire_reason"),
        superseded_by=d.get("superseded_by"),
        persists_across_games=d.get("persists_across_games", False),
    )


def _identity_to_dict(n: IdentityConstraintNode) -> dict:
    return {
        "id": n.id, "kind": n.kind, "label": n.label,
        "weight": n.weight, "note": n.note, "archetype": n.archetype,
    }


def _identity_from_dict(d: dict) -> IdentityConstraintNode:
    return IdentityConstraintNode(
        id=d["id"], kind=d["kind"], label=d["label"],
        weight=d.get("weight", 1.0), note=d.get("note", ""),
        archetype=d["archetype"],
    )


# ============================================================================
# Save / load
# ============================================================================

def mind_to_dict(mind: AgentMind) -> dict:
    """Serialize ONLY the cross-game-persistent slice of mind to a dict.

    Per design: identity, character_brief, persisting beliefs, games_played.
    Game-specific data (events, strategic intents, predictions, in-game
    commitments, plan nodes, intent commitments, intent revisions, belief
    revisions on probation) is dropped at save time.
    """
    return {
        "owner_power": mind.owner_power,
        "archetype": mind.archetype,
        "games_played": mind.games_played,
        "character_brief": ({
            "id": mind.character_brief.id,
            "archetype": mind.character_brief.archetype,
            "text": mind.character_brief.text,
            "generated_at": mind.character_brief.generated_at,
        } if mind.character_brief else None),
        "identity_constraints": [
            _identity_to_dict(n) for n in mind.identity_constraints.values()
        ],
        # Persistent beliefs only — DISPOSITION + CREDIBILITY in any status
        # except RETIRED-stale/RETIRED-broad. We'd save TACTICAL/RELATIONSHIP
        # /RISK as RETIRED for archival but per design they don't reload
        # so we drop them entirely on save to keep files small.
        "beliefs": [
            _belief_to_dict(b) for b in mind.beliefs.values()
            if b.persists_across_games
            or b.belief_type in (BeliefType.DISPOSITION, BeliefType.CREDIBILITY)
        ],
    }


def save_mind(mind: AgentMind, path: str) -> None:
    """Write the persistent slice of mind to `path` as JSON."""
    with open(path, "w") as f:
        json.dump(mind_to_dict(mind), f, indent=2)


def load_mind(
    power: str, archetype: str, path: str,
    *, identity_seeder=None,
) -> AgentMind:
    """Load a persisted AgentMind from `path`.

    If the path doesn't exist, returns a fresh AgentMind with
    `archetype` seeded via `identity_seeder` if provided.

    `identity_seeder` is a callable(mind, archetype) that adds
    IdentityConstraintNodes per the personality archetype. (Provided
    by the engine integration layer; not in the schema.)
    """
    import os
    if not os.path.exists(path):
        mind = AgentMind(owner_power=power, archetype=archetype)
        if identity_seeder:
            identity_seeder(mind, archetype)
        return mind

    with open(path) as f:
        data = json.load(f)
    mind = AgentMind(
        owner_power=data.get("owner_power", power),
        archetype=data.get("archetype", archetype),
    )
    mind.games_played = data.get("games_played", 0)
    cb = data.get("character_brief")
    if cb:
        mind.character_brief = CharacterBrief(
            id=cb["id"], archetype=cb["archetype"],
            text=cb["text"], generated_at=cb.get("generated_at", 0),
        )
    for d in data.get("identity_constraints", []):
        n = _identity_from_dict(d)
        mind.identity_constraints[n.id] = n
    for d in data.get("beliefs", []):
        b = _belief_from_dict(d)
        mind.beliefs[b.id] = b
    return mind


def reset_for_new_game(mind: AgentMind) -> None:
    """Apply the cross-game rule in-place to an in-memory mind.

    Use this BETWEEN games when you want to keep the same mind object but
    clear game-specific content. After this:
      - persistent beliefs (DISPOSITION, CREDIBILITY) keep their status
      - non-persistent beliefs are RETIRED with reason=game_ended_resettable
      - all strategic intents → RETIRED
      - plan nodes, predictions, commitments, message events, move events,
        phase states, intent commitments, belief revisions, intent revisions
        → cleared
      - games_played += 1
    """
    # Retire non-persistent beliefs
    for b in mind.beliefs.values():
        if b.belief_type not in (BeliefType.DISPOSITION, BeliefType.CREDIBILITY):
            if b.status not in (BeliefStatus.RETIRED, BeliefStatus.REVISED):
                b.status = BeliefStatus.RETIRED
                b.retire_reason = "game_ended_resettable"

    # Retire all intents
    for i in mind.strategic_intents.values():
        if i.status != StrategicIntentStatus.RETIRED:
            i.status = StrategicIntentStatus.RETIRED

    # Clear game-specific dicts
    mind.move_events.clear()
    mind.message_events.clear()
    mind.phase_states.clear()
    mind.adjustment_events.clear()
    mind.incoming_commitments.clear()
    mind.self_commitments.clear()
    mind.predictions.clear()
    mind.belief_revisions.clear()
    mind.plan_nodes.clear()
    mind.intent_commitments.clear()
    mind.intent_revisions.clear()

    # Drop the now-retired non-persistent beliefs from the dict (they were
    # marked above; tactical/relationship/risk lose their evidence trails
    # on game boundary)
    to_drop = [
        bid for bid, b in mind.beliefs.items()
        if b.belief_type not in (BeliefType.DISPOSITION, BeliefType.CREDIBILITY)
    ]
    for bid in to_drop:
        del mind.beliefs[bid]

    # Drop retired intents to keep things small
    to_drop = [iid for iid, i in mind.strategic_intents.items()]
    for iid in to_drop:
        del mind.strategic_intents[iid]

    mind.games_played += 1


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    import time as _t, tempfile, os
    from diplomacy_kg_schema import (
        AgentMind, CharacterBrief, BeliefNode, BeliefType, BeliefStatus,
        StrategicIntentNode, StrategicIntentStatus, new_id,
    )

    print("=" * 72)
    print("PERSISTENCE SANITY CHECK")
    print("=" * 72)

    mind = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    mind.character_brief = CharacterBrief(
        id=new_id("brief"), archetype="MARSHAL_VEIL",
        text="I am Marshal Veil.", generated_at=_t.time(),
    )

    # Two beliefs, one persistent, one not
    persist = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.CREDIBILITY,
        head="Russia keeps tactical commitments but evades long-term ones.",
        body="(elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1903-FALL-MOVES",
        status=BeliefStatus.ACTIVE, persists_across_games=True,
    )
    transient = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.RELATIONSHIP,
        head="Russia is allied with Austria this game.",
        body="",
        formed_at_phase="1902-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1903-FALL-MOVES",
        status=BeliefStatus.ACTIVE,
    )
    mind.beliefs[persist.id] = persist
    mind.beliefs[transient.id] = transient

    # An intent (game-specific)
    intent = StrategicIntentNode(
        id=new_id("intent"), head="Solo via Med.", body="",
        formed_at_phase="1901-WINTER-ADJUSTMENTS",
        target_powers=["ITALY"], target_provinces=["TUN"],
        horizon="1905-WINTER-ADJUSTMENTS",
        status=StrategicIntentStatus.ACTIVE,
    )
    mind.strategic_intents[intent.id] = intent

    print(f"  Pre-save: {len(mind.beliefs)} beliefs, "
          f"{len(mind.strategic_intents)} intents")

    # Save and reload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    save_mind(mind, path)
    print(f"  Saved to {path} ({os.path.getsize(path)} bytes)")

    mind2 = load_mind("FRANCE", "MARSHAL_VEIL", path)
    print(f"  Reloaded: {len(mind2.beliefs)} beliefs (only persistent kept), "
          f"{len(mind2.strategic_intents)} intents")
    assert len(mind2.beliefs) == 1, f"Expected 1 belief, got {len(mind2.beliefs)}"
    assert len(mind2.strategic_intents) == 0
    assert mind2.character_brief is not None
    assert mind2.character_brief.text == "I am Marshal Veil."

    surviving = list(mind2.beliefs.values())[0]
    assert surviving.belief_type == BeliefType.CREDIBILITY
    print(f"  Surviving belief: {surviving.head[:60]}")

    # Now test reset_for_new_game on the in-memory mind
    print()
    print("  Testing reset_for_new_game on in-memory mind...")
    print(f"  Pre-reset: {len(mind.beliefs)} beliefs, "
          f"{len(mind.strategic_intents)} intents, games_played={mind.games_played}")
    reset_for_new_game(mind)
    print(f"  Post-reset: {len(mind.beliefs)} beliefs, "
          f"{len(mind.strategic_intents)} intents, games_played={mind.games_played}")
    assert len(mind.beliefs) == 1
    assert mind.games_played == 1

    # Cleanup
    os.unlink(path)

    print()
    print("Persistence sanity check passed.")
