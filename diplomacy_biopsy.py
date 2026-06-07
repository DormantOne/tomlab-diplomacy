"""
diplomacy_biopsy.py — diagnostic snapshot writers.

Each function takes some piece of run state (a mind, a phase, a message log)
and writes a human-readable file for later inspection. The output format is
a mix of JSON (for things you might want to re-parse) and plain text (for
things you'll just read).

DESIGN NOTES:
  - Everything goes to a run directory so a single experiment's outputs
    stay grouped. Caller passes the directory; we write files inside it.
  - Per-phase KG snapshots are PER-AGENT (one file each). Cheaper to read
    "what was Austria thinking after F1901" than to grep one giant file.
  - Snapshots are appended-style — each phase adds a new file rather
    than overwriting. So you can compare across phases.
  - Run summary is a single file written at end-of-run with cross-game stats.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from diplomacy_kg_schema import (
    AgentMind, BeliefStatus, CommitmentStatus, PredictionStatus,
    StrategicIntentStatus,
)


# ============================================================================
# Helpers
# ============================================================================

def make_run_dir(parent: str = ".") -> str:
    """Create a timestamped directory for this run's biopsy outputs.

    Returns the absolute path."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(parent, f"biopsy_{stamp}"))
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Per-agent per-phase KG snapshot
# ============================================================================

def write_agent_snapshot(
    mind: AgentMind, phase: str, run_dir: str,
) -> str:
    """Write `mind`'s current state to a per-phase JSON file.

    File naming: AUSTRIA_1901-FALL-MOVES.json
    """
    safe_phase = phase.replace(" ", "_")
    path = os.path.join(run_dir, f"{mind.owner_power}_{safe_phase}.json")
    snap = {
        "owner_power": mind.owner_power,
        "archetype": mind.archetype,
        "phase": phase,
        "snapshot_time": time.time(),
        "summary": {
            "beliefs_total": len(mind.beliefs),
            "beliefs_active": sum(
                1 for b in mind.beliefs.values()
                if b.status == BeliefStatus.ACTIVE
            ),
            "beliefs_proto": sum(
                1 for b in mind.beliefs.values()
                if b.status == BeliefStatus.PROTO
            ),
            "beliefs_retired": sum(
                1 for b in mind.beliefs.values()
                if b.status == BeliefStatus.RETIRED
            ),
            "intents_total": len(mind.strategic_intents),
            "intents_active": sum(
                1 for i in mind.strategic_intents.values()
                if i.status == StrategicIntentStatus.ACTIVE
            ),
            "intents_succeeded": sum(
                1 for i in mind.strategic_intents.values()
                if i.status == StrategicIntentStatus.SUCCEEDED
            ),
            "incoming_commitments_pending": sum(
                1 for c in mind.incoming_commitments.values()
                if c.status == CommitmentStatus.PENDING
            ),
            "incoming_commitments_kept": sum(
                1 for c in mind.incoming_commitments.values()
                if c.status == CommitmentStatus.KEPT
            ),
            "incoming_commitments_broken": sum(
                1 for c in mind.incoming_commitments.values()
                if c.status == CommitmentStatus.BROKEN
            ),
            "predictions_total": len(mind.predictions),
            "predictions_open": sum(
                1 for p in mind.predictions.values()
                if p.status == PredictionStatus.OPEN
            ),
            "predictions_confirmed": sum(
                1 for p in mind.predictions.values()
                if p.status == PredictionStatus.CONFIRMED
            ),
            "predictions_refuted": sum(
                1 for p in mind.predictions.values()
                if p.status == PredictionStatus.REFUTED
            ),
            "plan_nodes_total": len(mind.plan_nodes),
            "intent_commitments_active": sum(
                1 for ic in mind.intent_commitments.values()
                if ic.status == "active"
            ),
        },
        "beliefs": [
            {
                "id": b.id, "about": b.about_power,
                "type": b.belief_type.value, "status": b.status.value,
                "head": b.head,
                "hp": round(b.hp, 3),
                "critic_score": round(b.critic_score, 3),
                "times_foveated": b.times_foveated,
                "evidence_for_count": len(b.evidence_for),
                "evidence_against_count": len(b.evidence_against),
                "retire_reason": b.retire_reason,
                "formed_at": b.formed_at_phase,
                "last_updated": b.last_updated_phase,
                "persists_across_games": b.persists_across_games,
            }
            for b in sorted(mind.beliefs.values(),
                            key=lambda b: (b.about_power, b.belief_type.value))
        ],
        "strategic_intents": [
            {
                "id": i.id, "head": i.head,
                "status": i.status.value,
                "target_powers": i.target_powers,
                "target_provinces": i.target_provinces,
                "horizon": i.horizon,
                "formed_at": i.formed_at_phase,
                "active_since": i.active_since_phase,
                "hp": round(i.hp, 3),
                "critic_score": round(i.critic_score, 3),
                "sc_delta_under_intent": i.sc_delta_under_intent,
                "predictions_confirmed": i.predictions_confirmed,
                "predictions_refuted": i.predictions_refuted,
                "supporting_plan_count": len(i.supporting_plan_ids),
                "times_committed": i.times_committed,
                "retire_reason": i.retire_reason,
            }
            for i in sorted(mind.strategic_intents.values(),
                            key=lambda i: i.formed_at_phase)
        ],
        "intent_commitments": [
            {
                "id": ic.id, "intent_id": ic.intent_id,
                "started_at": ic.started_at_phase,
                "window_phases": ic.window_phases,
                "phases_under_commitment": list(ic.phases_under_commitment),
                "plans_followed": ic.plans_followed,
                "plans_diverged": ic.plans_diverged,
                "predictions_confirmed_during": ic.predictions_confirmed_during,
                "predictions_refuted_during": ic.predictions_refuted_during,
                "status": ic.status,
                "end_reason": ic.end_reason,
                "ended_at": ic.ended_at_phase,
            }
            for ic in mind.intent_commitments.values()
        ],
        "incoming_commitments": [
            {
                "id": c.id, "speaker": c.speaker,
                "type": c.type.value,
                "subject_unit": c.subject_unit,
                "subject_province": c.subject_province,
                "target_province": c.target_province,
                "counterparty": c.counterparty,
                "deadline_phase": c.deadline_phase,
                "status": c.status.value,
                "resolved_at": c.resolved_at_phase,
                "evidence_count": len(c.grading_evidence),
                "raw": c.raw_commitspeak_line,
            }
            for c in sorted(mind.incoming_commitments.values(),
                            key=lambda c: c.deadline_phase)
        ],
        "self_commitments": [
            {
                "id": c.id, "to": c.target_power,
                "type": c.type.value,
                "subject_province": c.subject_province,
                "target_province": c.target_province,
                "deadline_phase": c.deadline_phase,
                "status": c.status.value,
                "raw": c.raw_commitspeak_line,
            }
            for c in sorted(mind.self_commitments.values(),
                            key=lambda c: c.deadline_phase)
        ],
        "recent_predictions": [
            {
                "id": p.id, "about": p.about_power,
                "type": p.predicted_event_type,
                "target": p.predicted_target,
                "subject_power": p.predicted_subject_power,
                "window_kind": p.window_kind.value,
                "prediction_window": p.prediction_window,
                "formed_at": p.formed_at_phase,
                "status": p.status.value,
                "confidence": round(p.confidence, 3),
                "rationale": p.rationale[:200] if p.rationale else "",
            }
            for p in sorted(mind.predictions.values(),
                            key=lambda p: p.formed_at_phase, reverse=True)[:30]
        ],
    }
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    return path


# ============================================================================
# Per-phase board state log (one file, appended)
# ============================================================================

def append_board_state(
    state, run_dir: str, *, phase_label: str,
) -> str:
    """Append a brief board snapshot to board_log.txt in run_dir.

    Cheap human-readable record of how the game is shaping up.
    """
    path = os.path.join(run_dir, "board_log.txt")
    from diplomacy_engine import (
        POWERS, supply_centers_owned, units_by_power,
    )
    lines = []
    lines.append(f"=== {phase_label} ===")
    for power in POWERS:
        if power in state.eliminated:
            lines.append(f"  {power[:3]}: eliminated")
            continue
        scs = supply_centers_owned(state, power)
        units = units_by_power(state, power)
        unit_str = ", ".join(f"{u.kind}{u.location}" for u in units) or "-"
        lines.append(f"  {power[:3]}: {len(scs)} SC | {unit_str}")
    lines.append("")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# Message log (one file, all messages chronological)
# ============================================================================

def append_messages(
    messages: list, run_dir: str, *, phase_label: str,
) -> str:
    """Append a phase's messages to messages.txt in run_dir."""
    path = os.path.join(run_dir, "messages.txt")
    lines = [f"=== {phase_label} ==="]
    for m in messages:
        target = "ALL" if getattr(m, "public", False) else (
            ",".join(getattr(m, "recipients", []) or []) or "?"
        )
        body = (getattr(m, "body", None) or "").strip()
        sender = getattr(m, "sender", "?")
        lines.append(f"--- {sender} -> {target} ---")
        lines.append(body)
        lines.append("")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ============================================================================
# Run summary (written at end-of-run)
# ============================================================================

def write_run_summary(
    agents: dict, final_state, run_dir: str,
    *, llm_kind: str, model: str, phases_run: int,
) -> str:
    """One-shot summary written at end-of-game.

    Both human-readable text and structured JSON, side by side.
    """
    text_path = os.path.join(run_dir, "summary.txt")
    json_path = os.path.join(run_dir, "summary.json")

    from diplomacy_engine import (
        POWERS, supply_centers_owned,
    )

    summary_data = {
        "run_metadata": {
            "llm_kind": llm_kind,
            "model": model,
            "phases_run": phases_run,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "agents": {},
    }

    text_lines = []
    text_lines.append("=" * 72)
    text_lines.append(f"RUN SUMMARY")
    text_lines.append("=" * 72)
    text_lines.append(f"  LLM:       {llm_kind} ({model})")
    text_lines.append(f"  Phases:    {phases_run}")
    text_lines.append(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    text_lines.append("")

    for power in POWERS:
        if power not in agents:
            continue
        agent = agents[power]
        mind = agent.mind
        scs = supply_centers_owned(final_state, power)

        # Belief stats
        b_active = sum(1 for b in mind.beliefs.values()
                       if b.status == BeliefStatus.ACTIVE)
        b_proto = sum(1 for b in mind.beliefs.values()
                      if b.status == BeliefStatus.PROTO)
        b_retired = sum(1 for b in mind.beliefs.values()
                        if b.status == BeliefStatus.RETIRED)
        b_retired_reasons = {}
        for b in mind.beliefs.values():
            if b.status == BeliefStatus.RETIRED and b.retire_reason:
                b_retired_reasons[b.retire_reason] = (
                    b_retired_reasons.get(b.retire_reason, 0) + 1
                )

        # Intent stats
        i_active = sum(1 for i in mind.strategic_intents.values()
                       if i.status == StrategicIntentStatus.ACTIVE)
        i_succeeded = sum(1 for i in mind.strategic_intents.values()
                          if i.status == StrategicIntentStatus.SUCCEEDED)
        i_failed = sum(1 for i in mind.strategic_intents.values()
                       if i.status == StrategicIntentStatus.FAILED)
        i_retired = sum(1 for i in mind.strategic_intents.values()
                        if i.status == StrategicIntentStatus.RETIRED)

        # Commitment stats
        c_kept = sum(1 for c in mind.incoming_commitments.values()
                     if c.status == CommitmentStatus.KEPT)
        c_broken = sum(1 for c in mind.incoming_commitments.values()
                       if c.status == CommitmentStatus.BROKEN)
        c_irrel = sum(1 for c in mind.incoming_commitments.values()
                      if c.status == CommitmentStatus.IRRELEVANT)
        c_pending = sum(1 for c in mind.incoming_commitments.values()
                        if c.status == CommitmentStatus.PENDING)

        # Self-commitment stats
        sc_kept = sum(1 for c in mind.self_commitments.values()
                      if c.status == CommitmentStatus.KEPT)
        sc_broken = sum(1 for c in mind.self_commitments.values()
                        if c.status == CommitmentStatus.BROKEN)

        # Prediction stats
        p_confirmed = sum(1 for p in mind.predictions.values()
                          if p.status == PredictionStatus.CONFIRMED)
        p_refuted = sum(1 for p in mind.predictions.values()
                        if p.status == PredictionStatus.REFUTED)
        p_partial = sum(1 for p in mind.predictions.values()
                        if p.status == PredictionStatus.PARTIAL)
        p_open = sum(1 for p in mind.predictions.values()
                     if p.status == PredictionStatus.OPEN)

        agent_data = {
            "archetype": mind.archetype,
            "final_sc": len(scs),
            "messages_total": len(mind.message_events),
            "beliefs": {
                "active": b_active, "proto": b_proto, "retired": b_retired,
                "retire_reasons": b_retired_reasons,
            },
            "intents": {
                "active": i_active, "succeeded": i_succeeded,
                "failed": i_failed, "retired": i_retired,
            },
            "incoming_commitments": {
                "kept": c_kept, "broken": c_broken,
                "irrelevant": c_irrel, "pending": c_pending,
                "credibility_rate": round(c_kept / max(c_kept + c_broken, 1), 3),
            },
            "self_commitments": {
                "kept": sc_kept, "broken": sc_broken,
                "self_credibility_rate": round(sc_kept / max(sc_kept + sc_broken, 1), 3),
            },
            "predictions": {
                "confirmed": p_confirmed, "refuted": p_refuted,
                "partial": p_partial, "open": p_open,
                "accuracy": round(p_confirmed / max(p_confirmed + p_refuted, 1), 3),
            },
            "plans_total": len(mind.plan_nodes),
            "intent_commitments_total": len(mind.intent_commitments),
        }
        summary_data["agents"][power] = agent_data

        text_lines.append(f"--- {power} ({mind.archetype}) ---")
        text_lines.append(f"  Final SC: {len(scs)}")
        text_lines.append(f"  Messages sent/received: {len(mind.message_events)}")
        text_lines.append(f"  Beliefs: active={b_active}, proto={b_proto}, retired={b_retired}")
        if b_retired_reasons:
            for reason, n in b_retired_reasons.items():
                text_lines.append(f"    - {reason}: {n}")
        text_lines.append(f"  Intents: active={i_active}, succeeded={i_succeeded}, "
                          f"failed={i_failed}, retired={i_retired}")
        text_lines.append(f"  Other-power commitments seen: kept={c_kept}, broken={c_broken}, "
                          f"irrel={c_irrel}, pending={c_pending}")
        if c_kept + c_broken > 0:
            text_lines.append(f"    -> credibility rate of others toward me: "
                              f"{c_kept / (c_kept + c_broken):.1%}")
        text_lines.append(f"  My own commitments: kept={sc_kept}, broken={sc_broken}")
        if sc_kept + sc_broken > 0:
            text_lines.append(f"    -> my own credibility rate: "
                              f"{sc_kept / (sc_kept + sc_broken):.1%}")
        text_lines.append(f"  Predictions: confirmed={p_confirmed}, refuted={p_refuted}, "
                          f"partial={p_partial}, open={p_open}")
        if p_confirmed + p_refuted > 0:
            text_lines.append(f"    -> theory-of-mind accuracy: "
                              f"{p_confirmed / (p_confirmed + p_refuted):.1%}")
        text_lines.append(f"  Plans authored: {len(mind.plan_nodes)}")
        text_lines.append(f"  Intent commitments: {len(mind.intent_commitments)}")
        text_lines.append("")

    with open(text_path, "w") as f:
        f.write("\n".join(text_lines))
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    return text_path


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from diplomacy_kg_schema import (
        AgentMind, CharacterBrief, BeliefNode, BeliefType, BeliefStatus,
        new_id,
    )

    print("=" * 72)
    print("BIOPSY SANITY CHECK")
    print("=" * 72)

    # Build a minimal mind
    mind = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    mind.character_brief = CharacterBrief(
        id=new_id("brief"), archetype="MARSHAL_VEIL",
        text="I am Marshal Veil.", generated_at=time.time(),
    )
    b = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.RELATIONSHIP,
        head="Russia coordinates with Austria this game.",
        body="(elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1902-SPRING-MOVES",
        status=BeliefStatus.ACTIVE,
    )
    mind.beliefs[b.id] = b

    with tempfile.TemporaryDirectory() as tmp:
        path = write_agent_snapshot(mind, "1902-SPRING-MOVES", tmp)
        print(f"  Wrote snapshot: {os.path.basename(path)} "
              f"({os.path.getsize(path)} bytes)")
        with open(path) as f:
            data = json.load(f)
        assert data["owner_power"] == "FRANCE"
        assert len(data["beliefs"]) == 1
        assert data["beliefs"][0]["about"] == "RUSSIA"
        print(f"  Round-trip ok.")

    print()
    print("Biopsy sanity check passed.")
