"""
diplomacy_inspection.py — full read-only serializer for AgentMind.

UNLIKE diplomacy_persistence.py's mind_to_dict (which writes the cross-game
PERSISTENT slice for save/load: identity, character_brief, persistent
beliefs only), this module emits the FULL in-game state for read-only
inspection by the GUI: predictions, intents, in-flight commitments, plan
nodes, events — everything the agent has actually thought.

This is the foundation Phase 1 of the substrate-GUI integration depends on.
The Flask lens routes in server/substrate_lenses.py call into this module;
the renderer in the front-end consumes its output. Nothing here mutates the
mind — pure serialization.

Public entry points:
  mind_to_inspection_dict(mind, *, include_events=True, message_limit=40)
      → full snapshot dict, JSON-clean
  theory_of_mind_view(mind, valid_powers, *, message_limit=5)
      → dict[other_power → ToM card], the flagship lens data
  lifecycle_view(phase_logs)
      → list[dict] — per-phase substrate telemetry from agent.phase_logs

Design notes
------------
- Enums become their .value strings.
- Sets become sorted lists.
- Dataclasses are walked manually (not via dataclasses.asdict) so we can
  drop noisy fields and keep the JSON predictable.
- Helpers are private (`_xxx_to_dict`) so external callers go through the
  public entry points and we keep one layer of indirection if shapes change.
"""

from __future__ import annotations

from typing import Iterable, Optional

from diplomacy_kg_schema import (
    AgentMind, BeliefNode, BeliefStatus, BeliefType,
    PredictionNode, PredictionStatus,
    CommitmentNode, SelfCommitmentNode, CommitmentStatus, CommitmentType,
    StrategicIntentNode, StrategicIntentStatus,
    PlanNode, IntentCommitmentNode,
    BeliefRevisionProposal, StrategicIntentRevisionProposal,
    IdentityConstraintNode, CharacterBrief,
    MoveEvent, MessageEvent, AdjustmentEvent, PhaseState,
)


# ============================================================================
# Per-record serializers
# ============================================================================
# All helpers are total: they accept the dataclass and emit a JSON-clean dict.
# None checks happen at the call site (e.g. character_brief can be None).


def _enum(x):
    return x.value if hasattr(x, "value") else x


def _belief_to_dict(b: BeliefNode) -> dict:
    return {
        "id": b.id,
        "about_power": b.about_power,
        "belief_type": b.belief_type.value,
        "head": b.head,
        "body": b.body,
        "status": b.status.value,
        "formed_at_phase": b.formed_at_phase,
        "formed_in_game": b.formed_in_game,
        "last_updated_phase": b.last_updated_phase,
        "evidence_for": list(b.evidence_for),
        "evidence_against": list(b.evidence_against),
        "evidence_for_count": len(b.evidence_for),
        "evidence_against_count": len(b.evidence_against),
        "hp": round(b.hp, 4),
        "critic_score": round(b.critic_score, 4),
        "success": round(b.success, 4),
        "times_foveated": b.times_foveated,
        "times_inspected": b.times_inspected,
        "retire_reason": b.retire_reason,
        "superseded_by": b.superseded_by,
        "persists_across_games": b.persists_across_games,
    }


def _belief_revision_to_dict(r: BeliefRevisionProposal) -> dict:
    return {
        "id": r.id,
        "parent_belief_id": r.parent_belief_id,
        "proposed_head": r.proposed_head,
        "proposed_body": r.proposed_body,
        "proposed_belief_type": r.proposed_belief_type.value,
        "reason": r.reason,
        "triggering_prediction_ids": list(r.triggering_prediction_ids),
        "formed_at_phase": r.formed_at_phase,
        "status": r.status,
    }


def _prediction_to_dict(p: PredictionNode) -> dict:
    return {
        "id": p.id,
        "about_power": p.about_power,
        "predicted_event_type": p.predicted_event_type,
        "predicted_target": p.predicted_target,
        "predicted_subject_power": p.predicted_subject_power,
        "prediction_window": p.prediction_window,
        "window_kind": p.window_kind.value,
        "confidence": round(p.confidence, 4),
        "status": p.status.value,
        "formed_at_phase": p.formed_at_phase,
        "resolved_at_phase": p.resolved_at_phase,
        "source_belief_ids": list(p.source_belief_ids),
        "source_message_ids": list(p.source_message_ids),
        "source_observation_ids": list(p.source_observation_ids),
        "parent_intent_id": p.parent_intent_id,
        "grading_evidence": list(p.grading_evidence),
        "rationale": p.rationale,
    }


def _commitment_common(c: CommitmentNode) -> dict:
    return {
        "id": c.id,
        "source_msg_id": c.source_msg_id,
        "speaker": c.speaker,
        "addressees": list(c.addressees),
        "type": c.type.value,
        "subject_unit": c.subject_unit,
        "subject_province": c.subject_province,
        "target_province": c.target_province,
        "counterparty": c.counterparty,
        "deadline_phase": c.deadline_phase,
        "conditional_on": c.conditional_on,
        "status": c.status.value,
        "resolved_at_phase": c.resolved_at_phase,
        "grading_evidence": list(c.grading_evidence),
        "raw_commitspeak_line": c.raw_commitspeak_line,
    }


def _commitment_to_dict(c: CommitmentNode) -> dict:
    return _commitment_common(c)


def _self_commitment_to_dict(c: SelfCommitmentNode) -> dict:
    out = _commitment_common(c)
    out["target_power"] = c.target_power
    return out


def _intent_to_dict(i: StrategicIntentNode) -> dict:
    return {
        "id": i.id,
        "head": i.head,
        "body": i.body,
        "status": i.status.value,
        "formed_at_phase": i.formed_at_phase,
        "active_since_phase": i.active_since_phase,
        "horizon": i.horizon,
        "target_powers": list(i.target_powers),
        "target_provinces": list(i.target_provinces),
        "depends_on_prediction_ids": list(i.depends_on_prediction_ids),
        "supporting_plan_ids": list(i.supporting_plan_ids),
        "requires_permits": list(i.requires_permits),
        "violates_forbids": list(i.violates_forbids),
        "aligned_doctrines": list(i.aligned_doctrines),
        "hp": round(i.hp, 4),
        "critic_score": round(i.critic_score, 4),
        "success": round(i.success, 4),
        "times_committed": i.times_committed,
        "predictions_confirmed": i.predictions_confirmed,
        "predictions_refuted": i.predictions_refuted,
        "sc_delta_under_intent": i.sc_delta_under_intent,
        "succeeded_evidence": list(i.succeeded_evidence),
        "failed_evidence": list(i.failed_evidence),
        "retire_reason": i.retire_reason,
        "superseded_by": i.superseded_by,
    }


def _intent_revision_to_dict(r: StrategicIntentRevisionProposal) -> dict:
    return {
        "id": r.id,
        "parent_intent_id": r.parent_intent_id,
        "proposed_head": r.proposed_head,
        "proposed_body": r.proposed_body,
        "reason": r.reason,
        "triggering_evidence_ids": list(r.triggering_evidence_ids),
        "formed_at_phase": r.formed_at_phase,
        "status": r.status,
    }


def _plan_to_dict(p: PlanNode) -> dict:
    return {
        "id": p.id,
        "head": p.head,
        "body": p.body,
        "status": p.status,
        "formed_at_phase": p.formed_at_phase,
        "parent_intent_id": p.parent_intent_id,
        "motivated_order_ids": list(p.motivated_order_ids),
        "emitted_prediction_ids": list(p.emitted_prediction_ids),
        "plan_outcome": p.plan_outcome,
        "sc_delta_this_phase": p.sc_delta_this_phase,
        "hp": round(p.hp, 4),
        "critic_score": round(p.critic_score, 4),
    }


def _intent_commitment_to_dict(ic: IntentCommitmentNode) -> dict:
    return {
        "id": ic.id,
        "intent_id": ic.intent_id,
        "started_at_phase": ic.started_at_phase,
        "window_phases": ic.window_phases,
        "phases_under_commitment": list(ic.phases_under_commitment),
        "plans_followed": ic.plans_followed,
        "plans_diverged": ic.plans_diverged,
        "predictions_confirmed_during": ic.predictions_confirmed_during,
        "predictions_refuted_during": ic.predictions_refuted_during,
        "sc_delta_during": ic.sc_delta_during,
        "end_reason": ic.end_reason,
        "ended_at_phase": ic.ended_at_phase,
        "status": ic.status,
    }


def _identity_to_dict(n: IdentityConstraintNode) -> dict:
    return {
        "id": n.id,
        "kind": n.kind,
        "label": n.label,
        "weight": round(n.weight, 4),
        "note": n.note,
        "archetype": n.archetype,
    }


def _brief_to_dict(b: CharacterBrief) -> dict:
    return {
        "id": b.id,
        "archetype": b.archetype,
        "text": b.text,
        "generated_at": b.generated_at,
    }


def _move_event_to_dict(m: MoveEvent) -> dict:
    return {
        "id": m.id,
        "phase": m.phase,
        "power": m.power,
        "unit_kind": m.unit_kind,
        "origin": m.origin,
        "order_type": m.order_type,
        "target": m.target,
        "support_of": m.support_of,
        "result": m.result,
        "resolved_at": m.resolved_at,
    }


def _message_event_to_dict(m: MessageEvent) -> dict:
    return {
        "id": m.id,
        "phase": m.phase,
        "sender": m.sender,
        "recipients": list(m.recipients),
        "public": m.public,
        "body": m.body,
        "commitspeak_tail": m.commitspeak_tail,
        "sent_at": m.sent_at,
    }


def _adjustment_event_to_dict(a: AdjustmentEvent) -> dict:
    return {
        "id": a.id,
        "phase": a.phase,
        "power": a.power,
        "kind": a.kind,
        "unit_kind": a.unit_kind,
        "location": a.location,
    }


def _phase_state_to_dict(ps: PhaseState) -> dict:
    # units_by_power values are list[tuple[kind, location]] — convert tuples
    # to lists so JSON round-trips cleanly.
    units_by_power = {
        power: [list(u) if isinstance(u, tuple) else u for u in units]
        for power, units in ps.units_by_power.items()
    }
    return {
        "id": ps.id,
        "phase": ps.phase,
        "sc_owner": dict(ps.sc_owner),
        "units_by_power": units_by_power,
        "eliminated": sorted(ps.eliminated),
        "captured_at": ps.captured_at,
    }


# ============================================================================
# Aggregate snapshot
# ============================================================================


def mind_to_inspection_dict(
    mind: AgentMind,
    *,
    include_events: bool = True,
    message_limit: int = 40,
) -> dict:
    """Serialize the full in-game state of an AgentMind for read-only display.

    Parameters
    ----------
    mind : AgentMind
        The mind to inspect.
    include_events : bool
        If False, drop move/message/adjustment/phase events from the output.
        Saves bandwidth on the catalog endpoint when only the inferential
        layers (beliefs, predictions, etc.) are needed.
    message_limit : int
        Cap on the most-recent message events emitted, sorted by sent_at
        descending. Older messages are summarized as a count.
    """
    out = {
        "owner_power": mind.owner_power,
        "archetype": mind.archetype,
        "games_played": mind.games_played,
        "character_brief": (
            _brief_to_dict(mind.character_brief) if mind.character_brief else None
        ),
        "identity_constraints": [
            _identity_to_dict(n) for n in mind.identity_constraints.values()
        ],
        "beliefs": [_belief_to_dict(b) for b in mind.beliefs.values()],
        "belief_revisions": [
            _belief_revision_to_dict(r) for r in mind.belief_revisions.values()
        ],
        "predictions": [
            _prediction_to_dict(p) for p in mind.predictions.values()
        ],
        "incoming_commitments": [
            _commitment_to_dict(c) for c in mind.incoming_commitments.values()
        ],
        "self_commitments": [
            _self_commitment_to_dict(c) for c in mind.self_commitments.values()
        ],
        "strategic_intents": [
            _intent_to_dict(i) for i in mind.strategic_intents.values()
        ],
        "plan_nodes": [_plan_to_dict(p) for p in mind.plan_nodes.values()],
        "intent_commitments": [
            _intent_commitment_to_dict(ic)
            for ic in mind.intent_commitments.values()
        ],
        "intent_revisions": [
            _intent_revision_to_dict(r) for r in mind.intent_revisions.values()
        ],
        "suspicions": list(getattr(mind, "suspicions", []) or []),
        "private_journal": list(getattr(mind, "private_journal", []) or []),
        "counts": {
            "beliefs": len(mind.beliefs),
            "predictions": len(mind.predictions),
            "incoming_commitments": len(mind.incoming_commitments),
            "self_commitments": len(mind.self_commitments),
            "strategic_intents": len(mind.strategic_intents),
            "plan_nodes": len(mind.plan_nodes),
            "intent_commitments": len(mind.intent_commitments),
            "move_events": len(mind.move_events),
            "message_events": len(mind.message_events),
            "phase_states": len(mind.phase_states),
            "adjustment_events": len(mind.adjustment_events),
        },
    }
    if include_events:
        # Sort messages by sent_at desc and cap. The ToM helper does its own
        # filtering, so the catalog view doesn't need to be exhaustive.
        sorted_msgs = sorted(
            mind.message_events.values(),
            key=lambda m: m.sent_at, reverse=True,
        )
        truncated = sorted_msgs[:message_limit]
        out["message_events"] = [_message_event_to_dict(m) for m in truncated]
        out["message_events_truncated"] = len(sorted_msgs) > message_limit
        out["move_events"] = [
            _move_event_to_dict(m) for m in mind.move_events.values()
        ]
        out["adjustment_events"] = [
            _adjustment_event_to_dict(a) for a in mind.adjustment_events.values()
        ]
        out["phase_states"] = [
            _phase_state_to_dict(ps) for ps in mind.phase_states.values()
        ]
    return out


# ============================================================================
# Theory of Mind — the flagship lens
# ============================================================================
# Pivots the mind by TARGET POWER. For each non-self power, aggregate
# everything the viewer's mind has on that target into one card. This is the
# cross-cutting view the structural lenses (Beliefs, Predictions, ...) can't
# express by themselves.


_BELIEF_TYPE_KEYS = [
    BeliefType.DISPOSITION.value,
    BeliefType.TACTICAL_PATTERN.value,
    BeliefType.RELATIONSHIP.value,
    BeliefType.RISK_ASSESSMENT.value,
    BeliefType.CREDIBILITY.value,
]


def _trust_from_ledger(resolved_kept: int, resolved_broken: int) -> Optional[float]:
    """Trust score = kept / (kept + broken); None if no resolved promises.

    Notes:
      - IRRELEVANT and UNPARSEABLE commitments are excluded by the caller.
      - This is a rough heuristic: broken-late vs. broken-immediately are
        treated equally. Phase 2 can refine with recency weighting.
    """
    n = resolved_kept + resolved_broken
    if n == 0:
        return None
    return round(resolved_kept / n, 4)


def theory_of_mind_view(
    mind: AgentMind,
    valid_powers: Iterable[str],
    *,
    message_limit: int = 5,
) -> dict:
    """Build a per-target-power Theory-of-Mind card.

    Returns
    -------
    dict
        {
          "viewer": <power>,
          "by_target": {
            "<other_power>": {
              "trust": float | None,
              "ledger": {kept, broken, pending, irrelevant},
              "credibility_belief": dict | None,
              "beliefs": [...],
              "beliefs_by_type": {<type>: [...], ...},
              "predictions": {open: [...], confirmed: [...], refuted: [...], partial: [...]},
              "commitments_from_them": [...],
              "commitments_to_them": [...],
              "my_intents_targeting_them": [...],
              "recent_messages_from_them": [...],
              "recent_messages_to_them": [...],
            },
            ...
          },
        }
    """
    viewer = mind.owner_power
    targets = [p for p in valid_powers if p != viewer]

    # Pre-bucket beliefs and predictions by about_power for one-pass lookup.
    beliefs_by_target: dict[str, list[BeliefNode]] = {t: [] for t in targets}
    for b in mind.beliefs.values():
        if b.about_power in beliefs_by_target:
            beliefs_by_target[b.about_power].append(b)

    preds_by_target: dict[str, list[PredictionNode]] = {t: [] for t in targets}
    for p in mind.predictions.values():
        if p.about_power in preds_by_target:
            preds_by_target[p.about_power].append(p)

    # Commitments: incoming = from THEM; self_commitments = to THEM (via target_power).
    incoming_by_speaker: dict[str, list[CommitmentNode]] = {t: [] for t in targets}
    for c in mind.incoming_commitments.values():
        if c.speaker in incoming_by_speaker:
            incoming_by_speaker[c.speaker].append(c)

    self_by_target: dict[str, list[SelfCommitmentNode]] = {t: [] for t in targets}
    for c in mind.self_commitments.values():
        # SelfCommitmentNode.target_power may be unset for broadcast/public
        # commitments — fall back to addressees in that case.
        candidates = [c.target_power] if c.target_power else list(c.addressees)
        for tgt in candidates:
            if tgt in self_by_target:
                self_by_target[tgt].append(c)

    intents_by_target: dict[str, list[StrategicIntentNode]] = {t: [] for t in targets}
    for i in mind.strategic_intents.values():
        for tgt in i.target_powers:
            if tgt in intents_by_target:
                intents_by_target[tgt].append(i)

    # Message events: split by direction. Sorted desc, capped per direction.
    msgs_from_target: dict[str, list[MessageEvent]] = {t: [] for t in targets}
    msgs_to_target: dict[str, list[MessageEvent]] = {t: [] for t in targets}
    for m in mind.message_events.values():
        if m.sender in msgs_from_target:
            msgs_from_target[m.sender].append(m)
        for r in m.recipients:
            if r in msgs_to_target:
                msgs_to_target[r].append(m)

    by_target = {}
    for t in targets:
        beliefs = beliefs_by_target[t]
        beliefs_sorted = sorted(beliefs, key=lambda b: -b.hp)

        # Group beliefs by type for renderer convenience.
        beliefs_by_type = {k: [] for k in _BELIEF_TYPE_KEYS}
        for b in beliefs_sorted:
            beliefs_by_type[b.belief_type.value].append(_belief_to_dict(b))

        # Singled-out credibility belief (the persistent "do they keep
        # promises" record). There can be more than one; take the most
        # recent active one if any, else the most recent of any status.
        cred = [b for b in beliefs if b.belief_type == BeliefType.CREDIBILITY]
        cred_active = [b for b in cred if b.status == BeliefStatus.ACTIVE]
        chosen = (cred_active or cred)
        chosen.sort(key=lambda b: b.last_updated_phase, reverse=True)
        credibility_belief = _belief_to_dict(chosen[0]) if chosen else None

        # Predictions, partitioned by status for rendering.
        preds = preds_by_target[t]
        preds_by_status = {
            "open": [], "confirmed": [], "refuted": [],
            "partial": [], "superseded": [],
        }
        for p in sorted(preds, key=lambda x: -x.confidence):
            bucket = p.status.value
            if bucket in preds_by_status:
                preds_by_status[bucket].append(_prediction_to_dict(p))
            else:
                preds_by_status.setdefault(bucket, []).append(_prediction_to_dict(p))

        # Commitment ledger.
        from_them = incoming_by_speaker[t]
        to_them = self_by_target[t]
        kept = sum(1 for c in from_them if c.status == CommitmentStatus.KEPT)
        broken = sum(1 for c in from_them if c.status == CommitmentStatus.BROKEN)
        pending = sum(1 for c in from_them if c.status == CommitmentStatus.PENDING)
        irrelevant = sum(
            1 for c in from_them
            if c.status in (CommitmentStatus.IRRELEVANT, CommitmentStatus.UNPARSEABLE)
        )

        # Recent messages, sorted desc.
        recent_from = sorted(msgs_from_target[t], key=lambda m: m.sent_at, reverse=True)
        recent_to = sorted(msgs_to_target[t], key=lambda m: m.sent_at, reverse=True)

        by_target[t] = {
            "trust": _trust_from_ledger(kept, broken),
            "ledger": {
                "kept": kept, "broken": broken,
                "pending": pending, "irrelevant": irrelevant,
            },
            "credibility_belief": credibility_belief,
            "beliefs": [_belief_to_dict(b) for b in beliefs_sorted],
            "beliefs_by_type": beliefs_by_type,
            "predictions": preds_by_status,
            "commitments_from_them": [_commitment_to_dict(c) for c in from_them],
            "commitments_to_them": [_self_commitment_to_dict(c) for c in to_them],
            "my_intents_targeting_them": [
                _intent_to_dict(i) for i in intents_by_target[t]
            ],
            "recent_messages_from_them": [
                _message_event_to_dict(m) for m in recent_from[:message_limit]
            ],
            "recent_messages_to_them": [
                _message_event_to_dict(m) for m in recent_to[:message_limit]
            ],
        }

    return {"viewer": viewer, "archetype": mind.archetype, "by_target": by_target}


# ============================================================================
# Lifecycle — per-phase telemetry from agent.phase_logs
# ============================================================================


def lifecycle_view(phase_logs: list) -> list[dict]:
    """Convert a DiplomacyAgentV2.phase_logs list into JSON-clean records.

    Each AgentLog has: phase, negotiate_calls, orders_call_succeeded,
    orders_retries, near_term_synthesized, commitments_graded,
    predictions_graded, beliefs_promoted, beliefs_retired, intents_promoted,
    intents_retired, revision_proposals_made, parse_notes.
    """
    out = []
    for log in phase_logs:
        out.append({
            "phase": log.phase,
            "negotiate_calls": log.negotiate_calls,
            "orders_call_succeeded": log.orders_call_succeeded,
            "orders_retries": log.orders_retries,
            "near_term_synthesized": log.near_term_synthesized,
            "commitments_graded": log.commitments_graded,
            "predictions_graded": log.predictions_graded,
            "beliefs_promoted": log.beliefs_promoted,
            "beliefs_retired": log.beliefs_retired,
            "intents_promoted": log.intents_promoted,
            "intents_retired": log.intents_retired,
            "revision_proposals_made": log.revision_proposals_made,
            "parse_notes": list(log.parse_notes),
        })
    return out


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    import time as _t
    import json as _json
    from diplomacy_kg_schema import (
        AgentMind, CharacterBrief, IdentityConstraintNode,
        BeliefNode, BeliefType, BeliefStatus,
        PredictionNode, PredictionStatus, PredictionWindowKind,
        CommitmentNode, SelfCommitmentNode, CommitmentType, CommitmentStatus,
        StrategicIntentNode, StrategicIntentStatus,
        MessageEvent, new_id,
    )

    print("=" * 72)
    print("INSPECTION SANITY CHECK")
    print("=" * 72)

    # Build a synthetic AgentMind with one record of each kind. This exercises
    # every helper without needing a real game.
    mind = AgentMind(owner_power="FRANCE", archetype="MARSHAL_VEIL")
    mind.character_brief = CharacterBrief(
        id=new_id("brief"), archetype="MARSHAL_VEIL",
        text="I am Marshal Veil. I plan in arcs.", generated_at=_t.time(),
    )

    # Identity constraint
    ic = IdentityConstraintNode(
        id=new_id("ident"), kind="trait", label="patient_planner",
        weight=1.0, note="seeded from MARSHAL_VEIL archetype",
        archetype="MARSHAL_VEIL",
    )
    mind.identity_constraints[ic.id] = ic

    # Two beliefs about RUSSIA — one credibility, one risk_assessment
    cred = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.CREDIBILITY,
        head="Russia keeps tactical promises but evades long-term ones.",
        body="Game-1: Spring promise on BLA kept. Fall ALLIANCE_FOR broken.",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1903-FALL-MOVES",
        evidence_for=["mv_1", "mv_4"], evidence_against=["mv_8"],
        hp=1.4, critic_score=0.7, success=0.3,
        times_foveated=12, times_inspected=2,
        status=BeliefStatus.ACTIVE, persists_across_games=True,
    )
    risk = BeliefNode(
        id=new_id("belief"), about_power="RUSSIA",
        belief_type=BeliefType.RISK_ASSESSMENT,
        head="Russia will pivot south against Turkey by F1903.",
        body="Galician build pattern + Black Sea posture suggest pivot.",
        formed_at_phase="1902-SPRING-MOVES", formed_in_game=1,
        last_updated_phase="1902-SPRING-MOVES",
        hp=1.0, critic_score=0.5,
        status=BeliefStatus.PROTO,
    )
    mind.beliefs[cred.id] = cred
    mind.beliefs[risk.id] = risk

    # A prediction about RUSSIA
    pred = PredictionNode(
        id=new_id("pred"), about_power="RUSSIA",
        formed_at_phase="1902-SPRING-MOVES",
        predicted_event_type="move_to", predicted_target="RUM",
        predicted_subject_power=None,
        prediction_window="1902-FALL-MOVES",
        window_kind=PredictionWindowKind.NEAR_TERM,
        confidence=0.7, source_belief_ids=[risk.id],
        status=PredictionStatus.OPEN, rationale="see risk belief",
    )
    mind.predictions[pred.id] = pred

    # An incoming commitment from RUSSIA, kept
    cin = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg_1",
        speaker="RUSSIA", addressees=["FRANCE"],
        type=CommitmentType.NOT_MOVE_TO,
        subject_unit=None, subject_province="GAL",
        target_province=None, counterparty=None,
        deadline_phase="1902-SPRING-MOVES",
        conditional_on=None, status=CommitmentStatus.KEPT,
        resolved_at_phase="1902-SPRING-MOVES",
        raw_commitspeak_line="not_move_to: GAL by 1902-SPRING-MOVES",
    )
    mind.incoming_commitments[cin.id] = cin
    # And one broken
    cbroken = CommitmentNode(
        id=new_id("cmt"), source_msg_id="msg_2",
        speaker="RUSSIA", addressees=["FRANCE"],
        type=CommitmentType.NON_AGGRESSION,
        subject_unit=None, subject_province=None,
        target_province=None, counterparty="FRANCE",
        deadline_phase="1903-FALL-MOVES",
        conditional_on=None, status=CommitmentStatus.BROKEN,
        resolved_at_phase="1903-FALL-MOVES",
        raw_commitspeak_line="non_aggression: with FRANCE through 1903",
    )
    mind.incoming_commitments[cbroken.id] = cbroken

    # A self-commitment to GERMANY
    cself = SelfCommitmentNode(
        id=new_id("cmt"), source_msg_id="msg_3",
        speaker="FRANCE", addressees=["GERMANY"],
        type=CommitmentType.NOT_MOVE_TO,
        subject_unit=None, subject_province="BUR",
        target_province=None, counterparty=None,
        deadline_phase="1902-FALL-MOVES",
        conditional_on=None, status=CommitmentStatus.PENDING,
        target_power="GERMANY",
        raw_commitspeak_line="not_move_to: BUR by 1902-FALL-MOVES",
    )
    mind.self_commitments[cself.id] = cself

    # An intent targeting RUSSIA
    intent = StrategicIntentNode(
        id=new_id("intent"),
        head="Block Russia's southern pivot via Italian alliance.",
        body="Detail: open a quiet south, pressure RUM via Italy/Austria.",
        formed_at_phase="1902-SPRING-MOVES",
        target_powers=["RUSSIA", "TURKEY"], target_provinces=["RUM", "BUL"],
        horizon="1904-FALL-MOVES",
        hp=1.1, critic_score=0.6, status=StrategicIntentStatus.ACTIVE,
        active_since_phase="1902-SPRING-MOVES",
    )
    mind.strategic_intents[intent.id] = intent

    # A message event from RUSSIA
    msg = MessageEvent(
        id=new_id("msg"), phase="1902-SPRING-MOVES",
        sender="RUSSIA", recipients=["FRANCE"], public=False,
        body="Mutual restraint in Galicia.\n[[commit\n  not_move_to: GAL by 1902-SPRING-MOVES\n]]",
        commitspeak_tail="[[commit\n  not_move_to: GAL by 1902-SPRING-MOVES\n]]",
        sent_at=_t.time(),
    )
    mind.message_events[msg.id] = msg

    # ----- Run the serializers -----

    full = mind_to_inspection_dict(mind)
    print(f"\n  full inspection dict — top-level keys: {sorted(full.keys())}")
    print(f"  counts: {full['counts']}")
    assert full["owner_power"] == "FRANCE"
    assert full["archetype"] == "MARSHAL_VEIL"
    assert len(full["beliefs"]) == 2
    assert len(full["predictions"]) == 1
    assert len(full["incoming_commitments"]) == 2
    assert len(full["self_commitments"]) == 1
    assert len(full["strategic_intents"]) == 1
    assert len(full["message_events"]) == 1

    # Round-trip through json to confirm cleanliness
    s = _json.dumps(full)
    print(f"  JSON-serializes cleanly: {len(s)} bytes")

    # Theory of mind view
    valid_powers = {"AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "RUSSIA", "TURKEY"}
    tom = theory_of_mind_view(mind, valid_powers)
    print(f"\n  ToM view — viewer: {tom['viewer']}, "
          f"targets: {sorted(tom['by_target'].keys())}")
    russia = tom["by_target"]["RUSSIA"]
    print(f"  RUSSIA card:")
    print(f"    trust={russia['trust']}  ledger={russia['ledger']}")
    print(f"    beliefs={len(russia['beliefs'])}, "
          f"credibility_belief={'present' if russia['credibility_belief'] else 'none'}")
    print(f"    predictions: open={len(russia['predictions']['open'])}")
    print(f"    promises from them: {len(russia['commitments_from_them'])}")
    print(f"    intents targeting them: {len(russia['my_intents_targeting_them'])}")
    print(f"    recent messages from them: {len(russia['recent_messages_from_them'])}")

    assert russia["trust"] == 0.5  # 1 kept, 1 broken
    assert russia["ledger"] == {"kept": 1, "broken": 1, "pending": 0, "irrelevant": 0}
    assert russia["credibility_belief"] is not None
    assert len(russia["beliefs"]) == 2
    assert len(russia["predictions"]["open"]) == 1
    assert len(russia["commitments_from_them"]) == 2
    assert len(russia["my_intents_targeting_them"]) == 1
    assert len(russia["recent_messages_from_them"]) == 1

    # GERMANY card should have the self-commitment, no beliefs
    germany = tom["by_target"]["GERMANY"]
    assert germany["trust"] is None
    assert len(germany["beliefs"]) == 0
    assert len(germany["commitments_to_them"]) == 1
    print(f"\n  GERMANY card:")
    print(f"    trust={germany['trust']}, promises_to_them="
          f"{len(germany['commitments_to_them'])}")

    # Round-trip ToM through json
    s = _json.dumps(tom)
    print(f"\n  ToM view JSON-serializes cleanly: {len(s)} bytes")

    print()
    print("Inspection sanity check passed.")
