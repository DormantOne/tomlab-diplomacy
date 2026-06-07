"""
diplomacy_agent_v2.py — the new agent.

Wraps an AgentMind with the same outward shape as the legacy LLMAgent
(negotiate, decide_orders, decide_retreats, decide_builds) so engine
integration stays minimal. Internally:

  - Every prompt is built via diplomacy_llm_protocol with a fovea-narrowed
    view of the mind state.
  - Outgoing messages are auto-parsed for commitspeak; the speaker's own
    self-commitments register immediately.
  - Incoming messages — passed in from the engine session — are parsed
    by the agent's intake_message() method, registering CommitmentNodes
    in the recipient's mind.
  - At phase boundaries, the engine calls absorb_phase_resolution() which
    captures MoveEvents/PhaseStates into the mind, runs the commitment
    grader, the prediction grader, the belief lifecycle, and the strategy
    lifecycle in dependency order.

The legacy llm_agent.py used six structured graphs (personality, soul,
ethics, theory_of_mind, strategy, counterfactuals) — this version uses
one AgentMind with five layers. Personality and ethics are seeded into
identity_constraints at construction time and consulted via the cached
character_brief; the dynamic content is in beliefs/predictions/intents/
commitments which all evolve from gameplay.

This file does NOT make network calls itself. The LLM call is injected
as a callable, allowing real Ollama, real Anthropic, or a deterministic
stub for testing.
"""

from __future__ import annotations

import time as _t
from dataclasses import dataclass, field
from typing import Optional, Callable

from diplomacy_kg_schema import (
    AgentMind, CharacterBrief, MessageEvent, MoveEvent, AdjustmentEvent,
    PhaseState, CommitmentStatus, BeliefStatus,
    PowerName, ProvinceCode, PhaseKey, new_id,
)
from diplomacy_commitspeak import (
    parse_message,
    commitspeak_lines_to_incoming_commitment_nodes,
)
from diplomacy_fovea import build_fovea, CallContext, mark_fovea_used
from diplomacy_graders import EventIndex, grade
from diplomacy_prediction_grader import grade_prediction
from diplomacy_belief_lifecycle import run_belief_lifecycle
from diplomacy_strategy_lifecycle import (
    run_strategy_lifecycle, start_intent_commitment,
)
from diplomacy_llm_protocol import (
    LLMCall, NegotiationOutput, OrdersOutput,
    negotiate, order_decision,
    compose_belief_revision_prompt, parse_belief_revision_response,
    compose_intent_revision_prompt, parse_intent_revision_response,
)


# ============================================================================
# Engine-state translation
# ============================================================================
# Engine uses singular "MOVEMENT/RETREAT/ADJUSTMENT" while the schema uses
# plural "MOVES/RETREATS/ADJUSTMENTS". This helper normalizes.

_PHASE_TRANSLATION = {
    "MOVEMENT": "MOVES",
    "RETREAT": "RETREATS",
    "ADJUSTMENT": "ADJUSTMENTS",
}


def schema_phase_key(year: int, season: str, phase: str) -> PhaseKey:
    """Convert engine (year, season, phase) → schema PhaseKey."""
    schema_phase = _PHASE_TRANSLATION.get(phase.upper(), phase.upper())
    return f"{year}-{season.upper()}-{schema_phase}"


# ============================================================================
# Agent
# ============================================================================

@dataclass
class AgentLog:
    """Per-phase telemetry for inspection — what happened during this phase
    inside the agent. Used by the session for the biopsy view."""
    phase: PhaseKey
    negotiate_calls: int = 0
    orders_call_succeeded: bool = False
    orders_retries: int = 0
    near_term_synthesized: bool = False
    commitments_graded: int = 0
    predictions_graded: int = 0
    beliefs_promoted: int = 0
    beliefs_retired: int = 0
    intents_promoted: int = 0
    intents_retired: int = 0
    revision_proposals_made: int = 0
    parse_notes: list[str] = field(default_factory=list)


class DiplomacyAgentV2:
    """The new agent.

    Outward interface matches the legacy LLMAgent for easy session swap:
      .power, .negotiate(state, recent), .decide_orders(state, recent),
      .decide_retreats(state), .decide_builds(state)
    """

    def __init__(
        self, *,
        power: PowerName,
        archetype: str,
        character_brief_text: str,
        llm_call: LLMCall,
        valid_powers: set[PowerName],
        valid_provinces: set[ProvinceCode],
    ):
        self.power = power
        self.archetype = archetype
        self.llm_call = llm_call
        self.valid_powers = set(valid_powers)
        self.valid_provinces = set(valid_provinces)

        self.mind = AgentMind(owner_power=power, archetype=archetype)
        self.mind.character_brief = CharacterBrief(
            id=new_id("brief"), archetype=archetype,
            text=character_brief_text, generated_at=_t.time(),
        )
        # Per-phase log for the most-recent N phases (biopsy)
        self.phase_logs: list[AgentLog] = []
        self._biopsy_max = 12

    # ---- engine-facing interface ----

    def negotiate(self, state, recent_messages: list,
                  *, addressees: Optional[list[PowerName]] = None
                  ) -> list[MessageEvent]:
        """Compose outgoing messages for one negotiation round.

        `state` is engine GameState; `recent_messages` is a list of
        Message-shaped objects (legacy or new). Returns list[MessageEvent]
        the session should distribute.
        """
        phase = schema_phase_key(state.year, state.season, state.phase)
        if addressees is None:
            addressees = [p for p in self.valid_powers
                          if p != self.power and p not in state.eliminated]

        log = self._current_log(phase)
        log.negotiate_calls += 1

        out: NegotiationOutput = negotiate(
            self.mind,
            addressees=addressees, phase=phase,
            board_summary_text=self._board_summary(state),
            recent_messages_text=self._recent_messages_text(recent_messages),
            valid_powers=self.valid_powers,
            valid_provinces=self.valid_provinces,
            llm_call=self.llm_call,
        )
        log.parse_notes.extend(out.parse_notes)
        return out.messages

    def decide_orders(self, state, recent_messages: list,
                      *, addressees: Optional[list[PowerName]] = None
                      ) -> tuple[list[str], OrdersOutput]:
        """Decide orders for the current phase.

        Returns (order_strings, OrdersOutput). Order strings are ready for
        the engine's parse_order; OrdersOutput is preserved for telemetry.
        """
        phase = schema_phase_key(state.year, state.season, state.phase)
        if addressees is None:
            addressees = [p for p in self.valid_powers
                          if p != self.power and p not in state.eliminated]

        from diplomacy_engine import units_by_power, ADJ, can_occupy
        my_units = units_by_power(state, self.power)
        own_unit_sigs = [f"{u.kind} {u.location}" for u in my_units]

        # Build per-unit options text
        unit_lines = []
        for u in my_units:
            key = "army" if u.kind == "A" else "fleet"
            neighbors = sorted(ADJ.get(u.location, {}).get(key, []))
            valid = [n for n in neighbors if can_occupy(u.kind, n)]
            if valid:
                unit_lines.append(
                    f"  {u.kind} {u.location}: HOLD or MOVE to one of: "
                    f"{', '.join(valid)}"
                )
            else:
                unit_lines.append(f"  {u.kind} {u.location}: must HOLD")
        unit_options_text = "\n".join(unit_lines) or "(no units)"

        log = self._current_log(phase)
        out: OrdersOutput = order_decision(
            self.mind,
            phase=phase, addressees=addressees,
            board_summary_text=self._board_summary(state),
            recent_messages_text=self._recent_messages_text(recent_messages),
            unit_options_text=unit_options_text,
            own_unit_signatures=own_unit_sigs,
            valid_powers=self.valid_powers,
            valid_provinces=self.valid_provinces,
            llm_call=self.llm_call,
            max_retries=1,
        )
        log.orders_call_succeeded = bool(out.accepted_orders)
        log.parse_notes.extend(out.parse_notes)
        log.near_term_synthesized = any(
            "synthesized default" in n for n in out.parse_notes
        )

        # Mark belief foveation (we know which beliefs the prompt fovea picked
        # because the fovea is built INSIDE order_decision; for now we skip
        # the mark step — a future pass can rebuild fovea here just to mark)
        order_strings = [ostr for _, ostr in out.accepted_orders]

        # If we have any units that didn't get an order, auto-hold
        ordered_origins = set()
        for unit_sig, _ in out.accepted_orders:
            tokens = unit_sig.split()
            if len(tokens) >= 2:
                ordered_origins.add(tokens[1])
        for u in my_units:
            if u.location not in ordered_origins:
                order_strings.append(f"{u.kind} {u.location} H")

        return order_strings, out

    def decide_retreats(self, state) -> list[str]:
        """Heuristic retreats — no LLM call. Same as legacy fallback."""
        from diplomacy_engine import ADJ, can_occupy
        my_dislodged = [u for u in state.dislodged if u.power == self.power]
        if not my_dislodged:
            return []
        orders: list[str] = []
        for u in my_dislodged:
            key = "army" if u.kind == "A" else "fleet"
            attacker_origin = state.dislodged_from.get(u.location)
            options = [n for n in ADJ.get(u.location, {}).get(key, [])
                       if n != attacker_origin
                       and can_occupy(u.kind, n)
                       and not any(x.location == n for x in state.units)]
            if options:
                orders.append(f"{u.kind} {u.location} R {options[0]}")
            else:
                orders.append(f"{u.kind} {u.location} D")
        return orders

    def decide_builds(self, state) -> list[str]:
        """Heuristic builds — no LLM call. Same as legacy fallback."""
        from diplomacy_engine import (
            HOME_CENTERS, supply_centers_owned, units_by_power, unit_at,
            PROVINCES,
        )
        scs = supply_centers_owned(state, self.power)
        units = units_by_power(state, self.power)
        delta = len(scs) - len(units)
        if delta == 0:
            return []
        orders: list[str] = []
        if delta > 0:
            available = [c for c in HOME_CENTERS[self.power]
                         if state.sc_owner.get(c) == self.power
                         and not unit_at(state, c)]
            for loc in available[:delta]:
                if self.power == "ENGLAND":
                    kind = "F"
                elif self.power == "RUSSIA" and loc in ("STP", "SEV"):
                    kind = "F"
                else:
                    kind = "A" if PROVINCES[loc][0] != "sea" else "F"
                orders.append(f"BUILD {kind} {loc}")
        else:
            home = set(HOME_CENTERS[self.power])
            sorted_units = sorted(units, key=lambda u: 0 if u.location in home else 1)
            for u in sorted_units[:-delta]:
                orders.append(f"DISBAND {u.kind} {u.location}")
        return orders

    # ---- intake / lifecycle ----

    def intake_message(self, message: MessageEvent) -> int:
        """Called by the session for every message addressed to us
        (or public). Parses commitspeak from the message body and
        registers CommitmentNodes in our mind. Returns count of new
        well-formed commitments registered."""
        if message.sender == self.power:
            return 0
        if not (message.public or self.power in message.recipients):
            return 0

        # Always store the message event (even if no commitspeak)
        self.mind.message_events[message.id] = message

        if not message.commitspeak_tail:
            return 0

        block = parse_message(
            message.body,
            provinces=self.valid_provinces,
            powers=self.valid_powers,
        )
        if block is None:
            return 0

        nodes, _malformed = commitspeak_lines_to_incoming_commitment_nodes(
            block, source_msg_id=message.id,
            speaker=message.sender,
            addressees=message.recipients or [self.power],
        )
        for n in nodes:
            self.mind.incoming_commitments[n.id] = n
        return len(nodes)

    def absorb_phase_resolution(
        self,
        *,
        resolved_phase: PhaseKey,
        move_events: list[MoveEvent],
        adjustment_events: list[AdjustmentEvent],
        phase_state: PhaseState,
        next_phase: PhaseKey,
    ) -> AgentLog:
        """Called by the session after each phase resolves.

        Stores events into the mind, runs commitment grading, prediction
        grading, belief lifecycle, and strategy lifecycle. Returns the
        per-phase telemetry log.
        """
        # Capture events
        for m in move_events:
            self.mind.move_events[m.id] = m
        for a in adjustment_events:
            self.mind.adjustment_events[a.id] = a
        self.mind.phase_states[phase_state.id] = phase_state

        log = self._current_log(resolved_phase)

        # Build event index from EVERYTHING in mind (cheap; events accumulate)
        index = EventIndex()
        for m in self.mind.move_events.values():
            index.add_move(m)
        for a in self.mind.adjustment_events.values():
            index.add_adjustment(a)
        for ps in self.mind.phase_states.values():
            index.add_phase_state(ps)

        # Grade pending commitments whose deadlines have passed
        from diplomacy_fovea import _phase_index
        resolved_idx = _phase_index(resolved_phase)
        all_resolved_phase_keys = sorted(
            self.mind.phase_states.keys(),
            key=lambda pid: _phase_index(self.mind.phase_states[pid].phase),
        )
        all_resolved_phases = [
            self.mind.phase_states[pid].phase for pid in all_resolved_phase_keys
        ]

        for c in list(self.mind.incoming_commitments.values()):
            if c.status != CommitmentStatus.PENDING:
                continue
            is_multi = c.type.value in ("non_aggression", "alliance_for", "demilitarize")
            deadline_resolved = _phase_index(c.deadline_phase) <= resolved_idx
            if not (deadline_resolved or is_multi):
                continue
            window_phases = (all_resolved_phases if is_multi else None)
            status, evidence = grade(c, index, phases_in_window=window_phases)
            # Multi-phase rule: only finalize if deadline reached OR violation detected.
            # Otherwise keep PENDING — a "no violation yet" snapshot is not the same
            # as "kept across the full window."
            if is_multi and not deadline_resolved and status == CommitmentStatus.KEPT:
                # Don't lock in KEPT mid-window; leave it PENDING and re-check next phase
                continue
            c.status = status
            c.resolved_at_phase = resolved_phase
            c.grading_evidence = evidence
            log.commitments_graded += 1

        # Same for self-commitments
        for c in list(self.mind.self_commitments.values()):
            if c.status != CommitmentStatus.PENDING:
                continue
            is_multi = c.type.value in ("non_aggression", "alliance_for", "demilitarize")
            deadline_resolved = _phase_index(c.deadline_phase) <= resolved_idx
            if not (deadline_resolved or is_multi):
                continue
            window_phases = (all_resolved_phases if is_multi else None)
            status, evidence = grade(c, index, phases_in_window=window_phases)
            if is_multi and not deadline_resolved and status == CommitmentStatus.KEPT:
                # Same rule for self-commitments
                continue
            c.status = status
            c.resolved_at_phase = resolved_phase
            c.grading_evidence = evidence

        # Grade open predictions
        for p in self.mind.predictions.values():
            from diplomacy_kg_schema import PredictionStatus
            if p.status != PredictionStatus.OPEN:
                continue
            status, evidence = grade_prediction(
                p, index,
                all_resolved_phases=all_resolved_phases,
                current_phase=resolved_phase,
            )
            if status != PredictionStatus.OPEN:
                p.status = status
                p.resolved_at_phase = resolved_phase
                p.grading_evidence = evidence
                # Update parent belief evidence lists
                for bid in p.source_belief_ids:
                    belief = self.mind.beliefs.get(bid)
                    if belief is None:
                        continue
                    if status == PredictionStatus.CONFIRMED:
                        belief.evidence_for.append(p.id)
                    elif status == PredictionStatus.REFUTED:
                        belief.evidence_against.append(p.id)
                log.predictions_graded += 1

        # Belief lifecycle
        b_summary = run_belief_lifecycle(
            self.mind, current_phase=next_phase,
            foveated_belief_ids_this_phase=set(),  # could track this in the future
        )
        log.beliefs_promoted = len(b_summary["promoted"])
        log.beliefs_retired = (len(b_summary["retired"]) + len(b_summary["archived"]))

        # Strategy lifecycle (plans get reviewed against this phase's events)
        s_summary = run_strategy_lifecycle(
            self.mind, current_phase=next_phase,
            just_resolved_phase=resolved_phase,
        )
        log.intents_promoted = len(s_summary["promoted"])
        log.intents_retired = len(s_summary["retired"])

        # Optional: invite revisions for failing items (1 per phase max each
        # to bound LLM cost). The actual proposal is a separate LLM call.
        revision_belief_candidates = b_summary.get("revision_candidates", [])
        revision_intent_candidates = s_summary.get("revision_candidates", [])

        if revision_belief_candidates:
            self._invite_belief_revision(
                revision_belief_candidates[0], next_phase, log,
            )
        if revision_intent_candidates:
            self._invite_intent_revision(
                revision_intent_candidates[0], next_phase, log,
            )

        # Auto-open commitments for newly-promoted intents
        for promoted_intent in s_summary["promoted"]:
            existing = [
                ic for ic in self.mind.intent_commitments.values()
                if ic.intent_id == promoted_intent.id and ic.status == "active"
            ]
            if not existing:
                start_intent_commitment(
                    self.mind, promoted_intent.id,
                    started_at_phase=next_phase,
                    window_phases=3,
                )

        # Mirror this phase's typed state into the substrate's multi-channel
        # router (decaying, persisted on the mind). Best-effort; never fatal.
        try:
            from diplomacy_kg_analysis import sync_router_from_mind
            sync_router_from_mind(self.mind)
        except Exception as _e:
            print(f"[kg sync hook] non-fatal: {_e}")

        return log

    def _invite_belief_revision(self, belief, phase, log):
        """One-shot LLM call to propose a narrower belief successor."""
        prompt = compose_belief_revision_prompt(
            self.mind, belief, phase=phase,
        )
        raw = self.llm_call(prompt)
        from diplomacy_kg_schema import PredictionStatus
        failing_pred_ids = [
            p.id for p in self.mind.predictions.values()
            if belief.id in p.source_belief_ids
            and p.status == PredictionStatus.REFUTED
        ]
        rev = parse_belief_revision_response(
            raw, parent_belief=belief, phase=phase,
            failing_pred_ids=failing_pred_ids,
        )
        if rev.proposal is not None:
            self.mind.belief_revisions[rev.proposal.id] = rev.proposal
            log.revision_proposals_made += 1
        else:
            log.parse_notes.extend(rev.parse_notes)

    def _invite_intent_revision(self, intent, phase, log):
        """One-shot LLM call to propose a narrower intent successor."""
        from diplomacy_kg_schema import PredictionStatus
        triggering = [
            p.id for p in self.mind.predictions.values()
            if p.parent_intent_id == intent.id
            and p.status == PredictionStatus.REFUTED
        ]
        prompt = compose_intent_revision_prompt(
            self.mind, intent, phase=phase,
        )
        raw = self.llm_call(prompt)
        rev = parse_intent_revision_response(
            raw, parent_intent=intent, phase=phase,
            triggering_evidence_ids=triggering,
        )
        if rev.proposal is not None:
            self.mind.intent_revisions[rev.proposal.id] = rev.proposal
            log.revision_proposals_made += 1
        else:
            log.parse_notes.extend(rev.parse_notes)

    # ---- helpers ----

    def _current_log(self, phase: PhaseKey) -> AgentLog:
        if self.phase_logs and self.phase_logs[-1].phase == phase:
            return self.phase_logs[-1]
        log = AgentLog(phase=phase)
        self.phase_logs.append(log)
        while len(self.phase_logs) > self._biopsy_max:
            self.phase_logs.pop(0)
        return log

    def _board_summary(self, state) -> str:
        from diplomacy_engine import units_by_power, supply_centers_owned, POWERS
        bits = []
        for power in POWERS:
            if power in state.eliminated:
                continue
            scs = supply_centers_owned(state, power)
            units = units_by_power(state, power)
            unit_str = "+".join(f"{u.kind}{u.location}" for u in units) or "-"
            bits.append(f"{power[:3]}:{len(scs)},{unit_str}")
        return "|".join(bits)

    def _recent_messages_text(self, recent: list) -> str:
        """Render the last ~6 messages (full) plus a count summary of older.

        Accepts either MessageEvent objects (new) or legacy Message objects
        with sender/recipients/text/public attrs. Both shapes work.
        """
        if not recent:
            return "Recent messages: (none)"
        last6 = recent[-6:]
        older = recent[:-6] if len(recent) > 6 else []
        lines = ["Recent messages:"]
        if older:
            counts = {}
            for m in older:
                counts[m.sender] = counts.get(m.sender, 0) + 1
            counts_str = ", ".join(f"{s}={n}" for s, n in counts.items())
            lines.append(f"  (older: {len(older)} msgs — {counts_str})")
        for m in last6:
            target = "ALL" if getattr(m, "public", False) else (
                ",".join(getattr(m, "recipients", []) or []) or "?"
            )
            text = getattr(m, "body", None) or getattr(m, "text", "")
            text = text.replace("\n", " ").strip()
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"  {m.sender} -> {target}: {text}")
        return "\n".join(lines)


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("AGENT V2 SANITY CHECK")
    print("=" * 72)

    # A deterministic stub LLM that returns valid JSON for orders.
    def stub_llm(prompt: str) -> str:
        if "orders" in prompt.lower() and "near_term" in prompt:
            return '''{
  "orders": ["A PAR - BUR", "F BRE H", "A MAR - SPA"],
  "plan": {"head": "Develop south.", "body": "Move A PAR to BUR; F BRE holds; A MAR -> SPA.",
           "parent_intent_id": null},
  "predictions": [
    {"about": "GERMANY", "type": "non_action", "target": "BEL", "window": "near_term"}
  ]
}'''
        if "compose 0-3" in prompt.lower() or "messages" in prompt.lower():
            return '''{
  "messages": [
    {"to": ["GERMANY"], "public": false,
     "text": "I propose mutual restraint.\\n[[commit\\n  not_move_to: BEL by 1902-SPRING-MOVES\\n]]"}
  ]
}'''
        return '{}'

    PROVINCES = {"PAR", "MAR", "BRE", "BUR", "SPA", "MUN", "BEL"}
    POWERS = {"AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "RUSSIA", "TURKEY"}

    agent = DiplomacyAgentV2(
        power="FRANCE", archetype="MARSHAL_VEIL",
        character_brief_text="I am Marshal Veil. Measured and deliberate.",
        llm_call=stub_llm,
        valid_powers=POWERS, valid_provinces=PROVINCES,
    )
    print(f"  Agent built: power={agent.power}, archetype={agent.archetype}")
    print(f"  Mind has character_brief: {agent.mind.character_brief is not None}")
    print(f"  Beliefs: {len(agent.mind.beliefs)}, Predictions: {len(agent.mind.predictions)}")

    # Test intake_message
    msg = MessageEvent(
        id=new_id("msg"), phase="1902-SPRING-MOVES",
        sender="GERMANY", recipients=["FRANCE"], public=False,
        body=("Truce in Belgium.\n\n[[commit\n"
              "  not_move_to: BEL by 1902-SPRING-MOVES\n]]"),
        commitspeak_tail="[[commit\n  not_move_to: BEL by 1902-SPRING-MOVES\n]]",
        sent_at=_t.time(),
    )
    n = agent.intake_message(msg)
    print(f"  intake_message: {n} new commitment(s)")
    print(f"  Mind incoming_commitments: {len(agent.mind.incoming_commitments)}")
    assert n == 1
    assert len(agent.mind.incoming_commitments) == 1

    print()
    print("Agent V2 sanity check passed.")
