"""
diplomacy_mute.py — magic_kg-style muted-control ablation for the substrate.

Phase 1.5 deliverable 1 of 2. Adds a per-agent `kg_advisory_mute` switch
that suppresses KG READS during LLM calls while leaving WRITES intact —
the exact pattern used in magic_kg's experimental harness for measuring
the effect of KG advisory on play quality.

How muting works
----------------
A muted agent runs the same LLM call cadence as a normal one (negotiate
each round; decide_orders each phase). The difference is in the prompt
context fed to those calls:

  - NORMAL: prompt is built from the agent's full AgentMind — beliefs,
    predictions, intents, commitments, identity. The LLM "sees" everything
    the agent has learned.

  - MUTED: prompt is built from a stripped-down AgentMind containing ONLY
    the identity layer (character_brief, identity_constraints,
    games_played). All in-game inferential content is hidden from the LLM
    call. The agent acts on the basis of immediate context — board state
    and the tail of the recent message log — not on accumulated memory.

Crucially, WRITES still happen. The protocol's negotiate() and
order_decision() write self_commitments / message_events / proto_beliefs /
proto_intents / predictions / plan_nodes to whatever mind they're given.
We let those writes land in a temporary muted mind, then transfer them
back to the real mind so the substrate keeps growing. The diff between
muted and full play is therefore precisely the effect of KG advisory on
each individual move.

Following magic_kg's design (see magic_kg/app.py lines 36-48,
Patch 5 + Patch 7).

Public surface
--------------
  MutableAgent(DiplomacyAgentV2)       — agent class with mute switch
  make_muted_mind(real_mind)            — empty AgentMind preserving identity
  transfer_writes(src, dst, *, kinds)   — forward new records src → dst
  make_mutable_agents(...)               — factory parallel to run_v2.make_agents
  set_mute(agents, muted_powers)         — update mute flags in place
  per_agent_metrics(agents, state)       — extract eval metrics per agent

This module is intentionally additive: nothing in diplomacy_agent_v2.py
or diplomacy_llm_protocol.py changes. The MutableAgent subclass overrides
two methods; everything else inherits unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from diplomacy_kg_schema import AgentMind
from diplomacy_agent_v2 import DiplomacyAgentV2, schema_phase_key
from diplomacy_llm_protocol import (
    negotiate as _protocol_negotiate,
    order_decision as _protocol_order_decision,
)


# ============================================================================
# Mind-level helpers
# ============================================================================


def make_muted_mind(real_mind: AgentMind) -> AgentMind:
    """Build a fresh AgentMind preserving ONLY the identity layer.

    Result has:
      - same owner_power, archetype, games_played
      - same character_brief reference
      - copy of identity_constraints (shallow, dict-of-references)

    All in-game state is empty: beliefs, predictions, commitments, intents,
    plans, message_events, move_events, phase_states, adjustment_events,
    intent_commitments, belief_revisions, intent_revisions.

    The shallow copy of identity_constraints is intentional — those are
    immutable seeded nodes; sharing the references is safe and avoids
    deep-copy cost. The same is true of character_brief.
    """
    muted = AgentMind(
        owner_power=real_mind.owner_power,
        archetype=real_mind.archetype,
    )
    muted.character_brief = real_mind.character_brief
    muted.identity_constraints = dict(real_mind.identity_constraints)
    muted.games_played = real_mind.games_played
    return muted


def transfer_writes(
    src: AgentMind, dst: AgentMind, *, kinds: Iterable[str],
) -> dict[str, int]:
    """Forward records added to `src` over to `dst`, by attribute name.

    Each name in `kinds` must be a dict-typed AgentMind attribute (e.g.
    "self_commitments", "beliefs", "predictions"). Records whose id is
    already present in dst are skipped (idempotent).

    Returns a count of records transferred per kind.
    """
    moved = {}
    for kind in kinds:
        src_dict = getattr(src, kind)
        dst_dict = getattr(dst, kind)
        n = 0
        for k, v in src_dict.items():
            if k not in dst_dict:
                dst_dict[k] = v
                n += 1
        moved[kind] = n
    return moved


# Writes the protocol makes during negotiate (line 323, 335 of llm_protocol.py)
_NEGOTIATE_WRITE_KINDS = ("message_events", "self_commitments")

# Writes the protocol makes during order_decision (lines 814, 819, 839, 846)
_ORDERS_WRITE_KINDS = ("beliefs", "strategic_intents", "predictions", "plan_nodes")


# ============================================================================
# MutableAgent
# ============================================================================


class MutableAgent(DiplomacyAgentV2):
    """DiplomacyAgentV2 with a per-instance kg_advisory_mute switch.

    When kg_advisory_mute is False (default), behavior is identical to
    DiplomacyAgentV2.

    When True, negotiate() and decide_orders() build their LLM prompts
    against an empty mind (identity-only). Writes from those calls are
    transferred to the real mind so the KG continues to grow.

    intake_message() and absorb_phase_resolution() are NOT muted — those
    are the pure-write paths and they always run on the real mind.
    decide_retreats() / decide_builds() are heuristic (no LLM) and need
    no muting.
    """

    def __init__(
        self, *args, kg_advisory_mute: bool = False, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kg_advisory_mute = bool(kg_advisory_mute)

    # ---- intercepted methods ----

    def negotiate(self, state, recent_messages, *, addressees=None):
        if not self.kg_advisory_mute:
            return super().negotiate(
                state, recent_messages, addressees=addressees,
            )
        return self._muted_negotiate(state, recent_messages, addressees)

    def decide_orders(self, state, recent_messages, *, addressees=None):
        if not self.kg_advisory_mute:
            return super().decide_orders(
                state, recent_messages, addressees=addressees,
            )
        return self._muted_decide_orders(state, recent_messages, addressees)

    # ---- muted implementations ----

    def _muted_negotiate(self, state, recent_messages, addressees):
        phase = schema_phase_key(state.year, state.season, state.phase)
        if addressees is None:
            addressees = [
                p for p in self.valid_powers
                if p != self.power and p not in state.eliminated
            ]
        log = self._current_log(phase)
        log.negotiate_calls += 1

        muted_mind = make_muted_mind(self.mind)
        out = _protocol_negotiate(
            muted_mind,
            addressees=addressees, phase=phase,
            board_summary_text=self._board_summary(state),
            recent_messages_text=self._recent_messages_text(recent_messages),
            valid_powers=self.valid_powers,
            valid_provinces=self.valid_provinces,
            llm_call=self.llm_call,
        )
        log.parse_notes.extend(out.parse_notes)
        log.parse_notes.append("kg_advisory_muted")

        # Forward protocol-side writes back to the real mind.
        transfer_writes(muted_mind, self.mind, kinds=_NEGOTIATE_WRITE_KINDS)
        return out.messages

    def _muted_decide_orders(self, state, recent_messages, addressees):
        # Build the unit_options_text the same way the parent class does —
        # this is engine-derived, not mind-derived, so it's identical
        # whether muted or not.
        phase = schema_phase_key(state.year, state.season, state.phase)
        if addressees is None:
            addressees = [
                p for p in self.valid_powers
                if p != self.power and p not in state.eliminated
            ]

        from diplomacy_engine import units_by_power, ADJ, can_occupy
        my_units = units_by_power(state, self.power)
        own_unit_sigs = [f"{u.kind} {u.location}" for u in my_units]

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

        muted_mind = make_muted_mind(self.mind)
        out = _protocol_order_decision(
            muted_mind,
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
        log.parse_notes.append("kg_advisory_muted")
        log.near_term_synthesized = any(
            "synthesized default" in n for n in out.parse_notes
        )

        order_strings = [ostr for _, ostr in out.accepted_orders]
        ordered_origins = set()
        for unit_sig, _ in out.accepted_orders:
            tokens = unit_sig.split()
            if len(tokens) >= 2:
                ordered_origins.add(tokens[1])
        for u in my_units:
            if u.location not in ordered_origins:
                order_strings.append(f"{u.kind} {u.location} H")

        # Forward protocol-side writes back to the real mind.
        transfer_writes(muted_mind, self.mind, kinds=_ORDERS_WRITE_KINDS)
        return order_strings, out


# ============================================================================
# Construction helpers
# ============================================================================


def make_mutable_agents(
    *,
    llm_call: Callable[[str], str],
    archetype_assignment: dict[str, str],
    briefs: Optional[dict[str, str]] = None,
    muted_powers: Optional[Iterable[str]] = None,
    valid_powers: Optional[set[str]] = None,
    valid_provinces: Optional[set[str]] = None,
) -> dict[str, MutableAgent]:
    """Build a dict of MutableAgent — parallel to run_v2.make_agents.

    `muted_powers` is the initial mute set; can be changed later via
    set_mute(). `briefs` defaults to run_v2.DEFAULT_BRIEFS.
    """
    if valid_powers is None:
        from diplomacy_engine import POWERS
        valid_powers = set(POWERS)
    if valid_provinces is None:
        from diplomacy_engine import PROVINCES
        valid_provinces = set(PROVINCES.keys())
    if briefs is None:
        try:
            from run_v2 import DEFAULT_BRIEFS as _BRIEFS
        except Exception:
            _BRIEFS = {"PLAYER_DEFAULT": "An LLM-driven Diplomacy player."}
        briefs = _BRIEFS

    muted_set = set(muted_powers or [])
    agents: dict[str, MutableAgent] = {}
    for power, archetype in archetype_assignment.items():
        brief_text = briefs.get(archetype, briefs.get("PLAYER_DEFAULT", ""))
        agents[power] = MutableAgent(
            power=power,
            archetype=archetype,
            character_brief_text=brief_text,
            llm_call=llm_call,
            valid_powers=valid_powers,
            valid_provinces=valid_provinces,
            kg_advisory_mute=power in muted_set,
        )
    return agents


def set_mute(
    agents: dict[str, Any], muted_powers: Iterable[str],
) -> dict[str, bool]:
    """Apply mute to a subset of agents in place. Returns the new mute map.

    Agents that are not MutableAgent are left alone (and reported in the
    return dict as None) — useful when only some powers in a session are
    using the mutable variant.
    """
    muted_set = set(muted_powers or [])
    out = {}
    for power, agent in agents.items():
        if isinstance(agent, MutableAgent):
            agent.kg_advisory_mute = power in muted_set
            out[power] = agent.kg_advisory_mute
        else:
            out[power] = None
    return out


# ============================================================================
# Per-agent metrics (for the eval harness)
# ============================================================================


def per_agent_metrics(agents: dict[str, Any], state) -> dict[str, dict]:
    """Extract eval metrics from each agent's mind plus current engine state.

    Returns dict[power → metrics dict]. Per-power metrics:
      - sc_count          int       supply centers owned
      - eliminated        bool
      - kg_advisory_mute  bool      (False for non-MutableAgent)
      - archetype         str
      - kept_rate         float|None  self-commitments kept / resolved
      - confirm_rate      float|None  predictions confirmed / resolved
      - beliefs           int
      - predictions       int
      - strategic_intents int
      - self_commitments  int       all self-commitments
      - self_resolved     int       resolved (kept + broken)
    """
    from diplomacy_engine import supply_centers_owned

    metrics: dict[str, dict] = {}
    eliminated_set = set(getattr(state, "eliminated", set()))
    for power, agent in agents.items():
        if not hasattr(agent, "mind"):
            continue
        mind = agent.mind
        try:
            scs = supply_centers_owned(state, power)
            sc_count = len(scs)
        except Exception:
            sc_count = 0

        # Kept rate from self-commitments
        self_commits = list(mind.self_commitments.values())
        resolved = [
            c for c in self_commits
            if c.status.value in ("kept", "broken")
        ]
        kept = sum(1 for c in resolved if c.status.value == "kept")
        kept_rate = (kept / len(resolved)) if resolved else None

        # Confirm rate from predictions
        preds = list(mind.predictions.values())
        resolved_preds = [
            p for p in preds
            if p.status.value in ("confirmed", "refuted")
        ]
        confirmed = sum(
            1 for p in resolved_preds if p.status.value == "confirmed"
        )
        confirm_rate = (
            (confirmed / len(resolved_preds)) if resolved_preds else None
        )

        metrics[power] = {
            "sc_count": sc_count,
            "eliminated": power in eliminated_set,
            "kg_advisory_mute": getattr(agent, "kg_advisory_mute", False),
            "archetype": getattr(agent, "archetype", None),
            "kept_rate": (
                round(kept_rate, 4) if kept_rate is not None else None
            ),
            "confirm_rate": (
                round(confirm_rate, 4) if confirm_rate is not None else None
            ),
            "self_commitments": len(self_commits),
            "self_resolved": len(resolved),
            "self_kept": kept,
            "predictions": len(preds),
            "predictions_resolved": len(resolved_preds),
            "predictions_confirmed": confirmed,
            "beliefs": len(mind.beliefs),
            "strategic_intents": len(mind.strategic_intents),
        }
    return metrics


# ============================================================================
# Sanity check
# ============================================================================
# Verifies (a) muted agent's mind stays empty during a call but accumulates
# writes, (b) unmuted agent behaves identically to the parent class.

if __name__ == "__main__":
    import json as _json
    import time as _t
    from diplomacy_kg_schema import (
        AgentMind, CharacterBrief, IdentityConstraintNode,
        BeliefNode, BeliefType, BeliefStatus,
        MessageEvent, new_id,
    )

    print("=" * 72)
    print("DIPLOMACY MUTE SANITY CHECK")
    print("=" * 72)

    # --- helper: deterministic stub LLM ---
    # Mirrors run_v2's StubLLM but minimal — enough to drive negotiate/orders
    # through to the write paths.
    def _stub(prompt: str) -> str:
        if "ORDER FORMS" in prompt or "near_term" in prompt.lower():
            return _json.dumps({
                "orders": ["A PAR H"],
                "plan": {"head": "Stub plan.", "body": "Stub", "parent_intent_id": None},
                "predictions": [{
                    "about": "GERMANY", "type": "non_action",
                    "target": "BEL", "window": "near_term",
                }],
            })
        if "Compose 0-3" in prompt or ("messages" in prompt.lower()
                                       and "outgoing" in prompt.lower()):
            return _json.dumps({
                "messages": [{
                    "to": ["GERMANY"], "public": False,
                    "text": ("Routine.\n\n[[commit\n"
                             "  not_move_to: BEL by 1902-SPRING-MOVES\n]]"),
                }],
            })
        return "{}"

    POWERS = {"AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "RUSSIA", "TURKEY"}
    PROVINCES = {"PAR", "MAR", "BRE", "BUR", "BEL", "MUN", "GAL", "WAR"}

    # --- Test 1: muted agent reports mute flag, transfers writes ---
    agent = MutableAgent(
        power="FRANCE", archetype="MARSHAL_VEIL",
        character_brief_text="I am Marshal Veil.",
        llm_call=_stub,
        valid_powers=POWERS, valid_provinces=PROVINCES,
        kg_advisory_mute=True,
    )

    # Plant a strong existing belief in the real mind. The MUTED call must
    # NOT see it (the protocol can't read it from the muted mind), but the
    # write back must NOT delete it either.
    seeded_belief = BeliefNode(
        id=new_id("belief"), about_power="GERMANY",
        belief_type=BeliefType.RISK_ASSESSMENT,
        head="Germany will attack BEL by F1902.",
        body="(elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1901-FALL-MOVES",
        status=BeliefStatus.ACTIVE,
    )
    agent.mind.beliefs[seeded_belief.id] = seeded_belief
    pre_belief_count = len(agent.mind.beliefs)
    pre_msg_count = len(agent.mind.message_events)
    pre_self_cmt = len(agent.mind.self_commitments)

    # Build a fake state minimally sufficient for the agent's helpers
    class _FakeUnit:
        def __init__(self, kind, location, power):
            self.kind = kind; self.location = location; self.power = power
    class _FakeState:
        year = 1902; season = "SPRING"; phase = "MOVEMENT"
        units = [_FakeUnit("A", "PAR", "FRANCE")]
        sc_owner = {}
        eliminated = set()
        dislodged = []
        dislodged_from = {}
    state = _FakeState()

    # negotiate (muted) — should produce one message + a self-commitment, both
    # transferred from the muted mind to the real mind
    msgs = agent.negotiate(state, recent_messages=[])
    print(f"\n  muted negotiate: produced {len(msgs)} message(s)")
    print(f"  real-mind beliefs after muted negotiate: "
          f"{len(agent.mind.beliefs)} (was {pre_belief_count}; should be unchanged)")
    print(f"  real-mind self_commitments after: "
          f"{len(agent.mind.self_commitments)} (was {pre_self_cmt}; should be +1)")
    print(f"  real-mind message_events after: "
          f"{len(agent.mind.message_events)} (was {pre_msg_count}; should be +1)")
    assert len(agent.mind.beliefs) == pre_belief_count, "muted negotiate altered beliefs"
    assert len(agent.mind.self_commitments) == pre_self_cmt + 1, \
        "muted negotiate did not transfer self_commitment"
    # Confirm the muted call was logged
    assert any("muted" in n for n in agent.phase_logs[-1].parse_notes), \
        "muted-call log marker not found"

    # decide_orders (muted) — should produce orders + transfer prediction/plan
    pre_pred = len(agent.mind.predictions)
    pre_plan = len(agent.mind.plan_nodes)
    order_strs, out = agent.decide_orders(state, recent_messages=[])
    print(f"\n  muted decide_orders: {len(order_strs)} order(s); "
          f"out has plan={out.plan is not None}, predictions={len(out.predictions)}")
    print(f"  real-mind predictions after muted orders: "
          f"{len(agent.mind.predictions)} (was {pre_pred}; should be +1)")
    print(f"  real-mind plan_nodes after: "
          f"{len(agent.mind.plan_nodes)} (was {pre_plan}; should be +1)")
    assert len(agent.mind.predictions) == pre_pred + 1
    assert len(agent.mind.plan_nodes) == pre_plan + 1
    assert len(agent.mind.beliefs) == pre_belief_count, \
        "muted orders altered seeded beliefs"

    # --- Test 2: unmuted agent runs normally ---
    agent_full = MutableAgent(
        power="FRANCE", archetype="MARSHAL_VEIL",
        character_brief_text="I am Marshal Veil.",
        llm_call=_stub,
        valid_powers=POWERS, valid_provinces=PROVINCES,
        kg_advisory_mute=False,
    )
    msgs_full = agent_full.negotiate(state, recent_messages=[])
    assert len(msgs_full) >= 1, "unmuted agent should still negotiate"
    print(f"\n  unmuted negotiate works: {len(msgs_full)} message(s)")
    # No muted marker on this one
    assert not any("muted" in n for n in agent_full.phase_logs[-1].parse_notes)

    # --- Test 3: set_mute toggles ---
    agents = {
        "FRANCE": agent,
        "GERMANY": agent_full,
    }
    result = set_mute(agents, ["GERMANY"])
    print(f"\n  set_mute({{'GERMANY'}}) → {result}")
    assert agents["FRANCE"].kg_advisory_mute is False
    assert agents["GERMANY"].kg_advisory_mute is True

    # --- Test 4: per_agent_metrics shape ---
    metrics = per_agent_metrics(agents, state)
    print(f"\n  per_agent_metrics keys: {sorted(metrics.keys())}")
    assert "FRANCE" in metrics
    france_m = metrics["FRANCE"]
    print(f"  FRANCE metrics: sc={france_m['sc_count']}, "
          f"muted={france_m['kg_advisory_mute']}, "
          f"beliefs={france_m['beliefs']}, "
          f"self_commitments={france_m['self_commitments']}, "
          f"predictions={france_m['predictions']}")
    assert france_m["beliefs"] == 1, "seeded belief should still be present"
    assert france_m["self_commitments"] == 1
    assert france_m["predictions"] == 1
    assert france_m["kg_advisory_mute"] is False  # we toggled it off
    # GERMANY had no seeded mind state; metrics still emit
    assert metrics["GERMANY"]["kg_advisory_mute"] is True

    # --- Test 5: factory + briefs default ---
    factory_agents = make_mutable_agents(
        llm_call=_stub,
        archetype_assignment={"FRANCE": "MARSHAL_VEIL", "GERMANY": "BARON_KORVIN"},
        muted_powers={"GERMANY"},
        valid_powers=POWERS, valid_provinces=PROVINCES,
    )
    assert factory_agents["FRANCE"].kg_advisory_mute is False
    assert factory_agents["GERMANY"].kg_advisory_mute is True
    assert factory_agents["FRANCE"].mind.character_brief is not None
    print(f"\n  factory built {len(factory_agents)} agents; "
          f"GERMANY muted={factory_agents['GERMANY'].kg_advisory_mute}")

    print()
    print("Diplomacy mute sanity check passed.")
