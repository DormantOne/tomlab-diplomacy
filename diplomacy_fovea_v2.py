"""
diplomacy_fovea_v2.py — richer fovea that actually surfaces beliefs,
predictions, the credibility ledger, and active intents.

Why v2 exists
-------------
Reading the prompts dumped by `dump_prompts.py` revealed that the v1 fovea
surfaces almost nothing into the prompt: just `OPEN_COMMITMENT_*` lines
and a single-line `CREDIBILITY_SUMMARY: kept-record: AUS:1/1 RUS:1/1 ...`
ratio. None of the beliefs about other powers, none of the predictions,
none of the active intents that target the addressees.

Two reasons:

1. v1's belief selector requires `BeliefStatus.ACTIVE`, but beliefs are
   created as PROTO and only promote after ≥2 resolved predictions with
   ≥1 confirmed and a ≥50% confirm rate. In a 12-phase game most beliefs
   never promote in time. So the prompt looks empty even though the mind
   has 5 beliefs per agent.

2. v1 only surfaces ONE belief per addressee, no predictions, no recent
   commitments-as-records (only the kept/broken ratio).

v2 fixes both: surface PROTO beliefs (with a marker), allow up to 2-3
beliefs per addressee, surface open predictions about each addressee for
the upcoming phase, show recent kept and broken commitments per addressee
as actual records (not just a ratio), and surface active strategic
intents that target the addressees.

Pluggable ranker for v2.5
--------------------------
Each category goes through a `Ranker` to pick top-K. The default
`HeuristicRanker` uses hp + recency + type-weight (mirrors v1's scoring).
A future LLMReranker can be substituted for any individual category by
wrapping a `Ranker` with `LLMRerank(base_ranker, llm_call)`. v2 is
deliberately built around this interface so v2.5 (LLM-as-judge for
"would this record change my move") drops in without refactor.

Activation
----------
v2 is opt-in via `enable_fovea_v2()`. This rebinds
`diplomacy_llm_protocol.build_fovea` to the v2 implementation. Idempotent
and reversible. Existing build_fovea v1 stays in place untouched.

Public surface
--------------
  TurnFoveaV2                  — duck-compatible with TurnFovea
  build_fovea_v2(mind, ctx)    — drop-in replacement for build_fovea
  enable_fovea_v2()            — monkey-patch in
  disable_fovea_v2()           — undo
  HeuristicRanker, Ranker      — for v2.5

This module does NOT modify any existing file. It can be deleted entirely
to revert to v1, just by not calling enable_fovea_v2().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

from diplomacy_kg_schema import (
    AgentMind,
    BeliefNode, BeliefStatus, BeliefType,
    PredictionNode, PredictionStatus, PredictionWindowKind,
    CommitmentNode, SelfCommitmentNode, CommitmentStatus,
    StrategicIntentNode, StrategicIntentStatus,
    PowerName, PhaseKey,
)
from diplomacy_fovea import CallContext


# ============================================================================
# Budget — keeps the fovea from blowing the prompt
# ============================================================================
# Per-addressee caps. Shared globally across all addressees so a turn with
# 5 addressees doesn't 5x the prompt. These bounds are heuristic; tune by
# reading dumped prompts and seeing what's actually useful.

PER_ADDRESSEE_BELIEFS_K     = 2     # top-K beliefs about this power
PER_ADDRESSEE_PREDICTIONS_K = 1     # top-K open predictions about this power
PER_ADDRESSEE_KEPT_K        = 1     # most recent kept promise from this power
PER_ADDRESSEE_BROKEN_K      = 1     # most recent broken promise from this power
PER_ADDRESSEE_PENDING_K     = 1     # most recent pending promise from this power
GLOBAL_INTENTS_K            = 3     # top active intents touching addressees


# ============================================================================
# Ranker interface (the v2.5 hook)
# ============================================================================


class Ranker(Protocol):
    """Sort candidates by relevance to the current call. Returns top-K."""
    def rank(self, kind: str, candidates: list, context: dict, k: int) -> list: ...


@dataclass
class HeuristicRanker:
    """Default ranker. Pure code, no LLM. Mirrors v1 weighting roughly:

      score = hp + critic_score + recency_factor + type_weight
              − 0.15 * times_foveated   (anti-anchoring)
    """

    def rank(self, kind: str, candidates: list, context: dict, k: int) -> list:
        if not candidates:
            return []
        scored = [(self._score(kind, c, context), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:k]]

    def _score(self, kind: str, c: Any, ctx: dict) -> float:
        if kind == "belief":
            return self._score_belief(c, ctx)
        if kind == "prediction":
            return self._score_prediction(c, ctx)
        if kind == "commitment":
            return self._score_commitment(c, ctx)
        if kind == "intent":
            return self._score_intent(c, ctx)
        return 0.0

    def _score_belief(self, b: BeliefNode, ctx: dict) -> float:
        # Base: hp (clamped to ~3) + critic_score
        score = min(b.hp, 3.0) + b.critic_score * 0.5
        # ACTIVE beliefs preferred over PROTO
        if b.status == BeliefStatus.ACTIVE:
            score += 0.5
        # Type weighting depends on call kind
        is_orders = ctx.get("call_kind") == "orders"
        type_weights_orders = {
            BeliefType.RISK_ASSESSMENT: 1.0,
            BeliefType.TACTICAL_PATTERN: 0.9,
            BeliefType.RELATIONSHIP: 0.7,
            BeliefType.CREDIBILITY: 0.6,
            BeliefType.DISPOSITION: 0.5,
        }
        type_weights_nego = {
            BeliefType.CREDIBILITY: 1.0,
            BeliefType.RELATIONSHIP: 0.85,
            BeliefType.RISK_ASSESSMENT: 0.7,
            BeliefType.DISPOSITION: 0.6,
            BeliefType.TACTICAL_PATTERN: 0.4,
        }
        weights = type_weights_orders if is_orders else type_weights_nego
        score += weights.get(b.belief_type, 0.5) * 0.5
        # Anti-anchoring: heavily-foveated beliefs score lower so different
        # facets surface across turns
        score -= 0.15 * min(b.times_foveated, 5)
        # Evidence boosts: more for than against is good
        score += 0.1 * (len(b.evidence_for) - len(b.evidence_against))
        return score

    def _score_prediction(self, p: PredictionNode, ctx: dict) -> float:
        # Confidence is the headline; boost open predictions targeting the
        # upcoming phase
        score = p.confidence
        if p.status == PredictionStatus.OPEN:
            score += 0.3
        upcoming = ctx.get("upcoming_phase")
        if upcoming and p.prediction_window == upcoming:
            score += 0.5
        if p.window_kind == PredictionWindowKind.NEAR_TERM:
            score += 0.2
        return score

    def _score_commitment(self, c: CommitmentNode, ctx: dict) -> float:
        # Recency: more recently-resolved promises matter more for trust
        # signals. Phase string comparison is rough but works since phase
        # strings sort chronologically by year/season.
        score = 0.0
        if c.resolved_at_phase:
            # Most-recent gets the highest score
            score += 1.0
            # In ctx, "phase_today" is the current call's phase. Closer = better.
            phase_today = ctx.get("phase_today", "")
            if c.resolved_at_phase >= phase_today[:4]:  # same year-ish
                score += 0.5
        return score

    def _score_intent(self, i: StrategicIntentNode, ctx: dict) -> float:
        score = min(i.hp, 3.0) + i.critic_score * 0.5
        if i.status == StrategicIntentStatus.ACTIVE:
            score += 0.5
        # Bonus if the intent's targets overlap with current addressees
        addressees = set(ctx.get("addressees", []))
        if addressees & set(i.target_powers):
            score += 0.5
        return score


# ============================================================================
# TurnFoveaV2 — the rendered slice
# ============================================================================


@dataclass
class FoveaSection:
    """One per-addressee section, plus a global section for intents."""
    target_power: Optional[PowerName]
    lines: list[str] = field(default_factory=list)


@dataclass
class TurnFoveaV2:
    """Duck-compatible with TurnFovea: same .render() interface so it
    plugs into existing protocol code unchanged.
    """
    phase: PhaseKey
    addressees: list[PowerName]
    sections: list[FoveaSection]
    strategic_intent_head: Optional[str]
    character_brief_text: str
    credibility_summary_line: Optional[str]

    # Diagnostic fields — useful for /api/kg/<power>/last_prompt to show
    # WHY each line was selected. Not used by render() itself.
    selection_log: list[str] = field(default_factory=list)

    # Private-thoughts integration. Optional — when set, render() includes
    # the agent's recent journal entries and active suspicions.
    _mind: Optional[object] = None

    def render(self) -> str:
        """Compact text rendering for the prompt."""
        out = [self.character_brief_text, ""]
        if self.strategic_intent_head:
            out.append(f"YOUR PLAN: {self.strategic_intent_head}")
            out.append("")
        if self.credibility_summary_line:
            out.append(f"  CREDIBILITY: {self.credibility_summary_line}")
            out.append("")
        # Private journal + suspicions (Layer 6 — inner narrative).
        # Imported lazily to avoid module-load coupling.
        try:
            from diplomacy_private_thoughts import (
                render_journal_for_prompt, render_suspicions_for_prompt,
            )
            jtxt = render_journal_for_prompt(self._mind, n=3) if self._mind else ""
            stxt = render_suspicions_for_prompt(self._mind, max_per_power=1) if self._mind else ""
            if jtxt:
                out.append("YOUR RECENT PRIVATE THOUGHTS (yours alone — never shared):")
                for line in jtxt.splitlines():
                    out.append(f"  {line}")
                out.append("")
            if stxt:
                out.append("YOUR SUSPICIONS (private):")
                for line in stxt.splitlines():
                    out.append(f"  {line}")
                out.append("")
        except Exception:
            pass  # journal module is optional
        for sec in self.sections:
            if not sec.lines:
                continue
            if sec.target_power is not None:
                out.append(f"═══ ABOUT {sec.target_power} ═══")
            for line in sec.lines:
                out.append(f"  {line}")
            out.append("")
        return "\n".join(out).rstrip() + "\n"


# ============================================================================
# Per-record formatters
# ============================================================================
# Each formatter compresses a record into one prompt line. Tuned for
# readability + token efficiency. ~80-150 chars per line.


def _fmt_belief(b: BeliefNode) -> str:
    """Belief one-liner.

      BELIEF[credibility, hp=1.4, ✓3/✗1]: Russia keeps tactical promises but evades long-term ones.
      BELIEF[risk, PROTO, hp=1.0]: Russia will pivot south against Turkey by F1903.
    """
    type_short = b.belief_type.value
    parts = [type_short, f"hp={b.hp:.1f}"]
    if b.status == BeliefStatus.PROTO:
        parts.insert(1, "PROTO")
    elif b.status == BeliefStatus.ACTIVE:
        # No marker for active — that's the default
        pass
    elif b.status == BeliefStatus.RETIRED:
        parts.insert(1, "RETIRED")
    if b.evidence_for or b.evidence_against:
        parts.append(f"✓{len(b.evidence_for)}/✗{len(b.evidence_against)}")
    head = b.head.strip()
    if len(head) > 140:
        head = head[:137] + "..."
    return f"BELIEF[{', '.join(parts)}]: {head}"


def _fmt_prediction(p: PredictionNode) -> str:
    """Prediction one-liner.

      PREDICTION[→RUM, near_term, conf=0.7, OPEN]: Galician build + BLA posture
    """
    target = p.predicted_target or p.predicted_subject_power or "?"
    parts = [f"→{target}", p.window_kind.value, f"conf={p.confidence:.1f}"]
    if p.status == PredictionStatus.OPEN:
        parts.append("OPEN")
    elif p.status == PredictionStatus.CONFIRMED:
        parts.append("✓")
    elif p.status == PredictionStatus.REFUTED:
        parts.append("✗")
    rationale = (p.rationale or "").strip()
    if len(rationale) > 100:
        rationale = rationale[:97] + "..."
    return f"PREDICTION[{', '.join(parts)}]: {rationale or p.predicted_event_type}"


def _fmt_commitment(c: CommitmentNode, marker: str) -> str:
    """Commitment one-liner. `marker` is KEPT/BROKEN/PENDING.

      PROMISE_KEPT[from RUSSIA]: not_move_to GAL by 1902-SPRING-MOVES (resolved S1902)
    """
    bits = [c.type.value]
    subject = c.subject_unit or c.subject_province or c.counterparty or ""
    if subject:
        bits.append(subject)
    if c.target_province:
        bits.append(f"→{c.target_province}")
    bits.append(f"by {c.deadline_phase}")
    body = " ".join(bits)
    suffix = ""
    if c.resolved_at_phase and marker != "PENDING":
        suffix = f" (resolved {c.resolved_at_phase})"
    return f"PROMISE_{marker}[from {c.speaker}]: {body}{suffix}"


def _fmt_self_commitment(c: SelfCommitmentNode, marker: str) -> str:
    """Self-commitment one-liner.

      MY_PROMISE_PENDING[to GERMANY]: not_move_to BUR by 1902-FALL-MOVES
    """
    bits = [c.type.value]
    subject = c.subject_unit or c.subject_province or c.counterparty or ""
    if subject:
        bits.append(subject)
    if c.target_province:
        bits.append(f"→{c.target_province}")
    bits.append(f"by {c.deadline_phase}")
    body = " ".join(bits)
    target = c.target_power or "all"
    return f"MY_PROMISE_{marker}[to {target}]: {body}"


def _fmt_intent(i: StrategicIntentNode) -> str:
    """Intent one-liner.

      INTENT[active]: Block Russia's southern pivot via Italian alliance (RUSSIA, ITALY)
    """
    targets = ", ".join(i.target_powers) if i.target_powers else "—"
    head = i.head.strip()
    if len(head) > 100:
        head = head[:97] + "..."
    return f"INTENT[{i.status.value}, hp={i.hp:.1f}]: {head} (targets: {targets})"


# ============================================================================
# Selection helpers (Pass A: structural filters)
# ============================================================================


def _beliefs_about(mind: AgentMind, target_power: PowerName) -> list[BeliefNode]:
    """All non-retired beliefs about a target power, both PROTO and ACTIVE."""
    return [
        b for b in mind.beliefs.values()
        if b.about_power == target_power
        and b.status in (BeliefStatus.PROTO, BeliefStatus.ACTIVE)
    ]


def _predictions_about(mind: AgentMind, target_power: PowerName,
                       upcoming_phase: Optional[PhaseKey]) -> list[PredictionNode]:
    """Open predictions about a target power.

    Prefer predictions whose window matches the upcoming phase, but include
    other OPEN predictions about this power as fallbacks.
    """
    out = []
    for p in mind.predictions.values():
        if p.about_power != target_power and p.predicted_subject_power != target_power:
            continue
        if p.status != PredictionStatus.OPEN:
            continue
        out.append(p)
    return out


def _incoming_commitments_from(
    mind: AgentMind, speaker: PowerName,
) -> dict[str, list[CommitmentNode]]:
    """Bucket incoming commitments by status for one speaker."""
    buckets: dict[str, list[CommitmentNode]] = {
        "kept": [], "broken": [], "pending": [],
    }
    for c in mind.incoming_commitments.values():
        if c.speaker != speaker:
            continue
        if c.status == CommitmentStatus.KEPT:
            buckets["kept"].append(c)
        elif c.status == CommitmentStatus.BROKEN:
            buckets["broken"].append(c)
        elif c.status == CommitmentStatus.PENDING:
            buckets["pending"].append(c)
        # IRRELEVANT and UNPARSEABLE intentionally dropped — they're noise
    return buckets


def _self_commitments_to(
    mind: AgentMind, target: PowerName,
) -> list[SelfCommitmentNode]:
    """Pending self-commitments where target is target_power or in addressees."""
    out = []
    for c in mind.self_commitments.values():
        if c.status != CommitmentStatus.PENDING:
            continue
        if c.target_power == target or target in c.addressees:
            out.append(c)
    return out


def _intents_targeting(
    mind: AgentMind, addressees: Iterable[PowerName],
) -> list[StrategicIntentNode]:
    """Active strategic intents that target any of the addressees."""
    addr_set = set(addressees)
    out = []
    for i in mind.strategic_intents.values():
        if i.status != StrategicIntentStatus.ACTIVE:
            continue
        if not i.target_powers or addr_set & set(i.target_powers):
            out.append(i)
    return out


def _credibility_summary_line(
    mind: AgentMind, addressees: Iterable[PowerName],
) -> Optional[str]:
    """Compact ratio: 'AUS:1/1 RUS:1/1 GER:0/1 ENG:--/--'.

    Format: 'POWER:kept/total' across all resolved promises from that power.
    Powers with no resolved data show '--/--'. v1 had this; we keep it as
    a cheap one-line overview alongside the per-record details.
    """
    bits = []
    for power in addressees:
        kept = 0
        total = 0
        for c in mind.incoming_commitments.values():
            if c.speaker != power:
                continue
            if c.status == CommitmentStatus.KEPT:
                kept += 1
                total += 1
            elif c.status == CommitmentStatus.BROKEN:
                total += 1
        short = power[:3]
        if total == 0:
            bits.append(f"{short}:--/--")
        else:
            bits.append(f"{short}:{kept}/{total}")
    if not bits:
        return None
    return " ".join(bits)


# ============================================================================
# build_fovea_v2 — the public entry point
# ============================================================================


def build_fovea_v2(
    mind: AgentMind,
    context: CallContext,
    *,
    ranker: Optional[Ranker] = None,
) -> TurnFoveaV2:
    """Drop-in replacement for build_fovea.

    Returns TurnFoveaV2 (duck-compatible with TurnFovea — same .render()
    method) with rich per-addressee content: top-K beliefs (PROTO + ACTIVE),
    top open predictions, recent kept/broken/pending commitments, plus
    active strategic intents that target the addressees.
    """
    if ranker is None:
        ranker = HeuristicRanker()

    upcoming = _guess_upcoming_phase(context.phase)
    addressees = [p for p in context.addressees if p != mind.owner_power]

    rank_ctx = {
        "call_kind": context.kind,
        "addressees": addressees,
        "phase_today": context.phase,
        "upcoming_phase": upcoming,
    }

    selection_log: list[str] = []
    sections: list[FoveaSection] = []

    # ---- Per-addressee sections ----
    for power in addressees:
        sec = FoveaSection(target_power=power)

        # Beliefs about this power (top-K, ACTIVE-preferred via ranker)
        belief_candidates = _beliefs_about(mind, power)
        if belief_candidates:
            top_beliefs = ranker.rank(
                "belief", belief_candidates, rank_ctx, PER_ADDRESSEE_BELIEFS_K,
            )
            for b in top_beliefs:
                sec.lines.append(_fmt_belief(b))
            selection_log.append(
                f"{power}: {len(belief_candidates)} belief candidates → "
                f"selected {len(top_beliefs)}"
            )

        # Open predictions about this power
        pred_candidates = _predictions_about(mind, power, upcoming)
        if pred_candidates:
            top_preds = ranker.rank(
                "prediction", pred_candidates, rank_ctx,
                PER_ADDRESSEE_PREDICTIONS_K,
            )
            for p in top_preds:
                sec.lines.append(_fmt_prediction(p))

        # Recent kept + broken + pending from this power (ledger as records)
        buckets = _incoming_commitments_from(mind, power)
        for marker_key, k in (("kept", PER_ADDRESSEE_KEPT_K),
                              ("broken", PER_ADDRESSEE_BROKEN_K),
                              ("pending", PER_ADDRESSEE_PENDING_K)):
            cands = buckets[marker_key]
            if not cands:
                continue
            top = ranker.rank("commitment", cands, rank_ctx, k)
            for c in top:
                sec.lines.append(_fmt_commitment(c, marker_key.upper()))

        # My pending self-commitments to this power
        my_pending = _self_commitments_to(mind, power)
        if my_pending:
            top_self = ranker.rank(
                "commitment", my_pending, rank_ctx, PER_ADDRESSEE_PENDING_K,
            )
            for c in top_self:
                sec.lines.append(_fmt_self_commitment(c, "PENDING"))

        if sec.lines:
            sections.append(sec)

    # ---- Global intent section ----
    intents = _intents_targeting(mind, addressees)
    if intents:
        top_intents = ranker.rank("intent", intents, rank_ctx, GLOBAL_INTENTS_K)
        if top_intents:
            intent_sec = FoveaSection(target_power=None)
            for i in top_intents:
                intent_sec.lines.append(_fmt_intent(i))
            sections.append(intent_sec)

    # ---- Single overview line ----
    cred_line = _credibility_summary_line(mind, addressees)

    # ---- Strategic intent head (the agent's currently committed intent) ----
    intent_head = None
    if mind.strategic_intents:
        active = [i for i in mind.strategic_intents.values()
                  if i.status == StrategicIntentStatus.ACTIVE]
        if active:
            active.sort(key=lambda i: -i.hp)
            intent_head = active[0].head

    return TurnFoveaV2(
        phase=context.phase,
        addressees=addressees,
        sections=sections,
        strategic_intent_head=intent_head,
        character_brief_text=(
            mind.character_brief.text if mind.character_brief else ""
        ),
        credibility_summary_line=cred_line,
        selection_log=selection_log,
        _mind=mind,
    )


def _guess_upcoming_phase(phase: PhaseKey) -> Optional[PhaseKey]:
    """Best-effort: phase strings look like '1902-SPRING-MOVES'.
    Compute a rough "upcoming" key for prediction-window matching.
    """
    if not phase or "-" not in phase:
        return None
    parts = phase.split("-")
    if len(parts) < 3:
        return None
    year, season, kind = parts[0], parts[1], parts[2]
    # Coarse: same year, next phase. Spring→Fall, Fall→Spring next year, etc.
    # Don't try to be perfect; this is a HINT for the ranker, not a contract.
    if season == "SPRING":
        return f"{year}-FALL-MOVES"
    if season == "FALL":
        try:
            ny = int(year) + 1
            return f"{ny}-SPRING-MOVES"
        except ValueError:
            return None
    return None


# ============================================================================
# Activation — opt-in monkey-patch
# ============================================================================


_PATCHED = False
_ORIGINAL_BUILD_FOVEA = None


def enable_fovea_v2() -> None:
    """Rebind diplomacy_llm_protocol.build_fovea to the v2 implementation.

    Idempotent. Reversible via disable_fovea_v2(). Note: this affects ALL
    agents that share the same Python process — there's no per-agent toggle
    at this layer. (Per-agent A/B testing in the eval harness should pass
    through eval_ablation conditions, not this switch.)
    """
    global _PATCHED, _ORIGINAL_BUILD_FOVEA
    import diplomacy_llm_protocol as _proto
    if _PATCHED:
        return
    _ORIGINAL_BUILD_FOVEA = _proto.build_fovea
    _proto.build_fovea = build_fovea_v2
    _PATCHED = True


def disable_fovea_v2() -> None:
    """Restore the original build_fovea binding."""
    global _PATCHED, _ORIGINAL_BUILD_FOVEA
    import diplomacy_llm_protocol as _proto
    if not _PATCHED:
        return
    _proto.build_fovea = _ORIGINAL_BUILD_FOVEA
    _ORIGINAL_BUILD_FOVEA = None
    _PATCHED = False


def is_fovea_v2_active() -> bool:
    return _PATCHED


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    import time as _t
    from diplomacy_kg_schema import (
        AgentMind, CharacterBrief, BeliefNode, BeliefType, BeliefStatus,
        PredictionNode, PredictionStatus, PredictionWindowKind,
        CommitmentNode, SelfCommitmentNode, CommitmentType, CommitmentStatus,
        StrategicIntentNode, StrategicIntentStatus, new_id,
    )

    print("=" * 72)
    print("FOVEA V2 SANITY CHECK")
    print("=" * 72)

    # Build a realistic mid-game mind
    mind = AgentMind(owner_power="FRANCE", archetype="ARCHITECT_LIRA")
    mind.character_brief = CharacterBrief(
        id=new_id("brief"), archetype="ARCHITECT_LIRA",
        text="I am Architect Lira. I look at the whole table and design "
             "the equilibrium I prefer.",
        generated_at=_t.time(),
    )

    # 5 beliefs (mix of PROTO and ACTIVE) about RUSSIA and GERMANY
    beliefs_data = [
        ("RUSSIA", BeliefType.CREDIBILITY, BeliefStatus.ACTIVE, 1.4,
         "Russia keeps tactical promises but evades long-term ones."),
        ("RUSSIA", BeliefType.RISK_ASSESSMENT, BeliefStatus.PROTO, 1.0,
         "Russia will pivot south against Turkey by F1903."),
        ("RUSSIA", BeliefType.RELATIONSHIP, BeliefStatus.PROTO, 0.9,
         "Russia and Austria coordinating on the Balkans."),
        ("GERMANY", BeliefType.DISPOSITION, BeliefStatus.ACTIVE, 1.2,
         "Germany is paranoid about French intentions in BUR."),
        ("ENGLAND", BeliefType.CREDIBILITY, BeliefStatus.PROTO, 0.8,
         "England has been reliable through 1902."),
    ]
    for pwr, btype, status, hp, head in beliefs_data:
        b = BeliefNode(
            id=new_id("belief"), about_power=pwr, belief_type=btype,
            head=head, body="(evidence summary)",
            formed_at_phase="1902-SPRING-MOVES", formed_in_game=1,
            last_updated_phase="1902-FALL-MOVES",
            hp=hp, evidence_for=["mv_1", "mv_2"], evidence_against=[],
            status=status,
        )
        mind.beliefs[b.id] = b

    # Predictions about RUSSIA and GERMANY for upcoming phase
    for pwr, target, conf in [
        ("RUSSIA", "RUM", 0.7),
        ("RUSSIA", "BUL", 0.5),
        ("GERMANY", "BUR", 0.6),
    ]:
        p = PredictionNode(
            id=new_id("pred"), about_power=pwr,
            formed_at_phase="1902-FALL-MOVES",
            predicted_event_type="move_to", predicted_target=target,
            predicted_subject_power=None,
            prediction_window="1903-SPRING-MOVES",
            window_kind=PredictionWindowKind.NEAR_TERM,
            confidence=conf, status=PredictionStatus.OPEN,
            rationale=f"based on observed {pwr} positioning",
        )
        mind.predictions[p.id] = p

    # Incoming commitments: 1 kept and 1 broken from RUSSIA, 1 pending
    for i, (pwr, status, deadline) in enumerate([
        ("RUSSIA", CommitmentStatus.KEPT, "1902-SPRING-MOVES"),
        ("RUSSIA", CommitmentStatus.BROKEN, "1902-FALL-MOVES"),
        ("RUSSIA", CommitmentStatus.PENDING, "1903-FALL-MOVES"),
        ("GERMANY", CommitmentStatus.KEPT, "1902-FALL-MOVES"),
    ]):
        c = CommitmentNode(
            id=new_id("cmt"), source_msg_id=f"msg_{i}",
            speaker=pwr, addressees=["FRANCE"],
            type=CommitmentType.NOT_MOVE_TO,
            subject_unit=None, subject_province="GAL" if pwr == "RUSSIA" else "BUR",
            target_province=None, counterparty=None,
            deadline_phase=deadline, conditional_on=None,
            status=status, resolved_at_phase=deadline,
            raw_commitspeak_line="not_move_to: GAL by ...",
        )
        mind.incoming_commitments[c.id] = c

    # My self-commitment to GERMANY
    sc = SelfCommitmentNode(
        id=new_id("cmt"), source_msg_id="msg_self_1",
        speaker="FRANCE", addressees=["GERMANY"],
        type=CommitmentType.NOT_MOVE_TO,
        subject_unit=None, subject_province="BUR",
        target_province=None, counterparty=None,
        deadline_phase="1903-SPRING-MOVES", conditional_on=None,
        status=CommitmentStatus.PENDING, target_power="GERMANY",
        raw_commitspeak_line="not_move_to: BUR by 1903-SPRING-MOVES",
    )
    mind.self_commitments[sc.id] = sc

    # Two strategic intents
    for head, targets, status in [
        ("Block Russia's southern pivot via Italian alliance.",
         ["RUSSIA", "ITALY"], StrategicIntentStatus.ACTIVE),
        ("Maintain quiet front with England through 1903.",
         ["ENGLAND"], StrategicIntentStatus.ACTIVE),
    ]:
        i_node = StrategicIntentNode(
            id=new_id("intent"), head=head, body="(detail)",
            formed_at_phase="1902-FALL-MOVES",
            target_powers=targets, target_provinces=[],
            horizon="1904-FALL-MOVES", hp=1.1, status=status,
            active_since_phase="1902-FALL-MOVES",
        )
        mind.strategic_intents[i_node.id] = i_node

    # ---- Build for negotiate to RUSSIA + GERMANY ----
    ctx = CallContext(
        kind="negotiate", phase="1903-SPRING-MOVES",
        addressees=["RUSSIA", "GERMANY"],
        current_phase_for_relevance="1903-SPRING-MOVES",
    )
    fovea = build_fovea_v2(mind, ctx)
    rendered = fovea.render()
    print(f"\n  RENDERED FOVEA ({len(rendered)} chars, ~{len(rendered)//4} tokens):")
    print("  " + "-" * 70)
    for ln in rendered.split("\n"):
        print(f"  {ln}")
    print("  " + "-" * 70)

    # ---- Assertions ----
    assert "Russia keeps tactical promises" in rendered, \
        "active belief about Russia must surface"
    assert "Russia will pivot south" in rendered or "PROTO" in rendered, \
        "PROTO belief must surface (with PROTO marker)"
    assert "Germany is paranoid" in rendered, \
        "active belief about Germany must surface"
    assert "PROMISE_KEPT" in rendered, "kept commitment must surface"
    assert "PROMISE_BROKEN" in rendered, "broken commitment must surface"
    assert "PROMISE_PENDING" in rendered, "pending commitment must surface"
    assert "MY_PROMISE_PENDING" in rendered, "my self-commitment must surface"
    assert "PREDICTION" in rendered, "open prediction must surface"
    assert "INTENT" in rendered, "active intent must surface"
    assert "═══ ABOUT RUSSIA ═══" in rendered, "RUSSIA section header"
    assert "═══ ABOUT GERMANY ═══" in rendered, "GERMANY section header"
    # Russia has 1 kept + 1 broken = 1/2; Germany has 1 kept + 0 = 1/1
    assert "RUS:1/2" in rendered, "credibility ratio for Russia"
    assert "GER:1/1" in rendered, "credibility ratio for Germany"
    print(f"\n  All content assertions passed.")

    # ---- Length check ----
    assert len(rendered) < 3000, \
        f"fovea too large: {len(rendered)} chars (target <3000)"
    print(f"  Length within budget: {len(rendered)} chars")

    # ---- Compare to v1 fovea ----
    from diplomacy_fovea import build_fovea as build_fovea_v1
    v1_fovea = build_fovea_v1(mind, ctx)
    v1_rendered = v1_fovea.render()
    print(f"\n  v1 fovea on same mind: {len(v1_rendered)} chars")
    print(f"  v2 fovea on same mind: {len(rendered)} chars  "
          f"(+{len(rendered) - len(v1_rendered)} chars more content)")
    # v1 should be much smaller because PROTO beliefs filtered out
    assert len(rendered) > len(v1_rendered) + 500, \
        "v2 must be substantially richer than v1"

    # ---- enable / disable ----
    print(f"\n  Testing enable/disable...")
    import diplomacy_llm_protocol as _proto
    original = _proto.build_fovea
    enable_fovea_v2()
    assert _proto.build_fovea is build_fovea_v2
    assert is_fovea_v2_active()
    # Idempotent
    enable_fovea_v2()
    assert _proto.build_fovea is build_fovea_v2
    disable_fovea_v2()
    assert _proto.build_fovea is original
    assert not is_fovea_v2_active()
    # Idempotent
    disable_fovea_v2()
    assert _proto.build_fovea is original
    print(f"  enable/disable: idempotent + reversible")

    print()
    print("Fovea v2 sanity check passed.")
