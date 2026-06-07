"""
diplomacy_dreaming.py — end-of-year consolidation for V2 agents.

Fires once per game year (after FALL ADJUSTMENT). The agent steps back
from live play and reflects across all of recent state:

  - Reviews accumulated beliefs (PROTO and ACTIVE)
  - Reviews self_commitments outcomes (kept_rate, broken_rate)
  - Reviews incoming_commitments outcomes (who keeps faith, who breaks it)
  - Reviews resolved predictions (calibration check)
  - Looks for cross-belief patterns the per-phase journal might miss
  - Writes a single LONG consolidation entry to private_journal
  - Adjusts belief HPs based on accumulated evidence
  - Promotes suspicions into typed PROTO beliefs (NEW in this version)

This addresses three documented failure modes:
  1. Belief pile-up (87 beliefs, none reviewed) — forces periodic review
  2. Vocabulary contagion — explicit prompt to notice phrasing drift
  3. Suspicion decay — promotes recurrent suspicions to beliefs

Cost: 1 LLM call per agent per game year. ~$0.05 added per 12-year game on Haiku.

CHANGELOG (this version):
  - Added PROMOTE_SUSPICION action. The dream can now graduate a suspicion
    (which is a free-text observation tag) into a typed BeliefNode. The
    new belief enters PROTO status and earns ACTIVE through the normal
    lifecycle. The suspicion is removed once promoted.
  - Telemetry now exposes `suspicions_promoted` count.
  - PROMOTE/RETIRE/DOWNGRADE on a ref that doesn't match a belief now
    falls back to checking the suspicions list before recording not_found.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional


# ============================================================================
# Prompt template helpers
# ============================================================================


def _summarize_beliefs(mind, max_per_target: int = 4) -> str:
    if not mind.beliefs:
        return "(no beliefs yet)"
    by_target: dict[str, list] = {}
    for b in mind.beliefs.values():
        by_target.setdefault(b.about_power or "(no target)", []).append(b)
    out = []
    for tgt, items in by_target.items():
        items.sort(key=lambda b: -getattr(b, "hp", 0))
        out.append(f"\n  {tgt}:")
        for b in items[:max_per_target]:
            ev_for = len(getattr(b, "evidence_for", []))
            ev_ag = len(getattr(b, "evidence_against", []))
            status = getattr(b.status, "value", str(b.status))
            hp = getattr(b, "hp", 0.0)
            # Show id stem so the LLM can reference it precisely if it wants
            short_id = (b.id or "").split(":")[-1][:6]
            out.append(f"    [{status} hp={hp:.2f} ✓{ev_for}/✗{ev_ag} id={short_id}] {b.head}")
    return "\n".join(out)


def _summarize_credibility(mind) -> str:
    """Per-power: how often did they keep promises to ME?"""
    by_speaker: dict[str, dict] = {}
    for c in mind.incoming_commitments.values():
        by_speaker.setdefault(c.speaker, {"kept": 0, "broken": 0, "pending": 0})
        s = getattr(c.status, "value", str(c.status))
        if s in ("kept", "broken", "pending"):
            by_speaker[c.speaker][s] += 1
    if not by_speaker:
        return "(no incoming commitments tracked yet)"
    lines = []
    for speaker, counts in by_speaker.items():
        total_resolved = counts["kept"] + counts["broken"]
        if total_resolved > 0:
            rate = counts["kept"] / total_resolved
            lines.append(f"  {speaker}: kept_rate={rate:.0%}  ({counts['kept']}✓ {counts['broken']}✗ {counts['pending']}…)")
        else:
            lines.append(f"  {speaker}: no resolved promises yet ({counts['pending']}…)")
    return "\n".join(lines)


def _summarize_self_commitments(mind) -> str:
    """How often did I keep my own promises?"""
    counts = {"kept": 0, "broken": 0, "pending": 0}
    for c in mind.self_commitments.values():
        s = getattr(c.status, "value", str(c.status))
        if s in counts:
            counts[s] += 1
    total_resolved = counts["kept"] + counts["broken"]
    if total_resolved == 0:
        return f"  (no self-promises resolved yet, {counts['pending']} pending)"
    rate = counts["kept"] / total_resolved
    return f"  kept_rate={rate:.0%}  ({counts['kept']}✓ {counts['broken']}✗ {counts['pending']}…)"


def _summarize_recent_journal(mind, n: int = 8) -> str:
    if not mind.private_journal:
        return "(no journal entries yet)"
    recent = mind.private_journal[-n:]
    out = []
    for e in recent:
        text = (e.get("text") or "").strip().replace("\n", " ")[:160]
        phase = e.get("phase", "?")
        out.append(f"  [{phase}] {text}")
    return "\n".join(out)


def _summarize_suspicions(mind) -> str:
    if not mind.suspicions:
        return "(no active suspicions)"
    by_p: dict[str, list] = {}
    for s in mind.suspicions:
        by_p.setdefault(s.get("about_power", "?"), []).append(s)
    out = []
    for p, items in by_p.items():
        items.sort(key=lambda s: -s.get("weight", 0))
        for s in items[:3]:
            tag = s.get("tag", "?")
            note = s.get("note", "")
            weight = s.get("weight", 0.0)
            out.append(f"  {p} [tag={tag}, w={weight:.2f}]: {note}")
    return "\n".join(out)


# ============================================================================
# The dream prompt
# ============================================================================


_BELIEF_TYPES_HELP = (
    "disposition (long-running character claim, persists across games), "
    "tactical_pattern (recurring observed move pattern, this game only), "
    "relationship (pairwise dynamic, this game only), "
    "risk_assessment (forward-looking threat, short-lived), "
    "credibility (persistent ledger of how they honor commitments)"
)


def compose_dream_prompt(
    *, mind, year: int, board_summary_text: str,
) -> str:
    char = (mind.character_brief.text if mind.character_brief
            else "(no character brief)")
    return f"""You are {mind.owner_power}. The game year {year} has just ended.

YOUR CHARACTER:
{char}

CURRENT BOARD AT YEAR-END:
{board_summary_text}

YOUR ACCUMULATED BELIEFS (sorted by hp per target):
{_summarize_beliefs(mind)}

CREDIBILITY OF OTHERS (their kept/broken promises to YOU):
{_summarize_credibility(mind)}

YOUR OWN PROMISE-KEEPING:
{_summarize_self_commitments(mind)}

YOUR RECENT JOURNAL ENTRIES (last 8):
{_summarize_recent_journal(mind, n=8)}

YOUR ACTIVE SUSPICIONS (free-text pattern observations — these are NOT
beliefs yet; promote them below if they have crystallized into
falsifiable claims):
{_summarize_suspicions(mind)}

This is your YEAR-END CONSOLIDATION. You are not in negotiation now. No
one will see this. Step back and think across the whole year. Address
these questions in 5-8 sentences as a continuous private reflection:

  1. WHAT MATTERS NOW: What is your honest assessment of your position?
     Who is your real ally, who is your real threat, and what changed
     this year?
  2. WHAT YOU GOT WRONG: Were any of your beliefs contradicted by what
     actually happened? What predictions failed?
  3. PATTERNS YOU MISSED: Looking across all your beliefs and the
     credibility data, do you see any patterns you hadn't noticed —
     coalitions forming, common phrasings shared between certain agents,
     a power that talks differently than it acts?
  4. NEXT YEAR PLAN: What are you actually going to do next year, in
     plain language?

After your reflection, optionally update your knowledge graph using the
following actions. ONE action per line, inside [[belief_updates ... ]]:

  PROMOTE_SUSPICION <tag> AS <type> head="<one-sentence falsifiable claim>" new_hp=<0.1-2.0>
      Graduate a suspicion (named by its tag) into a typed PROTO belief.
      The new belief enters lifecycle and earns ACTIVE through later
      predictions. The suspicion is removed.
      <type> is one of: {_BELIEF_TYPES_HELP}.
      Use this for suspicions you now believe deserve formal tracking.

  PROMOTE <belief_id_or_head_text> <new_hp>
      Strengthen an existing belief. Reference by id stem (e.g. "abc123")
      or a snippet of its head text. new_hp typically 1.5+.

  RETIRE <belief_id_or_head_text>
      Remove an existing belief.

  DOWNGRADE <belief_id_or_head_text> <new_hp>
      Weaken an existing belief. new_hp typically 0.3.

Example:

[[belief_updates
  PROMOTE_SUSPICION naked_partition AS relationship head="Germany is coordinating with Russia and Turkey to dismember Austria" new_hp=1.4
  PROMOTE Germany is the silent broker 1.6
  DOWNGRADE Russia respects my demilitarization proposal 0.2
  RETIRE Turkey is the primary threat
]]

YEAR {year} CONSOLIDATION:"""


# ============================================================================
# Parsing
# ============================================================================


_UPDATES_RE = re.compile(r"\[\[belief_updates\s*(.*?)\]\]", re.DOTALL | re.IGNORECASE)

# Structured PROMOTE_SUSPICION line.
# Tolerant of single-quoted, double-quoted, or unquoted head text.
# Captures: 1=tag, 2=belief_type, 3/4/5=head (one of three quoting styles), 6=hp (optional)
_PROMOTE_SUS_RE = re.compile(
    r"""^\s*PROMOTE_SUSPICION
        \s+(\S+)                         # tag
        \s+AS\s+(\w+)                    # belief_type
        \s+head\s*=\s*(?:                # head=
            "([^"]+)"                     # double-quoted head
            | '([^']+)'                   # single-quoted head
            | (\S.+?)                     # unquoted head (greedy until new_hp= or end)
        )
        (?:\s+new_hp\s*=\s*([\d.]+))?    # optional new_hp=
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _parse_belief_updates(text: str) -> list[dict]:
    """Parse the [[belief_updates ... ]] block. Returns a list of dicts.

    Each dict has at minimum {"action": str}. Action-specific fields:
      PROMOTE_SUSPICION → tag, belief_type, head, new_hp
      PROMOTE / DOWNGRADE → ref, new_hp
      RETIRE → ref
    """
    m = _UPDATES_RE.search(text)
    if not m:
        return []
    body = m.group(1).strip()
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try the structured PROMOTE_SUSPICION form first
        ps = _PROMOTE_SUS_RE.match(line)
        if ps:
            tag, btype, h_dq, h_sq, h_uq, hp_str = ps.groups()
            head = (h_dq or h_sq or h_uq or "").strip()
            try:
                new_hp = float(hp_str) if hp_str else 1.0
            except ValueError:
                new_hp = 1.0
            new_hp = max(0.1, min(2.0, new_hp))
            out.append({
                "action": "PROMOTE_SUSPICION",
                "tag": tag,
                "belief_type": btype.lower(),
                "head": head,
                "new_hp": new_hp,
            })
            continue

        # Fall back to the simple action grammar (PROMOTE / RETIRE / DOWNGRADE)
        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            continue
        action = parts[0].upper()
        if action not in ("PROMOTE", "RETIRE", "DOWNGRADE"):
            continue
        ref = parts[1]
        new_hp = None
        if action == "RETIRE" and len(parts) >= 3:
            # RETIRE has no hp; everything after the keyword is the ref.
            ref = parts[1] + " " + parts[2]
        elif action != "RETIRE" and len(parts) >= 3:
            try:
                new_hp = float(parts[2])
            except ValueError:
                # parts[2] is "head text ... 1.5" — split off trailing hp
                tokens = parts[2].rsplit(None, 1)
                if len(tokens) == 2:
                    try:
                        new_hp = float(tokens[1])
                        ref = parts[1] + " " + tokens[0]
                    except ValueError:
                        # No trailing number — treat entire tail as ref
                        ref = parts[1] + " " + parts[2]
        out.append({"action": action, "ref": ref, "new_hp": new_hp})
    return out


def _strip_updates_block(text: str) -> str:
    return _UPDATES_RE.sub("", text).strip()


# ============================================================================
# Apply belief updates
# ============================================================================


_BELIEF_TYPE_ALIASES = {
    "disposition": "disposition",
    "tactical": "tactical_pattern",
    "tactical_pattern": "tactical_pattern",
    "pattern": "tactical_pattern",
    "relationship": "relationship",
    "rel": "relationship",
    "risk": "risk_assessment",
    "risk_assessment": "risk_assessment",
    "threat": "risk_assessment",
    "credibility": "credibility",
    "cred": "credibility",
}


def _resolve_belief_type(name: str):
    """Map a string to a BeliefType enum value. Returns None on failure.

    Imports BeliefType lazily to keep this module's top-level import light.
    """
    from diplomacy_kg_schema import BeliefType
    canonical = _BELIEF_TYPE_ALIASES.get((name or "").strip().lower())
    if canonical is None:
        return None
    try:
        return BeliefType(canonical)
    except ValueError:
        return None


def _find_belief(mind, ref: str):
    """Find a belief by id, id stem, or head substring. Returns the belief or None."""
    ref_lower = ref.lower().strip()
    if ref in mind.beliefs:
        return mind.beliefs[ref]
    # Try id stem match (last segment after ':') and head substring
    for b in mind.beliefs.values():
        bid = b.id or ""
        if bid == ref:
            return b
        # id stem match: "abc123" matches "belief:abc123..."
        bid_stem = bid.split(":")[-1] if ":" in bid else bid
        if bid_stem.startswith(ref) or ref.startswith(bid_stem):
            return b
        if ref_lower and ref_lower in (b.head or "").lower():
            return b
    return None


def _find_suspicion(mind, tag_or_ref: str):
    """Find a suspicion by tag or by case-insensitive note substring."""
    if not mind.suspicions:
        return None
    needle = (tag_or_ref or "").strip().lower()
    if not needle:
        return None
    for s in mind.suspicions:
        if (s.get("tag") or "").lower() == needle:
            return s
    # Loose fallback: substring match in note or tag
    for s in mind.suspicions:
        if needle in (s.get("tag") or "").lower():
            return s
        if needle in (s.get("note") or "").lower():
            return s
    return None


def _promote_suspicion_to_belief(mind, sus: dict, *, belief_type, head: str,
                                  hp: float) -> Optional[object]:
    """Construct a new PROTO BeliefNode from a suspicion. Returns the belief."""
    from diplomacy_kg_schema import (
        BeliefNode, BeliefStatus, BeliefType, new_id,
    )
    about = sus.get("about_power") or "?"
    last_phase = (
        sus.get("last_seen_phase")
        or sus.get("first_seen_phase")
        or "year-end"
    )
    first_phase = sus.get("first_seen_phase") or last_phase
    tag = sus.get("tag") or "untagged"
    note = (sus.get("note") or "").strip()
    body = (
        f"Promoted from suspicion '{tag}' (weight={sus.get('weight', 0):.2f}). "
        f"First observed at {first_phase}; last at {last_phase}. "
        f"Original observation: {note}"
    )
    persists = belief_type in (BeliefType.DISPOSITION, BeliefType.CREDIBILITY)
    new_belief = BeliefNode(
        id=new_id("belief"),
        about_power=about,
        belief_type=belief_type,
        head=head,
        body=body,
        formed_at_phase=last_phase,
        formed_in_game=getattr(mind, "games_played", 0),
        last_updated_phase=last_phase,
        hp=hp,
        status=BeliefStatus.PROTO,
        persists_across_games=persists,
    )
    mind.beliefs[new_belief.id] = new_belief
    return new_belief


def _apply_belief_updates(mind, updates: list[dict]) -> dict:
    """Mutate the mind in place. Returns telemetry counts."""
    promoted = retired = downgraded = sus_promoted = not_found = bad_type = 0

    for u in updates:
        action = u["action"]

        # ---- PROMOTE_SUSPICION ----------------------------------------
        if action == "PROMOTE_SUSPICION":
            sus = _find_suspicion(mind, u.get("tag", ""))
            if sus is None:
                not_found += 1
                continue
            btype = _resolve_belief_type(u.get("belief_type", ""))
            if btype is None:
                bad_type += 1
                continue
            head = (u.get("head") or "").strip()
            if not head:
                # Fall back to suspicion note as head if LLM omitted it
                head = (sus.get("note") or u.get("tag") or "").strip()[:200]
            hp = u.get("new_hp", 1.0) or 1.0
            try:
                _promote_suspicion_to_belief(
                    mind, sus, belief_type=btype, head=head, hp=hp,
                )
                # Remove the promoted suspicion from the live list
                mind.suspicions = [s for s in mind.suspicions if s is not sus]
                sus_promoted += 1
            except Exception:
                not_found += 1
            continue

        # ---- PROMOTE / RETIRE / DOWNGRADE -----------------------------
        ref = u.get("ref", "")
        b = _find_belief(mind, ref)
        if b is None:
            # Forgiving fallback: maybe the LLM used PROMOTE on a suspicion.
            sus = _find_suspicion(mind, ref)
            if sus is not None and action in ("PROMOTE", "DOWNGRADE"):
                # Treat as PROMOTE_SUSPICION with a reasonable default type
                from diplomacy_kg_schema import BeliefType
                try:
                    head = (sus.get("note") or sus.get("tag") or "").strip()[:200]
                    hp = u.get("new_hp") or (1.5 if action == "PROMOTE" else 0.3)
                    _promote_suspicion_to_belief(
                        mind, sus, belief_type=BeliefType.RELATIONSHIP,
                        head=head, hp=hp,
                    )
                    mind.suspicions = [s for s in mind.suspicions if s is not sus]
                    sus_promoted += 1
                    continue
                except Exception:
                    pass
            not_found += 1
            continue

        if action == "RETIRE":
            try:
                del mind.beliefs[b.id]
                retired += 1
            except Exception:
                not_found += 1
        elif action in ("PROMOTE", "DOWNGRADE"):
            new_hp = u.get("new_hp")
            if new_hp is None:
                new_hp = 1.5 if action == "PROMOTE" else 0.3
            new_hp = max(0.1, min(2.0, new_hp))
            b.hp = new_hp
            if action == "PROMOTE":
                promoted += 1
            else:
                downgraded += 1

    return {
        "promoted": promoted,
        "retired": retired,
        "downgraded": downgraded,
        "suspicions_promoted": sus_promoted,
        "not_found": not_found,
        "bad_type": bad_type,
    }


# ============================================================================
# Top-level: run a dream pass
# ============================================================================


def run_dream(
    agent,
    *,
    year: int,
    board_summary_text: str,
) -> Optional[dict]:
    """One LLM call. Writes a long consolidation entry to mind.private_journal,
    adjusts beliefs, and promotes any suspicions the LLM elects to graduate.
    Failures are non-fatal.

    `agent` is a DiplomacyAgentV2 (use bridge.v2 for V2BridgeAgent).
    """
    try:
        prompt = compose_dream_prompt(
            mind=agent.mind, year=year, board_summary_text=board_summary_text,
        )
        response = agent.llm_call(prompt)
        if not response:
            return None
        updates = _parse_belief_updates(response)
        text = _strip_updates_block(response)
        # Cap entry length to keep the journal from blowing the prompt budget
        if len(text) > 1600:
            text = text[:1600] + "…"
        entry = {
            "phase": f"YEAR-{year}-DREAM",
            "kind": "dream-consolidation",
            "timestamp": time.time(),
            "text": text,
        }
        agent.mind.private_journal.append(entry)
        # Keep the journal bounded
        if len(agent.mind.private_journal) > 80:
            agent.mind.private_journal = agent.mind.private_journal[-80:]
        update_telemetry = _apply_belief_updates(agent.mind, updates) if updates else {}
        return {
            "entry": entry,
            "updates_telemetry": update_telemetry,
            "n_updates_parsed": len(updates),
        }
    except Exception as e:
        print(f"[dream] failed for {agent.power}: {e}")
        return None


# ============================================================================
# Sanity check
# ============================================================================


if __name__ == "__main__":
    print("=" * 72)
    print("DREAMING SANITY CHECK")
    print("=" * 72)

    # ---- 1. Parse a mixed update block --------------------------------
    sample = """
Some prose here.

[[belief_updates
  PROMOTE_SUSPICION naked_partition AS relationship head="Germany coordinates with Russia and Turkey to dismember Austria" new_hp=1.4
  PROMOTE_SUSPICION explicit_charter AS risk_assessment head='Turkey holds the Balkans by mutual agreement' new_hp=1.5
  PROMOTE Germany is the silent broker 1.6
  DOWNGRADE Russia respects my demilitarization proposal 0.2
  RETIRE Turkey is the primary threat
]]
"""
    updates = _parse_belief_updates(sample)
    print(f"  parsed {len(updates)} updates from sample")
    for u in updates:
        print(f"    {u}")
    assert len(updates) == 5
    assert updates[0]["action"] == "PROMOTE_SUSPICION"
    assert updates[0]["tag"] == "naked_partition"
    assert updates[0]["belief_type"] == "relationship"
    assert "Germany coordinates" in updates[0]["head"]
    assert updates[0]["new_hp"] == 1.4
    assert updates[1]["belief_type"] == "risk_assessment"
    assert updates[2]["action"] == "PROMOTE"
    assert updates[3]["action"] == "DOWNGRADE"
    assert updates[4]["action"] == "RETIRE"

    # ---- 2. Apply against a stub mind ---------------------------------
    from diplomacy_kg_schema import (
        AgentMind, CharacterBrief, BeliefNode, BeliefType, BeliefStatus,
        new_id,
    )

    mind = AgentMind(owner_power="AUSTRIA", archetype="PARSON_HAWTHORNE")
    mind.character_brief = CharacterBrief(
        id=new_id("brief"), archetype="PARSON_HAWTHORNE",
        text="I am Parson Hawthorne.", generated_at=time.time(),
    )
    # Two existing beliefs
    b1 = BeliefNode(
        id=new_id("belief"), about_power="GERMANY",
        belief_type=BeliefType.RELATIONSHIP,
        head="Germany is the silent broker—neither ally nor honest neutral.",
        body="(elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1903-FALL-MOVES",
        status=BeliefStatus.ACTIVE, hp=1.0,
    )
    b2 = BeliefNode(
        id=new_id("belief"), about_power="TURKEY",
        belief_type=BeliefType.RISK_ASSESSMENT,
        head="Turkey is the primary threat.",
        body="(elided)",
        formed_at_phase="1901-FALL-MOVES", formed_in_game=1,
        last_updated_phase="1903-FALL-MOVES",
        status=BeliefStatus.PROTO, hp=0.7,
    )
    mind.beliefs[b1.id] = b1
    mind.beliefs[b2.id] = b2

    # Two suspicions; one will be promoted
    mind.suspicions = [
        {
            "about_power": "GERMANY", "tag": "naked_partition",
            "note": "Germany acts as choreographer, not neutral",
            "first_seen_phase": "1903-SPRING-MOVES",
            "last_seen_phase": "1904-FALL-MOVES",
            "weight": 0.8,
        },
        {
            "about_power": "TURKEY", "tag": "explicit_charter",
            "note": "Turkey expands Balkans unopposed by all powers",
            "first_seen_phase": "1903-FALL-MOVES",
            "last_seen_phase": "1904-FALL-MOVES",
            "weight": 0.7,
        },
        {
            "about_power": "RUSSIA", "tag": "weakened_position",
            "note": "Lost Rumania quietly",
            "first_seen_phase": "1902-SPRING-MOVES",
            "last_seen_phase": "1902-FALL-MOVES",
            "weight": 0.4,
        },
    ]

    # Force a downgrade target that uses substring matching
    updates_for_apply = [
        {"action": "PROMOTE_SUSPICION", "tag": "naked_partition",
         "belief_type": "relationship",
         "head": "Germany coordinates with Russia and Turkey to dismember Austria",
         "new_hp": 1.4},
        {"action": "PROMOTE_SUSPICION", "tag": "explicit_charter",
         "belief_type": "risk_assessment",
         "head": "Turkey holds the Balkans by mutual agreement",
         "new_hp": 1.5},
        {"action": "PROMOTE", "ref": "silent broker", "new_hp": 1.6},
        {"action": "DOWNGRADE", "ref": "primary threat", "new_hp": 0.3},
        {"action": "RETIRE", "ref": "nonexistent belief reference"},
    ]
    telemetry = _apply_belief_updates(mind, updates_for_apply)
    print(f"  telemetry: {telemetry}")
    assert telemetry["suspicions_promoted"] == 2, f"{telemetry}"
    assert telemetry["promoted"] == 1
    assert telemetry["downgraded"] == 1
    assert telemetry["not_found"] == 1
    # Beliefs after: original 2 + 2 promoted = 4
    assert len(mind.beliefs) == 4, f"{len(mind.beliefs)} beliefs"
    # Suspicions after: 3 - 2 = 1 left (weakened_position)
    assert len(mind.suspicions) == 1
    assert mind.suspicions[0]["tag"] == "weakened_position"
    # Promoted beliefs are PROTO
    new_beliefs = [b for b in mind.beliefs.values()
                   if b.id not in (b1.id, b2.id)]
    assert all(b.status == BeliefStatus.PROTO for b in new_beliefs)
    # Existing beliefs got their hp adjusted
    assert b1.hp == 1.6
    assert b2.hp == 0.3

    # ---- 3. Forgiving fallback: PROMOTE on a suspicion tag ------------
    mind2 = AgentMind(owner_power="FRANCE", archetype="BARON_KORVIN")
    mind2.suspicions = [
        {"about_power": "RUSSIA", "tag": "encroachment",
         "note": "Russia inching toward Austria's border",
         "first_seen_phase": "1902-FALL-MOVES",
         "last_seen_phase": "1903-FALL-MOVES",
         "weight": 0.6},
    ]
    telemetry2 = _apply_belief_updates(mind2, [
        {"action": "PROMOTE", "ref": "encroachment", "new_hp": 1.4},
    ])
    assert telemetry2["suspicions_promoted"] == 1, f"{telemetry2}"
    assert len(mind2.beliefs) == 1
    assert len(mind2.suspicions) == 0
    print(f"  forgiving fallback: PROMOTE on suspicion tag → promoted to belief")

    # ---- 4. bad_type counter ------------------------------------------
    mind3 = AgentMind(owner_power="RUSSIA", archetype="ARCHITECT_LIRA")
    mind3.suspicions = [
        {"about_power": "AUSTRIA", "tag": "x", "note": "n", "weight": 0.5,
         "first_seen_phase": "1901-SPRING-MOVES",
         "last_seen_phase": "1901-SPRING-MOVES"},
    ]
    telemetry3 = _apply_belief_updates(mind3, [
        {"action": "PROMOTE_SUSPICION", "tag": "x",
         "belief_type": "totally_made_up_type", "head": "h", "new_hp": 1.0},
    ])
    assert telemetry3["bad_type"] == 1
    assert telemetry3["suspicions_promoted"] == 0
    print(f"  bad_type counter works: {telemetry3}")

    print()
    print("Dreaming sanity check passed.")
