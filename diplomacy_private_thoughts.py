"""
diplomacy_private_thoughts.py — the agent's inner narrative.

After each phase resolves, the agent gets ONE additional LLM call asking
it to write a brief private journal entry. This entry captures:

  - Suspicions (impressionistic — "X feels off, watching")
  - Working hypotheses ("if Y moves to Z, I should...")
  - Reactions to surprises ("Russia's restraint surprised me")
  - Reassessments ("Austria more aggressive than I thought")

These thoughts are NEVER shared with other agents. They're included in
the agent's own next-phase prompts under "Your recent private thoughts."

This module is purely additive — it doesn't change any existing belief or
prediction logic. It gives the agent a scratchpad for the kind of inchoate
reasoning that doesn't fit the structured belief slots.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

# Note: imports DiplomacyAgentV2 lazily inside functions to avoid circulars.


# ============================================================================
# Prompt template
# ============================================================================


def _format_recent_journal(mind, n: int = 4) -> str:
    """Render the agent's last N journal entries as a compact reference."""
    if not mind.private_journal:
        return "(no prior journal entries)"
    recent = mind.private_journal[-n:]
    lines = []
    for e in recent:
        text = e.get("text", "").strip()
        kind = e.get("kind", "")
        phase = e.get("phase", "")
        lines.append(f"[{phase} {kind}] {text}")
    return "\n".join(lines)


def _format_suspicions(mind, max_per_power: int = 2) -> str:
    """Render current active suspicions, top by weight per target."""
    if not mind.suspicions:
        return "(no suspicions on the table)"
    by_power: dict[str, list[dict]] = {}
    for s in mind.suspicions:
        by_power.setdefault(s.get("about_power", "?"), []).append(s)
    out = []
    for p, items in by_power.items():
        items.sort(key=lambda s: -s.get("weight", 0.0))
        for s in items[:max_per_power]:
            tag = s.get("tag", "?")
            note = s.get("note", "")
            w = s.get("weight", 0.0)
            out.append(f"  {p} [{tag}, w={w:.2f}]: {note}")
    return "\n".join(out) if out else "(no suspicions)"


def compose_journal_prompt(
    *, mind, board_summary_text: str, what_just_happened: str,
    phase: str, kind: str,
) -> str:
    """One-shot prompt asking the agent for a private journal entry.

    The output is plain text — short, impressionistic, in character. Plus
    optional structured suspicions in a tagged JSON block at the end.
    """
    char = (mind.character_brief.text if mind.character_brief
            else "(no character brief)")
    journal = _format_recent_journal(mind, n=4)
    susp = _format_suspicions(mind, max_per_power=2)
    return f"""You are {mind.owner_power} in a Diplomacy game.

YOUR CHARACTER:
{char}

CURRENT PHASE: {phase}
WHEN: {kind}

BOARD STATE:
{board_summary_text}

WHAT JUST HAPPENED:
{what_just_happened}

YOUR RECENT PRIVATE THOUGHTS (your own — NEVER shared):
{journal}

CURRENT SUSPICIONS:
{susp}

Now write a brief private journal entry — 2 to 4 short sentences. This is
inner monologue. Nobody else sees it. What are you noticing? What are you
suspicious about? What's your working hypothesis? Write in your own voice.
Be specific. Don't perform diplomacy — be honest with yourself.

Then, if you noticed something specific that warrants tracking, optionally
emit a structured suspicions block. Use this exact format (only if relevant):

[[suspicions
  POWER tag note text
]]

Where POWER is one of AUSTRIA/ENGLAND/FRANCE/GERMANY/RUSSIA/TURKEY, tag is
a short snake_case label like tonal_shift or move_inconsistency, and note
is a one-line specific observation.

JOURNAL ENTRY:"""


# ============================================================================
# Parsing the response
# ============================================================================


_SUSP_RE = re.compile(r"\[\[suspicions\s*(.*?)\]\]", re.DOTALL | re.IGNORECASE)


def _parse_suspicions_block(text: str) -> list[dict]:
    """Extract structured suspicions from the journal entry, if any."""
    m = _SUSP_RE.search(text)
    if not m:
        return []
    body = m.group(1).strip()
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # POWER tag note...
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        power, tag, note = parts[0].upper(), parts[1].lower(), parts[2]
        if power not in ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "RUSSIA", "TURKEY"):
            continue
        out.append({
            "about_power": power,
            "tag": tag,
            "note": note.strip(),
        })
    return out


def _strip_suspicions_block(text: str) -> str:
    """Remove the [[suspicions]] block from the text so the journal entry
    is clean prose."""
    return _SUSP_RE.sub("", text).strip()


# ============================================================================
# Suspicion bookkeeping
# ============================================================================


def _merge_suspicions(mind, new_items: list[dict], phase: str) -> int:
    """Add or reinforce suspicions in the mind. Returns count of new items."""
    new_count = 0
    for item in new_items:
        existing = next(
            (s for s in mind.suspicions
             if s.get("about_power") == item["about_power"]
             and s.get("tag") == item["tag"]),
            None
        )
        if existing:
            # Reinforce: bump weight, update last_seen, replace note
            existing["weight"] = min(1.0, existing.get("weight", 0.5) + 0.2)
            existing["last_seen_phase"] = phase
            existing["note"] = item["note"]  # latest note wins
        else:
            mind.suspicions.append({
                "about_power": item["about_power"],
                "tag": item["tag"],
                "note": item["note"],
                "first_seen_phase": phase,
                "last_seen_phase": phase,
                "weight": 0.5,
            })
            new_count += 1
    # Decay: drop very-stale suspicions (haven't been seen in 6+ phases)
    # Keeping it simple — just trim the list to the most recent 24
    if len(mind.suspicions) > 24:
        mind.suspicions = mind.suspicions[-24:]
    return new_count


# ============================================================================
# Top-level: write a journal entry
# ============================================================================


def write_journal_entry(
    agent,
    *,
    board_summary_text: str,
    what_just_happened: str,
    phase: str,
    kind: str = "post-resolution",
) -> Optional[dict]:
    """Make ONE LLM call to produce a private journal entry, parse the
    result, and append to mind.private_journal. Returns the entry dict
    or None on failure.

    `agent` is a DiplomacyAgentV2 (use bridge_agent.v2 for V2BridgeAgent).
    Failures are logged but don't raise — journals are best-effort.
    """
    try:
        prompt = compose_journal_prompt(
            mind=agent.mind,
            board_summary_text=board_summary_text,
            what_just_happened=what_just_happened,
            phase=phase, kind=kind,
        )
        response = agent.llm_call(prompt)
        if not response:
            return None
        suspicions_new = _parse_suspicions_block(response)
        text = _strip_suspicions_block(response)
        # Cap entry length to keep the prompt context manageable
        if len(text) > 800:
            text = text[:800] + "…"
        entry = {
            "phase": phase,
            "kind": kind,
            "timestamp": time.time(),
            "text": text,
        }
        agent.mind.private_journal.append(entry)
        # Cap journal length
        if len(agent.mind.private_journal) > 60:
            agent.mind.private_journal = agent.mind.private_journal[-60:]
        if suspicions_new:
            _merge_suspicions(agent.mind, suspicions_new, phase)
        return entry
    except Exception as e:
        print(f"[journal] write failed for {agent.power}: {e}")
        return None


def render_journal_for_prompt(mind, n: int = 3) -> str:
    """Render the most recent N journal entries for inclusion in a
    fovea prompt under 'YOUR RECENT PRIVATE THOUGHTS:'."""
    if not mind.private_journal:
        return ""
    recent = mind.private_journal[-n:]
    lines = []
    for e in recent:
        phase = e.get("phase", "")
        text = e.get("text", "").strip()
        if text:
            lines.append(f"({phase}) {text}")
    return "\n".join(lines)


def render_suspicions_for_prompt(mind, max_per_power: int = 1) -> str:
    """Render the top suspicion per power for inclusion in a fovea prompt."""
    if not mind.suspicions:
        return ""
    by_power: dict[str, list[dict]] = {}
    for s in mind.suspicions:
        by_power.setdefault(s.get("about_power", "?"), []).append(s)
    out = []
    for p, items in by_power.items():
        items.sort(key=lambda s: -s.get("weight", 0.0))
        for s in items[:max_per_power]:
            note = s.get("note", "")
            tag = s.get("tag", "")
            if note:
                out.append(f"{p} [{tag}]: {note}")
    return "\n".join(out)
