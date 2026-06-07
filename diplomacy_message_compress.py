"""
diplomacy_message_compress.py — compress recent_messages section to commitspeak.

Step 1 of the "context focus" iteration. Tests the signal:noise hypothesis:
that Haiku is being distracted by ~1000 chars of prose negotiation framing
in the recent_messages block, drowning out the substrate fovea content
that actually matters for play.

Strategy
--------
The agent's `_recent_messages_text` helper renders the last 6 messages
with full prose bodies (truncated at 200 chars each). For most diplomatic
messages, the strategically-meaningful content is in the `[[commit ...]]`
block at the end — typed promises like `not_move_to: GAL by F1902`. The
prose framing ("Gentlemen, Russia gathers strength...") is social padding
that contains no novel structured information.

This module monkey-patches `DiplomacyAgentV2._recent_messages_text` to
emit a compressed form: one line per message containing only the
commitspeak content. Prose-only messages (no `[[commit]]` block) get a
short body preview so they're not lost entirely.

Before / after on a typical message
-----------------------------------

Before (~200 chars):
  RUSSIA -> ENGLAND,FRANCE: Gentlemen, Russia gathers strength while we
  deliberate. I propose we three acknowledge spheres—Mediterranean and
  Atlantic yours, the Black Sea and Eastern waters mine. Stability through
  respect. [[commit   non_aggression: with ENGLAND through 1902 ...

After (~80 chars):
  RUSSIA → ENG,FRA: non_aggression with ENGLAND through 1902 | non_aggression with FRANCE through 1902

Public surface
--------------
  enable_compressed_messages()    — opt-in monkey-patch
  disable_compressed_messages()   — undo
  is_compression_active()         — query state
  compressed_recent_messages_text(recent)  — pure function (testable)
"""

from __future__ import annotations

import re
from typing import Optional


# ============================================================================
# Commitspeak parsing
# ============================================================================


_COMMIT_BLOCK_RE = re.compile(
    r"\[\[commit\s*\n?(.*?)\n?\s*\]\]", re.DOTALL,
)


def _extract_commit_lines_from_tail(tail: str) -> list[str]:
    """Parse a commitspeak_tail field. May be either form:
      (a) the [[commit ... ]] wrapper, or
      (b) just the inside of the wrapper.
    Returns the cleaned commit verb lines.
    """
    if not tail:
        return []
    if "[[commit" in tail:
        matches = _COMMIT_BLOCK_RE.findall(tail)
        if not matches:
            return []
        inside = "\n".join(matches)
    else:
        # Tail is the pre-extracted inside.
        inside = tail.strip()
    out = []
    for raw in inside.split("\n"):
        ln = raw.strip()
        if not ln or ln.startswith("[[") or ln.startswith("]]"):
            continue
        out.append(ln)
    return out


def _extract_commit_lines_from_body(body: str) -> list[str]:
    """Parse commit lines from a message body. REQUIRES [[commit ... ]]
    markers to be present. Won't try to interpret arbitrary prose as
    commit lines.
    """
    if not body or "[[commit" not in body:
        return []
    matches = _COMMIT_BLOCK_RE.findall(body)
    if not matches:
        return []
    inside = "\n".join(matches)
    out = []
    for raw in inside.split("\n"):
        ln = raw.strip()
        if not ln or ln.startswith("[[") or ln.startswith("]]"):
            continue
        out.append(ln)
    return out


def _strip_commit_block(body: str) -> str:
    """Return body with the [[commit ...]] block(s) removed."""
    return _COMMIT_BLOCK_RE.sub("", body or "").strip()


# ============================================================================
# Per-message compression
# ============================================================================


_PROSE_PREVIEW_CHARS = 50  # length of the prose snippet for no-commit msgs


def _compress_one_message(m) -> str:
    """One compressed line for a single MessageEvent (or legacy Message).

    Format:
      SENDER → RECIPIENTS: <commit lines joined by '|'>          [if commits]
      SENDER → RECIPIENTS: "<short prose>" (no commit)           [otherwise]
    """
    sender = getattr(m, "sender", "?")
    public = getattr(m, "public", False)
    if public:
        target = "ALL"
    else:
        recips = getattr(m, "recipients", []) or []
        # Shorten recipient names to first 3 chars for readability when
        # there are multiple. Single recipient stays full-length.
        if len(recips) == 1:
            target = recips[0]
        elif recips:
            target = ",".join(r[:3] for r in recips)
        else:
            target = "?"

    # Try commitspeak_tail first (newer MessageEvent), fall back to parsing
    # the body. Both can co-exist in different code paths.
    commit_lines: list[str] = []
    tail = getattr(m, "commitspeak_tail", None)
    if tail:
        commit_lines = _extract_commit_lines_from_tail(tail)
    if not commit_lines:
        body = getattr(m, "body", None) or getattr(m, "text", "")
        commit_lines = _extract_commit_lines_from_body(body)

    if commit_lines:
        body_str = " | ".join(commit_lines)
        return f"  {sender} → {target}: {body_str}"

    # No parseable commits — show a short prose preview so the message
    # isn't dropped entirely. Public stance announcements often live here.
    body = getattr(m, "body", None) or getattr(m, "text", "")
    prose = _strip_commit_block(body).replace("\n", " ").strip()
    if len(prose) > _PROSE_PREVIEW_CHARS:
        prose = prose[: _PROSE_PREVIEW_CHARS - 3] + "..."
    if not prose:
        return f"  {sender} → {target}: (empty)"
    return f'  {sender} → {target}: "{prose}" (no commit)'


# ============================================================================
# Full block builder — the replacement for _recent_messages_text
# ============================================================================


def compressed_recent_messages_text(recent: list, cap: int = 6) -> str:
    """Pure function: list[MessageEvent] → compressed text block.

    Same structure as the original _recent_messages_text:
      - Header line
      - Optional summary of older-than-cap messages (counts by sender)
      - Last `cap` messages in compressed form

    Bound is 6 by default to match the existing helper's default; pass cap
    higher if you want more of the recent log surfaced.
    """
    if not recent:
        return "Recent messages: (none)"
    last_n = recent[-cap:]
    older = recent[:-cap] if len(recent) > cap else []
    lines = ["Recent messages (compressed — commitspeak + previews):"]
    if older:
        counts: dict[str, int] = {}
        for m in older:
            sender = getattr(m, "sender", "?")
            counts[sender] = counts.get(sender, 0) + 1
        counts_str = ", ".join(f"{s}={n}" for s, n in counts.items())
        lines.append(f"  (older: {len(older)} msgs — {counts_str})")
    for m in last_n:
        lines.append(_compress_one_message(m))
    return "\n".join(lines)


# ============================================================================
# Activation — opt-in monkey-patch
# ============================================================================


_PATCHED = False
_ORIGINAL_HELPER = None


def enable_compressed_messages() -> None:
    """Monkey-patch DiplomacyAgentV2._recent_messages_text.

    Idempotent. Reversible via disable_compressed_messages(). Applies
    process-wide — every agent in the current Python process uses the
    compressed form after this call.

    Note: this affects MutableAgent too (it's a subclass). The mute
    machinery passes recent_messages through this helper unchanged, so
    muted and unmuted agents both get compressed messages. That keeps the
    A/B comparison clean: the only difference between them remains the
    fovea content, not the recent_messages content.
    """
    global _PATCHED, _ORIGINAL_HELPER
    if _PATCHED:
        return
    import diplomacy_agent_v2 as _av2
    _ORIGINAL_HELPER = _av2.DiplomacyAgentV2._recent_messages_text

    def _new_helper(self, recent):
        return compressed_recent_messages_text(recent)

    _av2.DiplomacyAgentV2._recent_messages_text = _new_helper
    _PATCHED = True


def disable_compressed_messages() -> None:
    """Restore the original _recent_messages_text."""
    global _PATCHED, _ORIGINAL_HELPER
    if not _PATCHED:
        return
    import diplomacy_agent_v2 as _av2
    _av2.DiplomacyAgentV2._recent_messages_text = _ORIGINAL_HELPER
    _ORIGINAL_HELPER = None
    _PATCHED = False


def is_compression_active() -> bool:
    return _PATCHED


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    import time as _t
    from diplomacy_kg_schema import MessageEvent, new_id

    print("=" * 72)
    print("MESSAGE COMPRESSION SANITY CHECK")
    print("=" * 72)

    # ---- Build representative messages ----
    msgs = []

    # 1. Long prose with multi-line commitspeak (the common case)
    msg1 = MessageEvent(
        id=new_id("msg"), phase="1902-SPRING-MOVES",
        sender="RUSSIA", recipients=["ENGLAND", "FRANCE"], public=False,
        body=(
            "Gentlemen, Russia gathers strength while we deliberate. "
            "I propose we three acknowledge spheres—Mediterranean and "
            "Atlantic yours, the Black Sea and Eastern waters mine. "
            "Stability through respect.\n"
            "[[commit\n"
            "  non_aggression: with ENGLAND through 1902\n"
            "  non_aggression: with FRANCE through 1902\n"
            "]]"
        ),
        commitspeak_tail=(
            "[[commit\n"
            "  non_aggression: with ENGLAND through 1902\n"
            "  non_aggression: with FRANCE through 1902\n"
            "]]"
        ),
        sent_at=_t.time(),
    )
    msgs.append(msg1)

    # 2. Public statement with no commitspeak (the prose-only case)
    msg2 = MessageEvent(
        id=new_id("msg"), phase="1902-SPRING-MOVES",
        sender="GERMANY", recipients=[], public=True,
        body="The great powers need not clash. I extend open hands to those "
             "who honor borders. Europe prospers through restraint, not conquest.",
        commitspeak_tail=None,
        sent_at=_t.time() + 1,
    )
    msgs.append(msg2)

    # 3. Bilateral with single commit
    msg3 = MessageEvent(
        id=new_id("msg"), phase="1902-SPRING-MOVES",
        sender="TURKEY", recipients=["AUSTRIA"], public=False,
        body=(
            "Austria, stability favors both of us through this season.\n"
            "[[commit\n"
            "  not_move_to: TRI by 1902-FALL-MOVES\n"
            "]]"
        ),
        commitspeak_tail=(
            "[[commit\n  not_move_to: TRI by 1902-FALL-MOVES\n]]"
        ),
        sent_at=_t.time() + 2,
    )
    msgs.append(msg3)

    # 4. Empty body edge case
    msg4 = MessageEvent(
        id=new_id("msg"), phase="1902-SPRING-MOVES",
        sender="AUSTRIA", recipients=["RUSSIA"], public=False,
        body="", commitspeak_tail=None, sent_at=_t.time() + 3,
    )
    msgs.append(msg4)

    # ---- Test the compressor ----
    print("\n  COMPRESSED OUTPUT:")
    print("  " + "-" * 70)
    out = compressed_recent_messages_text(msgs)
    for ln in out.split("\n"):
        print(f"  {ln}")
    print("  " + "-" * 70)

    assert "non_aggression: with ENGLAND through 1902" in out
    assert "non_aggression: with FRANCE through 1902" in out
    assert "non_aggression: with ENGLAND" in out and \
           "non_aggression: with FRANCE" in out and \
           " | " in out, "multi-commit message must use ' | ' separator"
    assert "no commit" in out, "prose-only message must show preview"
    assert "not_move_to: TRI by 1902-FALL-MOVES" in out
    assert "TURKEY → AUSTRIA" in out
    assert "RUSSIA → ENG,FRA" in out, "multi-recipient should compact to 3-char"
    assert "GERMANY → ALL" in out, "public should render as ALL"

    # ---- Compare lengths ----
    # Reproduce the v1 helper inline so we have a fair before/after
    def _v1_helper(recent: list) -> str:
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

    v1_text = _v1_helper(msgs)
    saved = len(v1_text) - len(out)
    print(f"\n  v1 helper output: {len(v1_text)} chars")
    print(f"  compressed:       {len(out)} chars  "
          f"(saved {saved} chars, {100*saved//len(v1_text)}%)")
    assert len(out) < len(v1_text) - 100, \
        f"compression should save >100 chars; saved {saved}"

    # ---- Larger sample matching real game density ----
    big = []
    for i in range(20):
        sender = ["RUSSIA", "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "TURKEY"][i % 6]
        recips = [["ENGLAND"], ["FRANCE"], ["AUSTRIA", "GERMANY"]][i % 3]
        big.append(MessageEvent(
            id=new_id("msg"), phase="1902-SPRING-MOVES",
            sender=sender, recipients=recips, public=False,
            body=(
                f"Routine diplomatic message of moderate length from {sender}. "
                f"Discussing mutual interests, conditions, and proposed coordination "
                f"with appropriate diplomatic phrasing.\n"
                f"[[commit\n"
                f"  non_aggression: with {recips[0]} through 1902\n"
                f"]]"
            ),
            commitspeak_tail=(
                f"[[commit\n  non_aggression: with {recips[0]} through 1902\n]]"
            ),
            sent_at=_t.time() + i,
        ))
    v1_big = _v1_helper(big)
    c_big = compressed_recent_messages_text(big)
    print(f"\n  Realistic 20-msg log:")
    print(f"    v1: {len(v1_big)} chars")
    print(f"    v2: {len(c_big)} chars  "
          f"(saved {len(v1_big) - len(c_big)} chars, "
          f"{100*(len(v1_big) - len(c_big))//len(v1_big)}%)")

    # ---- enable / disable ----
    print(f"\n  Testing enable/disable...")
    import diplomacy_agent_v2 as _av2
    original = _av2.DiplomacyAgentV2._recent_messages_text
    enable_compressed_messages()
    assert _av2.DiplomacyAgentV2._recent_messages_text is not original
    assert is_compression_active()
    # Idempotent
    enable_compressed_messages()
    assert is_compression_active()
    disable_compressed_messages()
    assert _av2.DiplomacyAgentV2._recent_messages_text is original
    assert not is_compression_active()
    # Idempotent
    disable_compressed_messages()
    print(f"  enable/disable: idempotent + reversible")

    print()
    print("Message compression sanity check passed.")
