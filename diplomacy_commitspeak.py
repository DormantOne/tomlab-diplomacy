"""
diplomacy_commitspeak.py — parser for the commitspeak structured tail.

A press message MAY end with a [[commit ...]] block. The parser:
  1. Locates the block (if any).
  2. Splits it into individual lines.
  3. Parses each line into a CommitspeakLine (well-formed) or a
     CommitspeakLine carrying parse_errors (malformed).
  4. Returns a CommitspeakBlock that the caller can convert into
     CommitmentNode / SelfCommitmentNode entries.

DESIGN NOTES:
  - Strict on structure (the line types are a closed enum), lenient on
    whitespace and case so a small model isn't punished for
    minor formatting drift.
  - Validates province codes and power names against engine-provided sets,
    passed in as parameters. The parser doesn't import the engine — it
    accepts the validation universes as arguments.
  - Returns malformed lines INSIDE the parsed block rather than throwing,
    so the caller can decide whether to (a) log them for the next prompt's
    grammar reminder, (b) skip them, or (c) reject the whole block.
  - Phase canonicalization: accepts "1902-FALL-MOVES" or "F1902" or
    "fall 1902 moves" and produces canonical "1902-FALL-MOVES".
  - "through 1903" → canonicalizes to "1903-WINTER-ADJUSTMENTS" (end of year).
  - Conditional handling: a top-level `conditional:` line in the block applies
    to ALL commitments in the block (per the grammar in the schema).

NON-GOALS:
  - We do not validate semantic plausibility (e.g. "the unit being
    promised about is owned by the speaker"). That's the grader's job.
"""

from __future__ import annotations

import re
from typing import Optional

from diplomacy_kg_schema import (
    CommitmentType, CommitspeakLine, CommitspeakBlock,
    PhaseKey, ProvinceCode, PowerName,
)


# ============================================================================
# Block extraction
# ============================================================================

# The block delimiters. We accept a few minor variations a small model might
# emit: [[commit, [[ commit, [[COMMIT, etc. Closing is ]] always.
_BLOCK_RE = re.compile(
    r"\[\[\s*commit\s*\n(.*?)\n\s*\]\]",
    re.IGNORECASE | re.DOTALL,
)


def extract_block(message_text: str) -> Optional[str]:
    """Return the raw inside of the [[commit ...]] block if one exists.

    A message may have at most one block. If multiple are present
    (probably a model error), we take the LAST one — that's the most
    likely to be the model's "final" answer rather than an example
    earlier in its output.
    """
    matches = _BLOCK_RE.findall(message_text)
    if not matches:
        return None
    return matches[-1].strip()


# ============================================================================
# Phase canonicalization
# ============================================================================

_SEASONS = {"S": "SPRING", "F": "FALL", "W": "WINTER",
            "SPRING": "SPRING", "FALL": "FALL", "WINTER": "WINTER"}
_PHASES = {"M": "MOVES", "R": "RETREATS", "A": "ADJUSTMENTS",
           "MOVES": "MOVES", "RETREATS": "RETREATS",
           "ADJUSTMENTS": "ADJUSTMENTS", "ADJ": "ADJUSTMENTS"}

# canonical formats: <YEAR>-<SEASON>-<PHASE>
_CANONICAL_RE = re.compile(r"^\s*(\d{4})-(SPRING|FALL|WINTER)-(MOVES|RETREATS|ADJUSTMENTS)\s*$",
                           re.IGNORECASE)
# short formats: F1902, S1903, W1904
_SHORT_RE = re.compile(r"^\s*([SFWsfw])\s*(\d{4})\s*$")
# loose form: "fall 1902 moves" / "spring 1903" / etc.
_LOOSE_RE = re.compile(
    r"^\s*(spring|fall|winter|s|f|w)\s+(\d{4})(?:\s+(moves|retreats|adjustments|m|r|a))?\s*$",
    re.IGNORECASE,
)


def canonicalize_phase(s: str) -> Optional[PhaseKey]:
    """Try to canonicalize a phase string to YYYY-SEASON-PHASE.

    Returns None if it can't be parsed.
    """
    if not s:
        return None
    s = s.strip()

    m = _CANONICAL_RE.match(s)
    if m:
        year, season, phase = m.groups()
        return f"{year}-{season.upper()}-{phase.upper()}"

    m = _SHORT_RE.match(s)
    if m:
        s_letter, year = m.groups()
        season = _SEASONS.get(s_letter.upper())
        if season is None:
            return None
        # short form defaults to MOVES (most common); WINTER defaults to ADJ
        default_phase = "ADJUSTMENTS" if season == "WINTER" else "MOVES"
        return f"{year}-{season}-{default_phase}"

    m = _LOOSE_RE.match(s)
    if m:
        season_raw, year, phase_raw = m.groups()
        season = _SEASONS.get(season_raw.upper())
        if season is None:
            return None
        if phase_raw:
            phase = _PHASES.get(phase_raw.upper())
            if phase is None:
                return None
        else:
            phase = "ADJUSTMENTS" if season == "WINTER" else "MOVES"
        return f"{year}-{season}-{phase}"

    # "through 1903" — end-of-year canonical
    through = re.match(r"^\s*through\s+(\d{4})\s*$", s, re.IGNORECASE)
    if through:
        return f"{through.group(1)}-WINTER-ADJUSTMENTS"

    return None


# ============================================================================
# Line parsing
# ============================================================================

# Each line type has its own regex, matched in priority order. The regexes
# are tolerant of extra whitespace and accept either "by <phase>" or just
# the phase at the end of the line.

_UNIT_RE = re.compile(r"\b([AFaf])\s+([A-Za-z]{3})\b")
_PROV_RE = re.compile(r"\b([A-Za-z]{3})\b")


def _parse_unit(s: str) -> Optional[str]:
    m = _UNIT_RE.search(s)
    if not m:
        return None
    return f"{m.group(1).upper()} {m.group(2).upper()}"


def _split_main_and_by(line_body: str) -> tuple[str, Optional[str]]:
    """Pull out an optional `by <phase>` clause."""
    m = re.search(r"\bby\b(.+)$", line_body, re.IGNORECASE)
    if m:
        head = line_body[:m.start()].rstrip(", ").strip()
        tail = m.group(1).strip()
        return head, tail
    return line_body.strip(), None


def _parse_move_to(body: str, line: CommitspeakLine,
                   provinces: set[ProvinceCode]) -> None:
    """`move_to: A WAR -> GAL by 1902-FALL-MOVES`"""
    main, by_part = _split_main_and_by(body)
    # Expect "<unit> -> <prov>"
    m = re.search(r"^\s*([AFaf]\s+[A-Za-z]{3})\s*(?:->|to|→)\s*([A-Za-z]{3})\s*$", main)
    if not m:
        line.parse_errors.append("move_to: expected '<unit> -> <prov>'")
        return
    unit = m.group(1).upper().replace("  ", " ")
    target = m.group(2).upper()
    if target not in provinces:
        line.parse_errors.append(f"move_to: unknown province {target!r}")
        return
    line.subject_unit = unit
    line.subject_province = unit.split()[1]
    line.target_province = target
    if by_part:
        ph = canonicalize_phase(by_part)
        if ph is None:
            line.parse_errors.append(f"move_to: cannot canonicalize phase {by_part!r}")
        else:
            line.deadline_phase = ph
    else:
        line.parse_errors.append("move_to: missing 'by <phase>'")


def _parse_not_move_to(body: str, line: CommitspeakLine,
                       provinces: set[ProvinceCode]) -> None:
    """`not_move_to: BLA by 1902-FALL-MOVES`"""
    main, by_part = _split_main_and_by(body)
    pm = _PROV_RE.match(main.strip())
    if not pm:
        line.parse_errors.append("not_move_to: expected '<prov>'")
        return
    prov = pm.group(1).upper()
    if prov not in provinces:
        line.parse_errors.append(f"not_move_to: unknown province {prov!r}")
        return
    line.target_province = prov
    if by_part:
        ph = canonicalize_phase(by_part)
        if ph is None:
            line.parse_errors.append(f"not_move_to: cannot canonicalize phase {by_part!r}")
        else:
            line.deadline_phase = ph
    else:
        line.parse_errors.append("not_move_to: missing 'by <phase>'")


def _parse_hold_at(body: str, line: CommitspeakLine,
                   provinces: set[ProvinceCode]) -> None:
    """`hold_at: A WAR by 1902-FALL-MOVES` or `hold_at: WAR by ...`"""
    main, by_part = _split_main_and_by(body)
    unit = _parse_unit(main)
    if unit is None:
        # Maybe just a province
        pm = _PROV_RE.match(main.strip())
        if not pm:
            line.parse_errors.append("hold_at: expected '<unit>' or '<prov>'")
            return
        prov = pm.group(1).upper()
        line.subject_province = prov
    else:
        line.subject_unit = unit
        line.subject_province = unit.split()[1]
    if line.subject_province and line.subject_province not in provinces:
        line.parse_errors.append(f"hold_at: unknown province {line.subject_province!r}")
        return
    if by_part:
        ph = canonicalize_phase(by_part)
        if ph is None:
            line.parse_errors.append(f"hold_at: cannot canonicalize phase {by_part!r}")
        else:
            line.deadline_phase = ph
    else:
        line.parse_errors.append("hold_at: missing 'by <phase>'")


def _parse_support(body: str, line: CommitspeakLine,
                   provinces: set[ProvinceCode]) -> None:
    """`support: A MUN S A KIE -> BER by 1902-FALL-MOVES`
    or  `support: A PAR S A MAR by 1902-FALL-MOVES`  (support a hold)"""
    main, by_part = _split_main_and_by(body)
    # With move target
    m = re.search(
        r"^\s*([AFaf]\s+[A-Za-z]{3})\s+S\s+([AFaf]\s+[A-Za-z]{3})\s*(?:->|to|→)\s*([A-Za-z]{3})\s*$",
        main, re.IGNORECASE,
    )
    if m:
        sup_unit = m.group(1).upper().replace("  ", " ")
        line.subject_unit = sup_unit
        line.subject_province = sup_unit.split()[1]
        line.target_province = m.group(3).upper()
        if line.target_province not in provinces:
            line.parse_errors.append(f"support: unknown target province {line.target_province!r}")
            return
    else:
        # Support-a-hold form
        m2 = re.search(
            r"^\s*([AFaf]\s+[A-Za-z]{3})\s+S\s+([AFaf]\s+[A-Za-z]{3})\s*$",
            main, re.IGNORECASE,
        )
        if not m2:
            line.parse_errors.append(
                "support: expected '<unit> S <unit> -> <prov>' or '<unit> S <unit>'"
            )
            return
        sup_unit = m2.group(1).upper().replace("  ", " ")
        line.subject_unit = sup_unit
        line.subject_province = sup_unit.split()[1]
        line.target_province = m2.group(2).upper().split()[1]   # province of supported unit

    if by_part:
        ph = canonicalize_phase(by_part)
        if ph is None:
            line.parse_errors.append(f"support: cannot canonicalize phase {by_part!r}")
        else:
            line.deadline_phase = ph
    else:
        line.parse_errors.append("support: missing 'by <phase>'")


def _parse_non_aggression(body: str, line: CommitspeakLine,
                          powers: set[PowerName]) -> None:
    """`non_aggression: with FRANCE through 1903`
    or  `non_aggression: with FRANCE`  (defaults to current year fall)"""
    m = re.match(
        r"^\s*with\s+([A-Za-z]+)(?:\s+(through\s+\d{4}|by\s+.+))?\s*$",
        body.strip(), re.IGNORECASE,
    )
    if not m:
        line.parse_errors.append("non_aggression: expected 'with <power> [through <year>]'")
        return
    power_raw = m.group(1).upper()
    if power_raw not in powers:
        line.parse_errors.append(f"non_aggression: unknown power {power_raw!r}")
        return
    line.counterparty = power_raw
    when = m.group(2)
    if when:
        ph = canonicalize_phase(when.replace("by ", "").strip())
        if ph is None:
            line.parse_errors.append(f"non_aggression: cannot canonicalize {when!r}")
        else:
            line.deadline_phase = ph
    else:
        line.parse_errors.append(
            "non_aggression: missing window — use 'through <year>' or 'by <phase>'"
        )


def _parse_demilitarize(body: str, line: CommitspeakLine,
                        provinces: set[ProvinceCode],
                        powers: set[PowerName]) -> None:
    """`demilitarize: GAL with FRANCE through 1903`"""
    m = re.match(
        r"^\s*([A-Za-z]{3})\s+with\s+([A-Za-z]+)\s+(through\s+\d{4}|by\s+.+)\s*$",
        body.strip(), re.IGNORECASE,
    )
    if not m:
        line.parse_errors.append(
            "demilitarize: expected '<prov> with <power> through <year>'"
        )
        return
    prov = m.group(1).upper()
    power = m.group(2).upper()
    when = m.group(3)
    if prov not in provinces:
        line.parse_errors.append(f"demilitarize: unknown province {prov!r}")
        return
    if power not in powers:
        line.parse_errors.append(f"demilitarize: unknown power {power!r}")
        return
    line.subject_province = prov
    line.counterparty = power
    ph = canonicalize_phase(when.replace("by ", "").strip())
    if ph is None:
        line.parse_errors.append(f"demilitarize: cannot canonicalize {when!r}")
    else:
        line.deadline_phase = ph


def _parse_alliance_for(body: str, line: CommitspeakLine,
                        powers: set[PowerName]) -> None:
    """`alliance_for: with FRANCE [vs RUSSIA] through 1903`"""
    m = re.match(
        r"^\s*with\s+([A-Za-z]+)\s+(?:vs\s+([A-Za-z]+)\s+)?through\s+(\d{4})\s*$",
        body.strip(), re.IGNORECASE,
    )
    if not m:
        line.parse_errors.append(
            "alliance_for: expected 'with <power> [vs <power>] through <year>'"
        )
        return
    ally = m.group(1).upper()
    target = (m.group(2) or "").upper() or None
    year = m.group(3)
    if ally not in powers:
        line.parse_errors.append(f"alliance_for: unknown ally {ally!r}")
        return
    if target and target not in powers:
        line.parse_errors.append(f"alliance_for: unknown target power {target!r}")
        return
    line.counterparty = ally
    if target:
        line.subject_province = None    # repurpose: track target via deadline only;
                                        # the grader will know
    line.deadline_phase = f"{year}-WINTER-ADJUSTMENTS"


def _parse_build(body: str, line: CommitspeakLine,
                 provinces: set[ProvinceCode]) -> None:
    """`build: A TRI in W1902` or `build: F TRI in 1902`"""
    m = re.match(
        r"^\s*([AFaf])\s*([A-Za-z]{3})\s+in\s+(?:[Ww])?(\d{4})\s*$",
        body.strip(),
    )
    if not m:
        line.parse_errors.append("build: expected '<A|F> <prov> in W<year>'")
        return
    kind = m.group(1).upper()
    prov = m.group(2).upper()
    year = m.group(3)
    if prov not in provinces:
        line.parse_errors.append(f"build: unknown province {prov!r}")
        return
    line.subject_unit = f"{kind} {prov}"
    line.subject_province = prov
    line.deadline_phase = f"{year}-WINTER-ADJUSTMENTS"


def _parse_disband(body: str, line: CommitspeakLine,
                   provinces: set[ProvinceCode]) -> None:
    """`disband: A WAR in W1903`"""
    m = re.match(
        r"^\s*([AFaf]\s+[A-Za-z]{3})\s+in\s+(?:[Ww])?(\d{4})\s*$",
        body.strip(),
    )
    if not m:
        line.parse_errors.append("disband: expected '<A|F> <prov> in W<year>'")
        return
    unit = m.group(1).upper().replace("  ", " ")
    prov = unit.split()[1]
    if prov not in provinces:
        line.parse_errors.append(f"disband: unknown province {prov!r}")
        return
    line.subject_unit = unit
    line.subject_province = prov
    line.deadline_phase = f"{m.group(2)}-WINTER-ADJUSTMENTS"


# Dispatch table
_LINE_PARSERS = {
    CommitmentType.MOVE_TO:        ("move_to", _parse_move_to),
    CommitmentType.NOT_MOVE_TO:    ("not_move_to", _parse_not_move_to),
    CommitmentType.HOLD_AT:        ("hold_at", _parse_hold_at),
    CommitmentType.SUPPORT:        ("support", _parse_support),
    CommitmentType.NON_AGGRESSION: ("non_aggression", _parse_non_aggression),
    CommitmentType.DEMILITARIZE:   ("demilitarize", _parse_demilitarize),
    CommitmentType.ALLIANCE_FOR:   ("alliance_for", _parse_alliance_for),
    CommitmentType.BUILD:          ("build", _parse_build),
    CommitmentType.DISBAND:        ("disband", _parse_disband),
}

_LINE_HEAD_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*(.*?)\s*$", re.IGNORECASE)


def parse_commitspeak_line(
    raw: str,
    *,
    provinces: set[ProvinceCode],
    powers: set[PowerName],
) -> CommitspeakLine:
    """Parse a single `<type>: <body>` line.

    Returns a CommitspeakLine. If parsing failed, parse_errors is non-empty
    and `type` may be None.
    """
    line = CommitspeakLine(type=None, raw=raw.strip())
    head_match = _LINE_HEAD_RE.match(raw)
    if not head_match:
        line.parse_errors.append("malformed line: expected '<type>: <body>'")
        return line

    type_str = head_match.group(1).lower()
    body = head_match.group(2)

    # Resolve type
    type_to_cmt = {name: cmt for cmt, (name, _) in _LINE_PARSERS.items()}
    if type_str not in type_to_cmt:
        line.parse_errors.append(f"unknown commitment type: {type_str!r}")
        return line
    cmt_type = type_to_cmt[type_str]
    line.type = cmt_type

    # Dispatch to per-type parser. Several parsers need the provinces or
    # powers universe.
    name, parser = _LINE_PARSERS[cmt_type]
    sig = parser.__code__.co_varnames[:parser.__code__.co_argcount]
    if "powers" in sig and "provinces" in sig:
        parser(body, line, provinces, powers)
    elif "powers" in sig:
        parser(body, line, powers)
    elif "provinces" in sig:
        parser(body, line, provinces)
    else:
        parser(body, line)
    return line


# ============================================================================
# Block parsing — the public entry point
# ============================================================================

def parse_commitspeak_block(
    raw_block: str,
    *,
    provinces: set[ProvinceCode],
    powers: set[PowerName],
) -> CommitspeakBlock:
    """Parse the inside of a [[commit ... ]] block (without the delimiters).

    Returns a CommitspeakBlock with one CommitspeakLine per non-blank line.
    A leading `conditional:` line is captured as block.conditional and
    applied to every commitment when the caller converts lines to nodes.
    """
    block = CommitspeakBlock(raw_block=raw_block, lines=[])
    if not raw_block:
        return block

    lines = [ln.rstrip() for ln in raw_block.split("\n") if ln.strip()]

    for raw_line in lines:
        # Conditional handling
        cond_match = re.match(r"^\s*conditional\s*:\s*(.*?)\s*$",
                              raw_line, re.IGNORECASE)
        if cond_match:
            block.conditional = cond_match.group(1).strip()
            continue

        line = parse_commitspeak_line(
            raw_line, provinces=provinces, powers=powers,
        )
        block.lines.append(line)

    return block


def parse_message(
    message_text: str,
    *,
    provinces: set[ProvinceCode],
    powers: set[PowerName],
) -> Optional[CommitspeakBlock]:
    """Find and parse the commitspeak block in `message_text` if any.

    Returns None if no commitspeak block is present.
    Returns a CommitspeakBlock otherwise (possibly with parse errors on
    individual lines).
    """
    raw = extract_block(message_text)
    if raw is None:
        return None
    return parse_commitspeak_block(raw, provinces=provinces, powers=powers)


# ============================================================================
# Node construction
# ============================================================================
# Convert parsed lines to CommitmentNode / SelfCommitmentNode entries.

def commitspeak_lines_to_self_commitment_nodes(
    block: CommitspeakBlock,
    *,
    source_msg_id: str,
    speaker: PowerName,
    addressees: list[PowerName],
):
    """Convert well-formed lines into SelfCommitmentNode entries (this
    speaker's own promises). The unparseable ones are returned separately
    so the caller can log them and emit a grammar reminder next prompt.
    """
    from diplomacy_kg_schema import (
        SelfCommitmentNode, CommitmentStatus, new_id,
    )
    well_formed: list[SelfCommitmentNode] = []
    malformed: list[CommitspeakLine] = []
    for line in block.lines:
        if not line.parsed_ok:
            malformed.append(line)
            continue
        for addressee in (addressees or [None]):
            n = SelfCommitmentNode(
                id=new_id("cmt"),
                source_msg_id=source_msg_id,
                speaker=speaker,
                addressees=addressees or [],
                type=line.type,
                subject_unit=line.subject_unit,
                subject_province=line.subject_province,
                target_province=line.target_province,
                counterparty=line.counterparty,
                deadline_phase=line.deadline_phase,
                conditional_on=block.conditional,
                status=CommitmentStatus.PENDING,
                raw_commitspeak_line=line.raw,
                target_power=addressee,
            )
            well_formed.append(n)
            # If addressees is non-empty, we make one node per addressee.
            # If it's empty (public), one with target_power=None is enough.
            if addressee is None:
                break
    return well_formed, malformed


def commitspeak_lines_to_incoming_commitment_nodes(
    block: CommitspeakBlock,
    *,
    source_msg_id: str,
    speaker: PowerName,
    addressees: list[PowerName],
):
    """Convert well-formed lines into CommitmentNode entries (a promise
    that *speaker* made to others). Used by the recipient agent to log
    what was promised TO them."""
    from diplomacy_kg_schema import (
        CommitmentNode, CommitmentStatus, new_id,
    )
    well_formed: list[CommitmentNode] = []
    malformed: list[CommitspeakLine] = []
    for line in block.lines:
        if not line.parsed_ok:
            malformed.append(line)
            continue
        n = CommitmentNode(
            id=new_id("cmt"),
            source_msg_id=source_msg_id,
            speaker=speaker,
            addressees=list(addressees),
            type=line.type,
            subject_unit=line.subject_unit,
            subject_province=line.subject_province,
            target_province=line.target_province,
            counterparty=line.counterparty,
            deadline_phase=line.deadline_phase,
            conditional_on=block.conditional,
            status=CommitmentStatus.PENDING,
            raw_commitspeak_line=line.raw,
        )
        well_formed.append(n)
    return well_formed, malformed


# ============================================================================
# Sanity check + worked examples
# ============================================================================

if __name__ == "__main__":
    # Stand up a fake universe of provinces and powers for testing.
    PROVINCES = {
        "PAR", "MAR", "BRE", "BUR", "GAS", "PIC", "MUN", "BER", "RUH",
        "KIE", "SIL", "PRU", "WAR", "MOS", "STP", "SEV", "UKR", "GAL",
        "TYR", "TRI", "VIE", "BUD", "BOH", "ROM", "VEN", "NAP", "TUS",
        "BLA", "ION", "AEG", "EAS", "TYS", "LYO", "WES", "ENG", "MAO",
        "NTH", "NWG", "BAR", "BOT", "BAL", "SKA", "HEL", "NAF", "TUN",
        "POR", "SPA", "BEL", "HOL", "DEN", "SWE", "NWY", "FIN", "LVN",
        "ARM", "SYR", "SMY", "ANK", "CON", "BUL", "RUM", "SER", "ALB",
        "GRE",
    }
    POWERS = {"AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"}

    # ---- Worked example 1: a clean, well-formed message ----
    msg1 = """\
Sustained order serves us both. Vienna will not march on Munich, and I would
welcome a similar restraint from your side regarding the Tyrol.

[[commit
  not_move_to: MUN by 1902-FALL-MOVES
  non_aggression: with FRANCE through 1903
  conditional: only if you support A MAR holds
]]"""

    block = parse_message(msg1, provinces=PROVINCES, powers=POWERS)
    assert block is not None, "Expected a block in msg1"
    print("=" * 72)
    print("EXAMPLE 1: Clean message")
    print("=" * 72)
    print(f"Conditional: {block.conditional!r}")
    for line in block.lines:
        if line.parsed_ok:
            print(f"  OK  {line.type.value:<16} subj={line.subject_province or line.counterparty}"
                  f" tgt={line.target_province} by={line.deadline_phase}")
        else:
            print(f"  FAIL  raw={line.raw!r}  errors={line.parse_errors}")

    nodes, errs = commitspeak_lines_to_self_commitment_nodes(
        block, source_msg_id="msg:abc", speaker="AUSTRIA", addressees=["FRANCE"],
    )
    print(f"\n  → {len(nodes)} SelfCommitmentNodes, {len(errs)} malformed.")

    # ---- Worked example 2: a slightly messy message that still parses ----
    msg2 = """Tactical brief follows.

[[commit
move_to: A WAR -> GAL by F1902
support: A BUD S A WAR -> GAL by F1902
build: F TRI in W1902
]]"""

    block2 = parse_message(msg2, provinces=PROVINCES, powers=POWERS)
    assert block2 is not None
    print("\n" + "=" * 72)
    print("EXAMPLE 2: Looser whitespace + short phase form")
    print("=" * 72)
    for line in block2.lines:
        if line.parsed_ok:
            print(f"  OK  {line.type.value:<16} subj={line.subject_unit}"
                  f" tgt={line.target_province} by={line.deadline_phase}")
        else:
            print(f"  FAIL  raw={line.raw!r}  errors={line.parse_errors}")

    # ---- Worked example 3: malformed lines (typical small-model errors) ----
    msg3 = """[[commit
  move_to: A WAR to GAL by 1902-FALL-MOVES
  not_move_to: ZZZ by 1902-FALL-MOVES
  not_move_to: BUR
  destroy: A MOS in W1902
  non_aggression: with TURKEY through 1903
]]"""

    block3 = parse_message(msg3, provinces=PROVINCES, powers=POWERS)
    print("\n" + "=" * 72)
    print("EXAMPLE 3: Malformed lines mixed with valid ones")
    print("=" * 72)
    for line in block3.lines:
        if line.parsed_ok:
            print(f"  OK    {line.type.value:<16} :: {line.raw}")
        else:
            print(f"  FAIL  ({', '.join(line.parse_errors)}) :: {line.raw}")
    print(f"\n  Well-formed: {block3.well_formed_count} of {len(block3.lines)}")

    # ---- Worked example 4: no commitspeak block at all ----
    msg4 = "I will not move to Munich. Trust me."
    block4 = parse_message(msg4, provinces=PROVINCES, powers=POWERS)
    print("\n" + "=" * 72)
    print("EXAMPLE 4: No commitspeak block — promise in prose, IGNORED")
    print("=" * 72)
    print(f"  Result: {block4!r}")
    print("  → Per design, prose promises are not tracked.")
