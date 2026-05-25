"""
Standard Diplomacy map data (simplified — split coasts collapsed).

Province codes follow the conventional 3-letter abbreviations.
Adjacency is a single graph; we tag whether a province is land/sea/coast and
whether armies / fleets can occupy each.
"""

LAND, SEA, COAST = "land", "sea", "coast"

# province -> (kind, full_name, is_supply_center)
PROVINCES = {
    # --- Sea spaces ---
    "ADR": (SEA, "Adriatic Sea", False),
    "AEG": (SEA, "Aegean Sea", False),
    "BAL": (SEA, "Baltic Sea", False),
    "BAR": (SEA, "Barents Sea", False),
    "BLA": (SEA, "Black Sea", False),
    "EAS": (SEA, "Eastern Mediterranean", False),
    "ENG": (SEA, "English Channel", False),
    "BOT": (SEA, "Gulf of Bothnia", False),
    "GOL": (SEA, "Gulf of Lyon", False),
    "HEL": (SEA, "Helgoland Bight", False),
    "ION": (SEA, "Ionian Sea", False),
    "IRI": (SEA, "Irish Sea", False),
    "MAO": (SEA, "Mid-Atlantic Ocean", False),
    "NAO": (SEA, "North Atlantic Ocean", False),
    "NTH": (SEA, "North Sea", False),
    "NWG": (SEA, "Norwegian Sea", False),
    "SKA": (SEA, "Skagerrak", False),
    "TYS": (SEA, "Tyrrhenian Sea", False),
    "WES": (SEA, "Western Mediterranean", False),

    # --- Inland (land-locked) ---
    "BOH": (LAND, "Bohemia", False),
    "BUR": (LAND, "Burgundy", False),
    "GAL": (LAND, "Galicia", False),
    "MOS": (LAND, "Moscow", True),
    "MUN": (LAND, "Munich", True),
    "PAR": (LAND, "Paris", True),
    "RUH": (LAND, "Ruhr", False),
    "SER": (LAND, "Serbia", True),
    "SIL": (LAND, "Silesia", False),
    "TYR": (LAND, "Tyrolia", False),
    "UKR": (LAND, "Ukraine", False),
    "VIE": (LAND, "Vienna", True),
    "WAR": (LAND, "Warsaw", True),
    "BUD": (LAND, "Budapest", True),

    # --- Coastal ---
    "ALB": (COAST, "Albania", False),
    "ANK": (COAST, "Ankara", True),
    "APU": (COAST, "Apulia", False),
    "ARM": (COAST, "Armenia", False),
    "BEL": (COAST, "Belgium", True),
    "BER": (COAST, "Berlin", True),
    "BRE": (COAST, "Brest", True),
    "BUL": (COAST, "Bulgaria", True),
    "CLY": (COAST, "Clyde", False),
    "CON": (COAST, "Constantinople", True),
    "DEN": (COAST, "Denmark", True),
    "EDI": (COAST, "Edinburgh", True),
    "FIN": (COAST, "Finland", False),
    "GAS": (COAST, "Gascony", False),
    "GRE": (COAST, "Greece", True),
    "HOL": (COAST, "Holland", True),
    "KIE": (COAST, "Kiel", True),
    "LON": (COAST, "London", True),
    "LVN": (COAST, "Livonia", False),
    "LVP": (COAST, "Liverpool", True),
    "MAR": (COAST, "Marseilles", True),
    "NAF": (COAST, "North Africa", False),
    "NAP": (COAST, "Naples", True),
    "NWY": (COAST, "Norway", True),
    "PIC": (COAST, "Picardy", False),
    "PIE": (COAST, "Piedmont", False),
    "POR": (COAST, "Portugal", True),
    "PRU": (COAST, "Prussia", False),
    "ROM": (COAST, "Rome", True),
    "RUM": (COAST, "Rumania", True),
    "SEV": (COAST, "Sevastopol", True),
    "SMY": (COAST, "Smyrna", True),
    "SPA": (COAST, "Spain", True),
    "STP": (COAST, "St Petersburg", True),
    "SWE": (COAST, "Sweden", True),
    "SYR": (COAST, "Syria", False),
    "TRI": (COAST, "Trieste", True),
    "TUN": (COAST, "Tunis", True),
    "TUS": (COAST, "Tuscany", False),
    "VEN": (COAST, "Venice", True),
    "WAL": (COAST, "Wales", False),
    "YOR": (COAST, "Yorkshire", False),
}

# Adjacency. For coastal provinces we list both land and sea neighbors.
# Armies use land/coast edges; fleets use sea/coast edges.
# Encoded as: prov: { "army": [...], "fleet": [...] }
ADJ = {
    # Seas
    "ADR": {"army": [], "fleet": ["ALB", "APU", "ION", "TRI", "VEN"]},
    "AEG": {"army": [], "fleet": ["BUL", "CON", "EAS", "GRE", "ION", "SMY"]},
    "BAL": {"army": [], "fleet": ["BER", "BOT", "DEN", "KIE", "LVN", "PRU", "SWE"]},
    "BAR": {"army": [], "fleet": ["NWG", "NWY", "STP"]},
    "BLA": {"army": [], "fleet": ["ANK", "ARM", "BUL", "CON", "RUM", "SEV"]},
    "EAS": {"army": [], "fleet": ["AEG", "ION", "SMY", "SYR"]},
    "ENG": {"army": [], "fleet": ["BEL", "BRE", "IRI", "LON", "MAO", "NTH", "PIC", "WAL"]},
    "BOT": {"army": [], "fleet": ["BAL", "FIN", "LVN", "STP", "SWE"]},
    "GOL": {"army": [], "fleet": ["MAR", "PIE", "SPA", "TUS", "TYS", "WES"]},
    "HEL": {"army": [], "fleet": ["DEN", "HOL", "KIE", "NTH"]},
    "ION": {"army": [], "fleet": ["ADR", "AEG", "ALB", "APU", "EAS", "GRE", "NAP", "TUN", "TYS"]},
    "IRI": {"army": [], "fleet": ["ENG", "LVP", "MAO", "NAO", "WAL"]},
    "MAO": {"army": [], "fleet": ["BRE", "ENG", "GAS", "IRI", "NAF", "NAO", "POR", "SPA", "WES"]},
    "NAO": {"army": [], "fleet": ["CLY", "IRI", "LVP", "MAO", "NWG"]},
    "NTH": {"army": [], "fleet": ["BEL", "DEN", "EDI", "ENG", "HEL", "HOL", "LON", "NWG", "NWY", "SKA", "YOR"]},
    "NWG": {"army": [], "fleet": ["BAR", "CLY", "EDI", "NAO", "NTH", "NWY"]},
    "SKA": {"army": [], "fleet": ["DEN", "NTH", "NWY", "SWE"]},
    "TYS": {"army": [], "fleet": ["GOL", "ION", "NAP", "ROM", "TUN", "TUS", "WES"]},
    "WES": {"army": [], "fleet": ["GOL", "MAO", "NAF", "SPA", "TUN", "TYS"]},

    # Land
    "BOH": {"army": ["GAL", "MUN", "SIL", "TYR", "VIE"], "fleet": []},
    "BUR": {"army": ["BEL", "GAS", "MAR", "MUN", "PAR", "PIC", "RUH"], "fleet": []},
    "GAL": {"army": ["BOH", "BUD", "RUM", "SIL", "UKR", "VIE", "WAR"], "fleet": []},
    "MOS": {"army": ["LVN", "SEV", "STP", "UKR", "WAR"], "fleet": []},
    "MUN": {"army": ["BER", "BOH", "BUR", "KIE", "RUH", "SIL", "TYR"], "fleet": []},
    "PAR": {"army": ["BRE", "BUR", "GAS", "PIC"], "fleet": []},
    "RUH": {"army": ["BEL", "BUR", "HOL", "KIE", "MUN"], "fleet": []},
    "SER": {"army": ["ALB", "BUD", "BUL", "GRE", "RUM", "TRI"], "fleet": []},
    "SIL": {"army": ["BER", "BOH", "GAL", "MUN", "PRU", "WAR"], "fleet": []},
    "TYR": {"army": ["BOH", "MUN", "PIE", "TRI", "VEN", "VIE"], "fleet": []},
    "UKR": {"army": ["GAL", "MOS", "RUM", "SEV", "WAR"], "fleet": []},
    "VIE": {"army": ["BOH", "BUD", "GAL", "TRI", "TYR"], "fleet": []},
    "WAR": {"army": ["GAL", "LVN", "MOS", "PRU", "SIL", "UKR"], "fleet": []},
    "BUD": {"army": ["GAL", "RUM", "SER", "TRI", "VIE"], "fleet": []},

    # Coastal
    "ALB": {"army": ["GRE", "SER", "TRI"], "fleet": ["ADR", "GRE", "ION", "TRI"]},
    "ANK": {"army": ["ARM", "CON", "SMY"], "fleet": ["ARM", "BLA", "CON"]},
    "APU": {"army": ["NAP", "ROM", "VEN"], "fleet": ["ADR", "ION", "NAP", "VEN"]},
    "ARM": {"army": ["ANK", "SMY", "SYR"], "fleet": ["ANK", "BLA", "SEV"]},
    "BEL": {"army": ["BUR", "HOL", "PIC", "RUH"], "fleet": ["ENG", "HOL", "NTH", "PIC"]},
    "BER": {"army": ["KIE", "MUN", "PRU", "SIL"], "fleet": ["BAL", "KIE", "PRU"]},
    "BRE": {"army": ["GAS", "PAR", "PIC"], "fleet": ["ENG", "GAS", "MAO", "PIC"]},
    "BUL": {"army": ["CON", "GRE", "RUM", "SER"], "fleet": ["AEG", "BLA", "CON", "GRE", "RUM"]},
    "CLY": {"army": ["EDI", "LVP"], "fleet": ["EDI", "LVP", "NAO", "NWG"]},
    "CON": {"army": ["ANK", "BUL", "SMY"], "fleet": ["AEG", "ANK", "BLA", "BUL", "SMY"]},
    "DEN": {"army": ["KIE", "SWE"], "fleet": ["BAL", "HEL", "KIE", "NTH", "SKA", "SWE"]},
    "EDI": {"army": ["CLY", "LVP", "YOR"], "fleet": ["CLY", "NTH", "NWG", "YOR"]},
    "FIN": {"army": ["NWY", "STP", "SWE"], "fleet": ["BOT", "STP", "SWE"]},
    "GAS": {"army": ["BRE", "BUR", "MAR", "PAR", "SPA"], "fleet": ["BRE", "MAO", "SPA"]},
    "GRE": {"army": ["ALB", "BUL", "SER"], "fleet": ["AEG", "ALB", "BUL", "ION"]},
    "HOL": {"army": ["BEL", "KIE", "RUH"], "fleet": ["BEL", "HEL", "KIE", "NTH"]},
    "KIE": {"army": ["BER", "DEN", "HOL", "MUN", "RUH"], "fleet": ["BAL", "BER", "DEN", "HEL", "HOL"]},
    "LON": {"army": ["WAL", "YOR"], "fleet": ["ENG", "NTH", "WAL", "YOR"]},
    "LVN": {"army": ["MOS", "PRU", "STP", "WAR"], "fleet": ["BAL", "BOT", "PRU", "STP"]},
    "LVP": {"army": ["CLY", "EDI", "WAL", "YOR"], "fleet": ["CLY", "IRI", "NAO", "WAL"]},
    "MAR": {"army": ["BUR", "GAS", "PIE", "SPA"], "fleet": ["GOL", "PIE", "SPA"]},
    "NAF": {"army": ["TUN"], "fleet": ["MAO", "TUN", "WES"]},
    "NAP": {"army": ["APU", "ROM"], "fleet": ["APU", "ION", "ROM", "TYS"]},
    "NWY": {"army": ["FIN", "STP", "SWE"], "fleet": ["BAR", "FIN", "NTH", "NWG", "SKA", "STP"]},
    "PIC": {"army": ["BEL", "BRE", "BUR", "PAR"], "fleet": ["BEL", "BRE", "ENG"]},
    "PIE": {"army": ["MAR", "TUS", "TYR", "VEN"], "fleet": ["GOL", "MAR", "TUS"]},
    "POR": {"army": ["SPA"], "fleet": ["MAO", "SPA"]},
    "PRU": {"army": ["BER", "LVN", "SIL", "WAR"], "fleet": ["BAL", "BER", "LVN"]},
    "ROM": {"army": ["APU", "NAP", "TUS", "VEN"], "fleet": ["NAP", "TUS", "TYS"]},
    "RUM": {"army": ["BUD", "BUL", "GAL", "SER", "SEV", "UKR"], "fleet": ["BLA", "BUL", "SEV"]},
    "SEV": {"army": ["ARM", "MOS", "RUM", "UKR"], "fleet": ["ARM", "BLA", "RUM"]},
    "SMY": {"army": ["ANK", "ARM", "CON", "SYR"], "fleet": ["AEG", "CON", "EAS", "SYR"]},
    "SPA": {"army": ["GAS", "MAR", "POR"], "fleet": ["GAS", "GOL", "MAO", "MAR", "POR", "WES"]},
    "STP": {"army": ["FIN", "LVN", "MOS", "NWY"], "fleet": ["BAR", "BOT", "FIN", "LVN", "NWY"]},
    "SWE": {"army": ["DEN", "FIN", "NWY"], "fleet": ["BAL", "BOT", "DEN", "FIN", "NWY", "SKA"]},
    "SYR": {"army": ["ARM", "SMY"], "fleet": ["EAS", "SMY"]},
    "TRI": {"army": ["ALB", "BUD", "SER", "TYR", "VEN", "VIE"], "fleet": ["ADR", "ALB", "VEN"]},
    "TUN": {"army": ["NAF"], "fleet": ["ION", "NAF", "TYS", "WES"]},
    "TUS": {"army": ["PIE", "ROM", "VEN"], "fleet": ["GOL", "PIE", "ROM", "TYS"]},
    "VEN": {"army": ["APU", "PIE", "ROM", "TRI", "TUS", "TYR"], "fleet": ["ADR", "APU", "TRI"]},
    "WAL": {"army": ["LON", "LVP", "YOR"], "fleet": ["ENG", "IRI", "LON", "LVP"]},
    "YOR": {"army": ["EDI", "LON", "LVP"], "fleet": ["EDI", "LON", "NTH"]},
}

# Powers and their starting positions.
# We use a 6-power variant: Italy is omitted (its home centers begin neutral).
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "RUSSIA", "TURKEY"]

# Home supply centers (where new units may be built)
HOME_CENTERS = {
    "AUSTRIA": ["VIE", "BUD", "TRI"],
    "ENGLAND": ["LON", "LVP", "EDI"],
    "FRANCE":  ["PAR", "MAR", "BRE"],
    "GERMANY": ["BER", "MUN", "KIE"],
    "RUSSIA":  ["MOS", "WAR", "SEV", "STP"],
    "TURKEY":  ["ANK", "CON", "SMY"],
}

# Starting units: list of (kind, province) where kind is "A" or "F"
STARTING_UNITS = {
    "AUSTRIA": [("A", "VIE"), ("A", "BUD"), ("F", "TRI")],
    "ENGLAND": [("F", "LON"), ("F", "EDI"), ("A", "LVP")],
    "FRANCE":  [("A", "PAR"), ("A", "MAR"), ("F", "BRE")],
    "GERMANY": [("A", "BER"), ("A", "MUN"), ("F", "KIE")],
    "RUSSIA":  [("A", "MOS"), ("A", "WAR"), ("F", "SEV"), ("F", "STP")],
    "TURKEY":  [("A", "CON"), ("A", "SMY"), ("F", "ANK")],
}

# All supply centers (for owner tracking)
ALL_SUPPLY_CENTERS = [p for p, info in PROVINCES.items() if info[2]]

# Supply centers that start neutral (everything not a home center).
NEUTRAL_CENTERS = [
    sc for sc in ALL_SUPPLY_CENTERS
    if not any(sc in homes for homes in HOME_CENTERS.values())
]


def is_adjacent(unit_kind: str, src: str, dst: str) -> bool:
    """Can a unit of kind ('A' or 'F') move from src to dst in one step?"""
    if src not in ADJ or dst not in ADJ:
        return False
    key = "army" if unit_kind == "A" else "fleet"
    return dst in ADJ[src][key]


def can_occupy(unit_kind: str, prov: str) -> bool:
    """Can a unit of given kind occupy this province?"""
    if prov not in PROVINCES:
        return False
    kind = PROVINCES[prov][0]
    if unit_kind == "A":
        return kind in (LAND, COAST)
    if unit_kind == "F":
        return kind in (SEA, COAST)
    return False
