"""
diplomacy_kg_analysis.py
========================

Read-only, helix-style analytics over the live Diplomacy minds. It does NOT
change how agents think — it reads the typed collections each AgentMind already
maintains (commitments, beliefs, predictions, suspicions) and projects them into
a multi-channel power→power graph, then computes the same things the PubMed
"helix" lab surfaces:

    - multi-channel edges   (a pair related several ways at once)
    - summed-weight degree  (how central a power is)
    - hubs                  (top powers by total connection weight)
    - bridges               (powers many other minds track — cross-mind connectors)
    - composite trust        (one scalar per edge: kept − broken + belief − suspicion)

Channels (owner → other), mirroring "similar/cited/mesh/author" in helix:
    promise_out  — promises we made to them          (self_commitments.target_power)
    promise_in   — promises they made to us           (incoming_commitments.speaker)
    kept         — commitments involving them resolved KEPT
    broken       — commitments involving them resolved BROKEN
    belief       — beliefs we hold about them          (weighted by hp)
    prediction   — predictions about them              (weighted by confidence)
    suspicion    — standing suspicions about them       (weighted)

Everything is defensive: missing fields are skipped, so it works across agent
variants and won't crash a running game.

Edge container shape is the substrate's own router:  router[dst][channel] = weight
(see diplomacy_kg_universal.add_edge), so this is interoperable with that module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import diplomacy_kg_universal as U   # the substrate engine: add_edge / decay_edges / AugmentRecord

Router = Dict[str, Dict[str, float]]          # dst -> channel -> weight
Routers = Dict[str, Router]                    # src -> Router

# Weights for collapsing channels into one trust scalar (tune freely).
TRUST_SIGN = {"kept": +1.0, "promise_in": +0.3, "promise_out": +0.2,
              "belief": +0.2, "prediction": +0.1, "broken": -1.2, "suspicion": -0.8}


def _enum_str(v: Any) -> str:
    return str(getattr(v, "value", v) or "").lower()


def _bump(router: Router, dst: str, channel: str, delta: float) -> None:
    if not dst or delta == 0:
        return
    b = router.setdefault(dst, {})
    b[channel] = round(b.get(channel, 0.0) + float(delta), 4)


# ---------------------------------------------------------------------------
# Projection: AgentMind -> router
# ---------------------------------------------------------------------------
def from_mind(mind: Any) -> Routers:
    """Project ONE mind into {owner: {other_power: {channel: weight}}}."""
    owner = getattr(mind, "owner_power", None) or "SELF"
    router: Router = {}

    for c in getattr(mind, "incoming_commitments", {}).values():
        other = getattr(c, "speaker", None)
        st = _enum_str(getattr(c, "status", ""))
        _bump(router, other, "promise_in", 1.0)
        if st == "kept":
            _bump(router, other, "kept", 1.0)
        elif st == "broken":
            _bump(router, other, "broken", 1.0)

    for c in getattr(mind, "self_commitments", {}).values():
        other = getattr(c, "target_power", None) or getattr(c, "counterparty", None)
        st = _enum_str(getattr(c, "status", ""))
        _bump(router, other, "promise_out", 1.0)
        if st == "kept":
            _bump(router, other, "kept", 0.6)   # our own kept promise still builds the tie
        elif st == "broken":
            _bump(router, other, "broken", 0.6)

    for b in getattr(mind, "beliefs", {}).values():
        other = getattr(b, "about_power", None)
        hp = float(getattr(b, "hp", 1.0) or 0.0)
        _bump(router, other, "belief", max(0.1, hp))

    for p in getattr(mind, "predictions", {}).values():
        other = getattr(p, "about_power", None)
        conf = float(getattr(p, "confidence", 0.0) or 0.0)
        _bump(router, other, "prediction", conf)

    for s in getattr(mind, "suspicions", []) or []:
        other = s.get("about_power") if isinstance(s, dict) else getattr(s, "about_power", None)
        w = (s.get("weight", 0.5) if isinstance(s, dict) else getattr(s, "weight", 0.5)) or 0.5
        _bump(router, other, "suspicion", float(w))

    router.pop(owner, None)   # drop self-loops
    return {owner: router}


def from_minds(minds: Dict[str, Any]) -> Routers:
    """Merge several minds into one combined multi-power graph (the whole table)."""
    combined: Routers = {}
    for mind in minds.values():
        for src, router in routers_from_mind(mind).items():
            dst_map = combined.setdefault(src, {})
            for dst, chans in router.items():
                tgt = dst_map.setdefault(dst, {})
                for ch, w in chans.items():
                    tgt[ch] = round(tgt.get(ch, 0.0) + w, 4)
    return combined


# ---------------------------------------------------------------------------
# LIVE WIRING — populate the substrate router on the mind itself
# ---------------------------------------------------------------------------
# This is the piece that was missing: the substrate engine (add_edge / decay_edges
# / NodeAugment / handles) existed but nothing fed it during play. `sync_router_
# from_mind` is called once per phase from absorb_phase_resolution and mirrors the
# mind's typed collections into a real, persisted, DECAYING multi-channel graph
# stored at `mind.augment` (an AugmentRecord). After this runs, the helix views
# read the live graph instead of re-projecting each call.
#
# Two flavours of channel:
#   - EVENT channels (promise_in/out, kept, broken, prediction): folded ONCE per
#     (node_id, status) so each event adds weight a single time, then decays —
#     this is how a betrayal or a kept promise leaves a fading memory.
#   - STATE channels (belief, suspicion): re-asserted each phase in proportion to
#     current strength, so a persistently-held belief stays lit while a fading one
#     decays out. Steady-state weight ≈ the belief's hp.

PHASE_DECAY = 0.9          # per-phase fade applied before re-asserting state edges
PHASE_EPSILON = 0.02       # prune edges weaker than this
STATE_REINFORCE = 0.1      # belief/suspicion reinforcement fraction per phase


def _owner_node(owner: str) -> str:
    return f"power:{owner}"


def sync_router_from_mind(mind: Any, *, attach_beliefs: bool = True,
                          max_belief_nodes: int = 60) -> None:
    """Fold the mind's current typed state into its substrate router. Idempotent
    and safe to call every phase. Never raises (best-effort)."""
    try:
        owner = getattr(mind, "owner_power", None) or "SELF"
        if not hasattr(mind, "augment") or mind.augment is None:
            mind.augment = U.AugmentRecord()
        aug = mind.augment
        oid = _owner_node(owner)
        onode = aug.get(oid)
        if not onode.handles:
            aug.attach(oid, content=owner, ctx="self", kind="power")
        router = onode.router
        folded = set(onode.meta.get("folded", []))

        # 1) fade everything one step (event memory decays; state re-asserts below)
        U.decay_edges(router, factor=PHASE_DECAY, epsilon=PHASE_EPSILON)

        def ensure_power(p: str) -> str:
            nid = _owner_node(p)
            if nid not in aug.by_node_id:
                aug.attach(nid, content=p, ctx="power", kind="power")
            return nid

        def event(node_key: str, other: str, channel: str, delta: float) -> None:
            if not other or node_key in folded:
                return
            U.add_edge(router, ensure_power(other), channel=channel, delta=delta)
            folded.add(node_key)

        # 2) EVENT channels — promises and their resolutions, predictions
        for c in getattr(mind, "incoming_commitments", {}).values():
            other = getattr(c, "speaker", None)
            st = _enum_str(getattr(c, "status", ""))
            event(f"{c.id}:in", other, "promise_in", 1.0)
            if st in ("kept", "broken"):
                event(f"{c.id}:{st}", other, st, 1.0 if st == "kept" else 1.2)
        for c in getattr(mind, "self_commitments", {}).values():
            other = getattr(c, "target_power", None) or getattr(c, "counterparty", None)
            st = _enum_str(getattr(c, "status", ""))
            event(f"{c.id}:out", other, "promise_out", 0.8)
            if st in ("kept", "broken"):
                event(f"{c.id}:{st}", other, st, 0.6 if st == "kept" else 0.8)
        for p in getattr(mind, "predictions", {}).values():
            st = _enum_str(getattr(p, "status", ""))
            if st in ("confirmed", "refuted"):
                conf = float(getattr(p, "confidence", 0.0) or 0.0)
                event(f"{p.id}:{st}", getattr(p, "about_power", None),
                      "prediction", conf if st == "confirmed" else -conf)

        # 3) STATE channels — re-assert each phase proportional to current strength
        for b in getattr(mind, "beliefs", {}).values():
            other = getattr(b, "about_power", None)
            hp = float(getattr(b, "hp", 1.0) or 0.0)
            if other:
                U.add_edge(router, ensure_power(other), channel="belief",
                           delta=max(0.05, hp) * STATE_REINFORCE)
        for s in getattr(mind, "suspicions", []) or []:
            other = s.get("about_power") if isinstance(s, dict) else getattr(s, "about_power", None)
            w = (s.get("weight", 0.5) if isinstance(s, dict) else getattr(s, "weight", 0.5)) or 0.5
            if other:
                U.add_edge(router, ensure_power(other), channel="suspicion",
                           delta=float(w) * STATE_REINFORCE)

        router.pop(oid, None)                      # no self-loop
        onode.meta["folded"] = sorted(folded)

        # 4) Optional: register belief nodes with embedding handles so the
        #    substrate's associative retrieval has real content to match on.
        if attach_beliefs:
            count = 0
            for b in getattr(mind, "beliefs", {}).values():
                if count >= max_belief_nodes:
                    break
                bid = f"belief:{getattr(b, 'id', id(b))}"
                head = getattr(b, "head", "") or ""
                about = getattr(b, "about_power", "") or ""
                bnode = aug.get(bid)
                if not bnode.handles and head:
                    aug.attach(bid, content=head, ctx=about, kind="belief",
                               hp=float(getattr(b, "hp", 1.0) or 1.0),
                               meta={"about_power": about})
                if about:
                    U.add_edge(bnode.router, ensure_power(about), channel="about", delta=1.0)
                count += 1
    except Exception as e:  # never break the game loop
        print(f"[kg sync] non-fatal: {e}")


def live_routers_from_mind(mind: Any) -> Optional[Routers]:
    """Read the populated substrate router (if any) as a power→power graph,
    stripping the 'power:' namespace so output matches the projection path."""
    aug = getattr(mind, "augment", None)
    owner = getattr(mind, "owner_power", None) or "SELF"
    if aug is None:
        return None
    onode = aug.by_node_id.get(_owner_node(owner))
    if not onode or not onode.router:
        return None
    out: Router = {}
    for dst, chans in onode.router.items():
        if not dst.startswith("power:"):
            continue
        bare = dst[len("power:"):]
        live = {c: round(w, 4) for c, w in chans.items() if abs(w) > 1e-9}
        if live:
            out[bare] = live
    return {owner: out} if out else None


def routers_from_mind(mind: Any) -> Routers:
    """Prefer the LIVE substrate router; fall back to on-the-fly projection."""
    live = live_routers_from_mind(mind)
    return live if live else from_mind(mind)


# ---------------------------------------------------------------------------
# Helix-style analytics (work on any router-of-routers)
# ---------------------------------------------------------------------------
def node_degree(routers: Routers) -> Dict[str, Dict[str, Any]]:
    """Summed weighted degree per node across all channels (out + in)."""
    deg: Dict[str, Dict[str, Any]] = {}

    def acc(n: str, w: float, ch: str) -> None:
        d = deg.setdefault(n, {"weight": 0.0, "edges": 0, "channels": {}})
        d["weight"] = round(d["weight"] + w, 4)
        d["edges"] += 1
        d["channels"][ch] = round(d["channels"].get(ch, 0.0) + w, 4)

    for src, router in routers.items():
        for dst, chans in router.items():
            for ch, w in chans.items():
                acc(src, abs(w), ch)
                acc(dst, abs(w), ch)
    return deg


def hubs(routers: Routers, top: int = 12) -> List[Dict[str, Any]]:
    deg = node_degree(routers)
    items = [{"node": n, "weight": d["weight"], "edges": d["edges"],
              "channels": sorted(d["channels"].keys())} for n, d in deg.items()]
    items.sort(key=lambda x: (-x["weight"], -x["edges"]))
    return items[:top]


def multi_channel_edges(routers: Routers, min_channels: int = 2) -> List[Dict[str, Any]]:
    """Directed edges carrying ≥ min_channels at once (the multi-channel ties)."""
    out: List[Dict[str, Any]] = []
    for src, router in routers.items():
        for dst, chans in router.items():
            live = {c: round(w, 4) for c, w in chans.items() if abs(w) > 1e-9}
            if len(live) >= min_channels:
                out.append({"src": src, "dst": dst, "channels": dict(sorted(live.items())),
                            "n_channels": len(live),
                            "weight": round(sum(live.values()), 4),
                            "trust": composite_trust(live)})
    out.sort(key=lambda e: (-e["n_channels"], -e["weight"]))
    return out


def bridges(minds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Powers tracked by many DIFFERENT minds — the cross-mind connectors.
    (Analogous to a PubMed paper that several queries all retrieved.)"""
    tracked_by: Dict[str, set] = {}
    for owner, mind in minds.items():
        for src, router in routers_from_mind(mind).items():
            for dst in router:
                tracked_by.setdefault(dst, set()).add(src)
    out = [{"node": n, "tracked_by": sorted(g), "count": len(g)}
           for n, g in tracked_by.items() if len(g) > 1]
    out.sort(key=lambda x: -x["count"])
    return out


def composite_trust(channels: Dict[str, float]) -> float:
    """Collapse a multi-channel edge into one scalar (green/gold/red in the lens)."""
    return round(sum(TRUST_SIGN.get(c, 0.0) * w for c, w in channels.items()), 4)


def trust_band(score: float) -> str:
    return "green" if score >= 0.5 else ("red" if score <= -0.5 else "gold")


# ---------------------------------------------------------------------------
# Bundled report — drop into the biopsy / graph lens
# ---------------------------------------------------------------------------
def analyze_mind(mind: Any) -> Dict[str, Any]:
    """Single-power view: its multi-channel edges to every other power + trust."""
    routers = routers_from_mind(mind)
    owner = next(iter(routers))
    router = routers[owner]
    edges = []
    for dst, chans in router.items():
        live = {c: round(w, 4) for c, w in chans.items() if abs(w) > 1e-9}
        score = composite_trust(live)
        edges.append({"power": dst, "channels": dict(sorted(live.items())),
                      "n_channels": len(live), "weight": round(sum(live.values()), 4),
                      "trust": score, "band": trust_band(score)})
    edges.sort(key=lambda e: -e["weight"])
    return {"owner": owner, "edges": edges,
            "hubs": hubs(routers, top=8),
            "multi_channel": multi_channel_edges(routers, min_channels=2)}


def analyze_table(minds: Dict[str, Any]) -> Dict[str, Any]:
    """Whole-table view across all minds: hubs, bridges, multi-channel ties."""
    routers = from_minds(minds)
    return {"powers": sorted(minds.keys()),
            "hubs": hubs(routers, top=12),
            "bridges": bridges(minds),
            "multi_channel": multi_channel_edges(routers, min_channels=2),
            "edge_count": sum(len(r) for r in routers.values())}
