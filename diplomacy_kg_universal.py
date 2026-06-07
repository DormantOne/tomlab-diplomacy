"""
diplomacy_kg_universal.py — Phase 1 / Phase 2 bridge: universal KG primitives.

This module is the magic_kg-shaped foundation that Phase 2 will wire into
the substrate. In Phase 1 it is STAND-ALONE — none of the typed records in
diplomacy_kg_schema.py reference it yet. The point of shipping it now is so
Phase 2 only has to wire it in, not design it.

Three concerns are bundled here:

  1. EMBEDDINGS + SIMILARITY
     Tiny hashed-BoW embedding (placeholder; Phase 3 swaps in
     sentence-transformers). Cosine similarity. The handles-bundle
     convention from magic_kg: each retrievable node has THREE embedding
     vectors — content, context, kind+content — and similarity is the max
     across the cross-product against the query's three vectors.

  2. MULTI-CHANNEL WEIGHTED EDGES
     The `router` dict shape from magic_kg: `dict[other_id, dict[channel,
     weight]]`. add_edge / decay_edges helpers. Channels are arbitrary
     strings — semantic, used_together, evidence_for, contradicts, etc.

  3. STRUCTURED PRECONDITION PREDICATES
     A small safe predicate language so nodes can declare when they should
     fire. Predicate := { "all" | "any": [ {feature, op, value}, ... ] }.
     Used by Phase 2's fovea/retrieve to gate which nodes contribute to the
     prompt for the current phase.

Plus:

  4. AugmentRecord
     A side-car dataclass that holds the universal fields (handles, router,
     hp, hits, success, critic_score, meta, precondition) keyed by node id.
     Phase 2 attaches one of these to each AgentMind so the typed records
     stay pristine. This is option B from earlier discussion: external
     augmentation, reversible, no schema migration required.

This module deliberately does NOT import diplomacy_kg_schema. It can be
unit-tested in isolation.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Any


# ============================================================================
# 1. Embeddings + similarity
# ============================================================================
# Phase 1 uses a hashed bag-of-words embedder. This is intentionally crude —
# it works for sanity tests and for retrieving by exact-token overlap. For
# Diplomacy negotiation text (full English), Phase 3 will swap in
# sentence-transformers/all-MiniLM-L6-v2; the rest of this file is unchanged
# because the contract is just "embed: str -> list[float]".

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]{2,}")


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall((s or "").lower())


def embed(text: str, dim: int = 64) -> list[float]:
    """Hashed bag-of-words → normalized dim-D vector. Placeholder embedder."""
    v = [0.0] * dim
    toks = tokenize(text) or ["empty"]
    for t in toks:
        h = _stable_hash(t)
        sign = 1 if ((h >> 8) & 1) else -1
        weight = 1.0 + min(3, toks.count(t)) * 0.1
        v[h % dim] += sign * weight
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors. 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def make_handles(content: str, ctx: str = "", kind: str = "") -> list[list[float]]:
    """The magic_kg 3-vector handle bundle: content / context / kind+content.

    Retrieval scores `max(cosine(qv, h) for qv in query_vecs for h in handles)`,
    so a query that matches any of the three angles boosts the node.
    """
    return [
        embed(content),
        embed(ctx or content),
        embed(f"{kind} {content} {ctx}".strip()),
    ]


def query_handles(query: str) -> list[list[float]]:
    """Mirror of make_handles for the query side. Three angles into the index."""
    return [embed(query), embed(f"context {query}"), embed(f"intent {query}")]


def best_handle_similarity(handles: list[list[float]],
                           query_vecs: list[list[float]]) -> float:
    """max over the cross-product. Returns 0.0 if either side is empty."""
    if not handles or not query_vecs:
        return 0.0
    return max(cosine(qv, h) for qv in query_vecs for h in handles)


# ============================================================================
# 2. Multi-channel weighted edges
# ============================================================================
# Edge shape: router[dst_id][channel] = weight (float). One node's `router`
# is a dict of dicts. Edges are directed; a bidirectional relationship needs
# entries on both endpoints.

EdgeChannel = str  # 'semantic' | 'used_together' | 'evidence_for' | ...


def add_edge(
    router: dict[str, dict[EdgeChannel, float]],
    dst_id: str,
    channel: EdgeChannel = "semantic",
    delta: float = 0.05,
) -> None:
    """Add or strengthen an edge in `router` along `channel` by `delta`.

    Net-zero deltas (-0.0) and self-loops are silently ignored. Weights can
    go negative — that's how 'used_together but lost' encodes anti-coupling.
    """
    if not dst_id:
        return
    bucket = router.setdefault(dst_id, {})
    bucket[channel] = bucket.get(channel, 0.0) + delta


def edge_weight(router: dict[str, dict[EdgeChannel, float]],
                dst_id: str,
                channel: Optional[EdgeChannel] = None) -> float:
    """Total weight to dst across all channels (or just one channel)."""
    bucket = router.get(dst_id)
    if not bucket:
        return 0.0
    if channel is None:
        return sum(bucket.values())
    return bucket.get(channel, 0.0)


def decay_edges(
    router: dict[str, dict[EdgeChannel, float]],
    *,
    factor: float = 0.995,
    epsilon: float = 0.01,
) -> int:
    """Multiply every channel weight by `factor`; drop edges below |epsilon|.

    Returns count of channels pruned. Used by the dream/compaction pass in
    Phase 2 to keep the router from accumulating dead weight.
    """
    pruned = 0
    for dst in list(router.keys()):
        bucket = router[dst]
        for ch in list(bucket.keys()):
            bucket[ch] *= factor
            if abs(bucket[ch]) < epsilon:
                del bucket[ch]
                pruned += 1
        if not bucket:
            del router[dst]
    return pruned


# ============================================================================
# 3. Structured precondition predicates
# ============================================================================
# A precondition is a small safe predicate over a "frame" — a flat dict of
# named features. Lifted from magic_kg with the feature whitelist removed,
# since Diplomacy's feature set differs from Go's. Operator whitelist stays.

_ALLOWED_OPS = {"<", "<=", ">", ">=", "==", "!=", "in", "between"}


def _coerce_num(x: Any) -> Any:
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, str) and re.fullmatch(r"-?\d+(\.\d+)?", x.strip()):
            return float(x) if "." in x else int(x)
    except Exception:
        pass
    return x


def _cmp(a: Any, op: str, b: Any) -> bool:
    a = _coerce_num(a)
    b = _coerce_num(b)
    try:
        if op == "<":  return a < b
        if op == "<=": return a <= b
        if op == ">":  return a > b
        if op == ">=": return a >= b
        if op == "==":
            if isinstance(a, str) or isinstance(b, str):
                return str(a).lower() == str(b).lower()
            return a == b
        if op == "!=": return not _cmp(a, "==", b)
        if op == "in":
            choices = b if isinstance(b, list) else [b]
            return str(a).lower() in [str(x).lower() for x in choices]
        if op == "between" and isinstance(b, list) and len(b) >= 2:
            return b[0] <= a <= b[1]
    except Exception:
        return False
    return False


def sanitize_precondition(
    pc: Any, *, allowed_features: Optional[set[str]] = None,
) -> Optional[dict]:
    """Return a small validated predicate dict, or None.

    Shape after sanitization:  {"all" | "any": [{feature, op, value}, ...]}

    `allowed_features` constrains which feature names are accepted; pass
    None to allow any feature name (lenient mode used in Phase 1).
    """
    if not isinstance(pc, dict):
        return None
    if isinstance(pc.get("all"), list):
        root, items = "all", pc["all"]
    elif isinstance(pc.get("any"), list):
        root, items = "any", pc["any"]
    elif all(k in pc for k in ("feature", "op", "value")):
        root, items = "all", [pc]
    else:
        return None
    clean = []
    for it in items[:8]:
        if not isinstance(it, dict):
            continue
        f = str(it.get("feature", "")).strip()
        op = str(it.get("op", "==")).strip()
        if not f or op not in _ALLOWED_OPS:
            continue
        if allowed_features is not None and f not in allowed_features:
            continue
        val = it.get("value")
        if isinstance(val, list):
            val = [_coerce_num(v) for v in val[:8]]
        else:
            val = _coerce_num(val)
        clean.append({"feature": f, "op": op, "value": val})
    if not clean:
        return None
    return {root: clean}


def precondition_holds(pc: Optional[dict], frame: dict[str, Any]) -> bool:
    """True if predicate is satisfied by the frame. None precondition ⇒ True.

    `frame` is a flat dict; missing features compare as None which fails
    most numeric ops gracefully.
    """
    pc = sanitize_precondition(pc)
    if not pc:
        return True
    key = "all" if "all" in pc else "any"
    results = [_cmp(frame.get(it["feature"]), it["op"], it["value"])
               for it in pc[key]]
    return all(results) if key == "all" else any(results)


# ============================================================================
# 4. AugmentRecord — the side-car for typed records
# ============================================================================
# Each AgentMind in Phase 2 will own one AugmentRecord. Lookups by node id
# return a NodeAugment with the universal fields. The typed records stay
# unchanged.

@dataclass
class NodeAugment:
    """Universal fields for a single typed record (BeliefNode, etc.)."""
    handles: list[list[float]] = field(default_factory=list)
    router: dict[str, dict[EdgeChannel, float]] = field(default_factory=dict)
    hp: float = 1.0
    hits: int = 0
    success: float = 0.0
    critic_score: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)
    precondition: Optional[dict] = None
    last_hit_at: Optional[float] = None
    selected_count_this_game: int = 0


@dataclass
class AugmentRecord:
    """Per-mind side-car. Maps node_id → NodeAugment. Phase 2 hooks this up."""
    by_node_id: dict[str, NodeAugment] = field(default_factory=dict)

    def get(self, node_id: str) -> NodeAugment:
        if node_id not in self.by_node_id:
            self.by_node_id[node_id] = NodeAugment()
        return self.by_node_id[node_id]

    def attach(self, node_id: str, *,
               content: str = "", ctx: str = "", kind: str = "",
               hp: float = 1.0, critic_score: float = 0.5,
               precondition: Optional[dict] = None,
               meta: Optional[dict] = None) -> NodeAugment:
        """Initialize augmentation for a freshly-created node."""
        a = self.get(node_id)
        if content or ctx or kind:
            a.handles = make_handles(content, ctx, kind)
        a.hp = hp
        a.critic_score = critic_score
        a.precondition = sanitize_precondition(precondition)
        if meta:
            a.meta.update(meta)
        return a

    def detach(self, node_id: str) -> None:
        self.by_node_id.pop(node_id, None)


def score_node(
    augment: NodeAugment, query_vecs: list[list[float]],
    *, hp_floor: float = 0.05, hp_cap: float = 3.0,
) -> float:
    """magic_kg-style retrieval score: similarity * critic_factor * hp_factor.

    Score = sim * (0.6 + 0.4 * critic_score) * (0.5 + min(hp_cap, hp) / hp_cap)
    """
    sim = best_handle_similarity(augment.handles, query_vecs)
    critic_factor = 0.6 + 0.4 * augment.critic_score
    hp_clamped = max(hp_floor, min(hp_cap, augment.hp))
    hp_factor = 0.5 + hp_clamped / hp_cap
    return sim * critic_factor * hp_factor


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("KG UNIVERSAL PRIMITIVES SANITY CHECK")
    print("=" * 72)

    # Embeddings
    v1 = embed("Russia will move to Galicia in spring")
    v2 = embed("Russia will move to Galicia in spring")
    v3 = embed("France considers a Mediterranean campaign")
    assert cosine(v1, v2) > 0.999, "identical text should be cosine ≈ 1"
    print(f"  identical-text cosine: {cosine(v1, v2):.4f}")
    print(f"  unrelated-text cosine: {cosine(v1, v3):.4f}")
    assert cosine(v1, v3) < 0.5, f"unrelated text should be < 0.5, got {cosine(v1, v3)}"

    # Handle bundle and query
    handles = make_handles(
        content="Russia tends to break long-term alliances in fall",
        ctx="game 3, observed against Austria",
        kind="belief credibility",
    )
    qvs = query_handles("does Russia keep alliances")
    sim = best_handle_similarity(handles, qvs)
    print(f"  handle-vs-query similarity: {sim:.4f}")
    assert sim > 0.0

    # Multi-channel edges
    router: dict = {}
    add_edge(router, "node_b", channel="semantic", delta=0.5)
    add_edge(router, "node_b", channel="used_together", delta=0.3)
    add_edge(router, "node_c", channel="evidence_for", delta=0.8)
    assert edge_weight(router, "node_b") == 0.8
    assert edge_weight(router, "node_b", "semantic") == 0.5
    assert edge_weight(router, "missing") == 0.0
    print(f"  edges: node_b total={edge_weight(router, 'node_b'):.2f}, "
          f"node_c total={edge_weight(router, 'node_c'):.2f}")

    # Edge decay
    pruned = decay_edges(router, factor=0.5, epsilon=0.5)
    print(f"  after decay+prune: pruned={pruned}, surviving={list(router.keys())}")
    # node_b semantic: 0.5*0.5=0.25 (pruned); used_together: 0.3*0.5=0.15 (pruned)
    # node_c evidence_for: 0.8*0.5=0.4 (pruned). All gone.
    assert "node_b" not in router and "node_c" not in router

    # Preconditions
    pc = sanitize_precondition({
        "all": [
            {"feature": "phase", "op": "==", "value": "FALL"},
            {"feature": "sc_count", "op": ">=", "value": 6},
        ],
    })
    assert pc is not None
    assert precondition_holds(pc, {"phase": "FALL", "sc_count": 7})
    assert not precondition_holds(pc, {"phase": "SPRING", "sc_count": 7})
    assert not precondition_holds(pc, {"phase": "FALL", "sc_count": 4})
    # Bad ops are stripped
    pc_dirty = sanitize_precondition({
        "any": [
            {"feature": "x", "op": "DROP TABLE", "value": 1},
            {"feature": "y", "op": "==", "value": 2},
        ],
    })
    assert pc_dirty == {"any": [{"feature": "y", "op": "==", "value": 2}]}
    print(f"  precondition test passed; sanitized dirty input cleanly")

    # AugmentRecord
    aug = AugmentRecord()
    aug.attach("belief_1",
               content="Russia keeps tactical promises but evades long-term ones",
               ctx="game 3 observation",
               kind="belief credibility",
               hp=1.4, critic_score=0.7)
    aug.attach("belief_2",
               content="France plans a southern campaign",
               ctx="own intent", kind="belief disposition",
               hp=1.0, critic_score=0.5)
    add_edge(aug.get("belief_1").router, "belief_2",
             channel="contradicts", delta=0.3)

    qvs = query_handles("does Russia honor long-term alliances")
    score_a = score_node(aug.get("belief_1"), qvs)
    score_b = score_node(aug.get("belief_2"), qvs)
    print(f"  retrieval scores — credibility belief: {score_a:.4f}, "
          f"disposition belief: {score_b:.4f}")
    assert score_a > score_b, "Russia-belief should outrank France-belief for that query"

    print()
    print("KG universal primitives sanity check passed.")
