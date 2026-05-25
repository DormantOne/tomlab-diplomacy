"""
Per-agent knowledge graphs. Each agent owns SIX graphs:

  1. PERSONALITY  - Big-five-ish traits, tone, communication style.
  2. SOUL         - Deep values, fears, desires, formative beliefs.
  3. ETHICS       - Moral rules: what they will / won't do; thresholds.
  4. THEORY_OF_MIND - Models of every other player: trust, intentions, dispositions.
  5. STRATEGY     - Doctrines, current plan, evaluations of provinces / units.
  6. COUNTERFACTUALS - "What if" scenarios kept as branching hypotheses.

Each KG is a directed multigraph stored as plain Python dicts (no networkx
dependency). Nodes have types and attributes. Edges are typed relationships
with optional weights.

The graphs are queryable by node-type, by relation, by neighborhood, etc.
The agent renders relevant slices into prompts before each LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
import json
import time


@dataclass
class Node:
    id: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    rel: str
    weight: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """A single labeled, directed graph."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.created_at = time.time()

    # ---------- Mutation ---------- #
    def add_node(self, node_id: str, node_type: str, **attrs) -> Node:
        if node_id in self.nodes:
            self.nodes[node_id].attrs.update(attrs)
            self.nodes[node_id].type = node_type
            return self.nodes[node_id]
        n = Node(id=node_id, type=node_type, attrs=dict(attrs))
        self.nodes[node_id] = n
        return n

    def add_edge(self, src: str, dst: str, rel: str, weight: float = 1.0, **attrs) -> Edge:
        if src not in self.nodes:
            self.add_node(src, "_implicit")
        if dst not in self.nodes:
            self.add_node(dst, "_implicit")
        e = Edge(src=src, dst=dst, rel=rel, weight=weight, attrs=dict(attrs))
        self.edges.append(e)
        return e

    def remove_edges(self, src: Optional[str] = None, dst: Optional[str] = None,
                     rel: Optional[str] = None) -> int:
        before = len(self.edges)
        self.edges = [e for e in self.edges
                      if not ((src is None or e.src == src) and
                              (dst is None or e.dst == dst) and
                              (rel is None or e.rel == rel))]
        return before - len(self.edges)

    # ---------- Queries ---------- #
    def nodes_of_type(self, node_type: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == node_type]

    def neighbors(self, node_id: str, rel: Optional[str] = None) -> list[tuple[Edge, Node]]:
        out = []
        for e in self.edges:
            if e.src == node_id and (rel is None or e.rel == rel):
                if e.dst in self.nodes:
                    out.append((e, self.nodes[e.dst]))
        return out

    def predecessors(self, node_id: str, rel: Optional[str] = None) -> list[tuple[Edge, Node]]:
        out = []
        for e in self.edges:
            if e.dst == node_id and (rel is None or e.rel == rel):
                if e.src in self.nodes:
                    out.append((e, self.nodes[e.src]))
        return out

    def edges_with_relation(self, rel: str) -> list[Edge]:
        return [e for e in self.edges if e.rel == rel]

    # ---------- Rendering ---------- #
    def render(self, max_lines: int = 60) -> str:
        """Compact, prompt-friendly text rendering of the whole graph."""
        lines = [f"# {self.name}"]
        # group edges by source node for readability
        by_src: dict[str, list[Edge]] = {}
        for e in self.edges:
            by_src.setdefault(e.src, []).append(e)
        for nid, n in self.nodes.items():
            head = f"- ({n.type}) {nid}"
            if n.attrs:
                attr_bits = ", ".join(f"{k}={v}" for k, v in n.attrs.items()
                                     if k != "_internal")
                if attr_bits:
                    head += f"  [{attr_bits}]"
            lines.append(head)
            for e in by_src.get(nid, []):
                w = f" (w={e.weight:.2f})" if e.weight != 1.0 else ""
                lines.append(f"    --{e.rel}{w}--> {e.dst}")
            if len(lines) > max_lines:
                lines.append(f"... ({len(self.edges) - max_lines} more edges truncated)")
                break
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": [{"id": n.id, "type": n.type, "attrs": n.attrs}
                      for n in self.nodes.values()],
            "edges": [{"src": e.src, "dst": e.dst, "rel": e.rel,
                       "weight": e.weight, "attrs": e.attrs}
                      for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeGraph":
        g = cls(d.get("name", "unnamed"))
        for n in d.get("nodes", []):
            g.nodes[n["id"]] = Node(id=n["id"], type=n.get("type", "_implicit"),
                                    attrs=dict(n.get("attrs", {})))
        for e in d.get("edges", []):
            g.edges.append(Edge(
                src=e["src"], dst=e["dst"], rel=e.get("rel", "rel"),
                weight=float(e.get("weight", 1.0)),
                attrs=dict(e.get("attrs", {})),
            ))
        return g


# --------------------------- The 6-KG bundle -------------------------------- #


GRAPH_NAMES = ["personality", "soul", "ethics",
               "theory_of_mind", "strategy", "counterfactuals"]


class AgentKGBundle:
    """
    The six knowledge graphs an agent reasons with.

    Each graph has its own schema of node-types and edge-relations,
    documented in the seed methods below.
    """

    def __init__(self, owner_power: str):
        self.owner_power = owner_power
        self.graphs: dict[str, KnowledgeGraph] = {
            name: KnowledgeGraph(name) for name in GRAPH_NAMES
        }

    def __getitem__(self, name: str) -> KnowledgeGraph:
        return self.graphs[name]

    # ---- Specialized accessors used during prompt assembly ----

    def render_personality(self) -> str:
        return self.graphs["personality"].render(max_lines=40)

    def render_soul(self) -> str:
        return self.graphs["soul"].render(max_lines=40)

    def render_ethics(self) -> str:
        return self.graphs["ethics"].render(max_lines=40)

    def render_theory_of_mind(self, focus_powers: Optional[Iterable[str]] = None) -> str:
        """Render only the slice involving the given powers (or all)."""
        g = self.graphs["theory_of_mind"]
        if not focus_powers:
            return g.render(max_lines=80)
        focus = set(focus_powers)
        lines = [f"# {g.name}"]
        for nid, n in g.nodes.items():
            if n.type == "power" and nid not in focus:
                continue
            attr_bits = ", ".join(f"{k}={v}" for k, v in n.attrs.items())
            head = f"- ({n.type}) {nid}"
            if attr_bits:
                head += f"  [{attr_bits}]"
            lines.append(head)
            for e in g.edges:
                if e.src == nid:
                    w = f" (w={e.weight:.2f})" if e.weight != 1.0 else ""
                    lines.append(f"    --{e.rel}{w}--> {e.dst}")
        return "\n".join(lines)

    def render_strategy(self) -> str:
        return self.graphs["strategy"].render(max_lines=80)

    def render_counterfactuals(self) -> str:
        return self.graphs["counterfactuals"].render(max_lines=60)

    # ---- Update helpers used by the agent after each turn ----

    def update_trust(self, other_power: str, delta: float, reason: str) -> None:
        g = self.graphs["theory_of_mind"]
        nid = f"power:{other_power}"
        if nid not in g.nodes:
            g.add_node(nid, "power", trust=0.0, last_action="")
        node = g.nodes[nid]
        node.attrs["trust"] = max(-1.0, min(1.0, node.attrs.get("trust", 0.0) + delta))
        node.attrs["last_reason"] = reason

    def add_counterfactual(self, label: str, premise: str, expected: str,
                           weight: float = 0.5) -> None:
        g = self.graphs["counterfactuals"]
        cf_id = f"cf:{label}:{int(time.time()*1000) % 100000}"
        g.add_node(cf_id, "counterfactual", premise=premise,
                   expected=expected, weight=weight)
        g.add_edge(cf_id, f"power:{self.owner_power}", "concerns")

    def add_strategic_target(self, province: str, value: float, rationale: str) -> None:
        g = self.graphs["strategy"]
        nid = f"target:{province}"
        g.add_node(nid, "target", value=value, rationale=rationale)
        g.add_edge(f"power:{self.owner_power}", nid, "wants",
                   weight=value)

    def to_dict(self) -> dict:
        return {name: g.to_dict() for name, g in self.graphs.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def load_from_dict(self, d: dict, only: Optional[Iterable[str]] = None) -> None:
        """Replace graphs in-place from a serialized dict.

        If `only` is given, only those graph names are loaded — useful for
        cross-game persistence where we keep theory_of_mind/counterfactuals
        but reset strategy.
        """
        keys = set(only) if only else set(d.keys())
        for name in keys:
            if name in d and name in self.graphs:
                self.graphs[name] = KnowledgeGraph.from_dict(d[name])
