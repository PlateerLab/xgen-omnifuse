"""Coercion + ingestion helpers — turn loose input (tuples, dicts, files) into models.

Zero-dep (stdlib json/csv). Lets users hand OmniFuse plain ``(s, p, o)`` tuples,
dicts, or JSONL/CSV files instead of constructing Node/Triple/Chunk by hand.
"""
from __future__ import annotations

import csv
import json

from .models import Chunk, Node, Triple

_ISA = {"instanceOf", "type", "subClassOf", "rdf:type"}


def to_triple(t) -> Triple:
    if isinstance(t, Triple):
        return t
    if isinstance(t, dict):
        return Triple(t["s"], t["p"], t["o"])
    return Triple(t[0], t[1], t[2])  # (s, p, o)


def to_chunk(c) -> Chunk:
    if isinstance(c, Chunk):
        return c
    if isinstance(c, dict):
        return Chunk(c["id"], c.get("text", ""), list(c.get("entities") or []),
                     c.get("embedding"), dict(c.get("meta") or {}), c.get("title", ""))
    # (id, text) or (id, text, entities)
    ents = list(c[2]) if len(c) > 2 and c[2] else []
    return Chunk(c[0], c[1] if len(c) > 1 else "", ents)


def to_node(n) -> Node:
    if isinstance(n, Node):
        return n
    if isinstance(n, dict):
        return Node(n["id"], n.get("label", n["id"]), n.get("kind", "instance"))
    return Node(n[0], n[1] if len(n) > 1 else n[0], n[2] if len(n) > 2 else "instance")


def derive_nodes(triples: list[Triple], labels: dict[str, str] | None = None) -> list[Node]:
    """Infer nodes from triples when none are given: object of an is-a edge -> class."""
    labels = labels or {}
    ids: dict[str, None] = {}
    classes: set[str] = set()
    for t in triples:
        ids.setdefault(t.s, None)
        ids.setdefault(t.o, None)
        if t.p in _ISA:
            classes.add(t.o)
    return [Node(i, labels.get(i, i), "class" if i in classes else "instance") for i in ids]


def read_jsonl(path: str) -> list[dict]:
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_triples_csv(path: str) -> list[tuple]:
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            s = row.get("s") or row.get("subject")
            p = row.get("p") or row.get("predicate")
            o = row.get("o") or row.get("object")
            if s and p and o:
                out.append((s, p, o))
    return out


def read_chunks_csv(path: str) -> list[tuple]:
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ents = [e for e in (row.get("entities") or "").split("|") if e]
            out.append((row["id"], row.get("text", ""), ents))
    return out
