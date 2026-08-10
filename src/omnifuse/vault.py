"""Vault — omnifuse-native memory: ``fuse`` knowledge in, ``surface`` it out.

Not a generic remember/recall store. Two omnifuse-specific dynamics:
  • fuse-on-write — facts are merged & deduped, entities coreferenced by label, so
    repeated knowledge collapses into one growing graph instead of piling up.
  • salience — every fuse (and every surface hit) bumps a node's salience; surface
    re-ranks results by it, so frequently-fused/used knowledge rises. (No PPR/Hebbian.)

    v = Vault()
    v.fuse("담보 한도는 5억원", facts=[("담보", "한도", "5억")])
    v.surface("담보 한도")            # fusion search, salience-ranked
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from .facade import from_triples
from .loaders import to_triple
from .models import SearchResult


class Vault:
    def __init__(self, *, llm=None, embedder: Optional[Callable[[str], list[float]]] = None,
                 auto_link: bool = True, reinforce: float = 0.5, **search_kwargs):
        self._facts: list[tuple] = []
        self._fact_set: set[tuple] = set()
        self._notes: list[tuple] = []
        self._labels: set[str] = set()
        self._salience: dict[str, float] = {}
        self._of = None
        self._dirty = True
        self.llm = llm
        self.embedder = embedder
        self.auto_link = auto_link
        self.reinforce = reinforce
        self.search_kwargs = search_kwargs

    def fuse(self, text: Optional[str] = None, *, facts: Optional[list] = None,
             entities: Optional[list[str]] = None, id: Optional[str] = None) -> "Vault":
        """Fuse knowledge in: facts (triples) and/or a note (text). Deduped & merged by label."""
        for t in (facts or []):
            tr = to_triple(t)
            key = (tr.s, tr.p, tr.o)
            if key not in self._fact_set:
                self._fact_set.add(key)
                self._facts.append(key)
                self._labels.add(tr.s)
                self._labels.add(tr.o)
                self._bump(tr.s)
                self._bump(tr.o)
        if text is not None:
            ents = list(entities) if entities is not None else (self._auto_link(text) if self.auto_link else [])
            self._notes.append((id or f"n{len(self._notes)}", text, ents))
            for e in ents:
                self._bump(e)
        self._dirty = True
        return self

    def surface(self, query: str, *, limit: Optional[int] = None) -> SearchResult:
        """Fusion search over everything fused, re-ranked by salience."""
        if self._dirty or self._of is None:
            self._of = from_triples(self._facts, self._notes, llm=self.llm,
                                    embedder=self.embedder, **self.search_kwargs)
            self._dirty = False
        r = self._of.search(query)
        r.evidence_nodes = sorted(r.evidence_nodes, key=lambda n: (-self._salience.get(n, 0.0), n))
        if limit is not None:
            r.evidence_nodes = r.evidence_nodes[:limit]
        for n in r.evidence_nodes:
            self._bump(n, self.reinforce)
        return r

    def salience(self, label: str) -> float:
        return self._salience.get(label, 0.0)

    def stats(self) -> dict:
        return {"facts": len(self._facts), "notes": len(self._notes), "entities": len(self._labels)}

    def _bump(self, label: str, amount: float = 1.0) -> None:
        self._salience[label] = self._salience.get(label, 0.0) + amount

    def _auto_link(self, text: str) -> list[str]:
        return sorted(label for label in self._labels if len(label) >= 2 and label in text)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s, p, o in self._facts:
                f.write(json.dumps({"f": [s, p, o]}, ensure_ascii=False) + "\n")
            for cid, text, ents in self._notes:
                f.write(json.dumps({"n": [cid, text, list(ents)]}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"sal": self._salience}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str, **kwargs) -> "Vault":
        v = cls(**kwargs)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "f" in d:
                    key = tuple(d["f"])
                    v._fact_set.add(key)
                    v._facts.append(key)
                    v._labels.update([key[0], key[2]])
                elif "n" in d:
                    v._notes.append((d["n"][0], d["n"][1], d["n"][2]))
                elif "sal" in d:
                    v._salience = {k: float(x) for k, x in d["sal"].items()}
        v._dirty = True
        return v
