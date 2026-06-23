"""Memory — a growing store you ``remember()`` facts/notes into and ``recall()`` from.

Built on OmniFuse's own search: you accumulate facts (triples) and notes (text), and
recall runs the one-shot graph+vector fusion over everything stored. Zero infra; notes
auto-link to known entities by label; optionally persist to JSONL.

    m = Memory()
    m.add_fact("담보", "instanceOf", "규정")
    m.remember("담보 한도는 5억원이다", triples=[("담보", "한도", "5억")])
    print(m.recall("담보 한도").answer)
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from .facade import from_triples
from .loaders import derive_nodes, to_triple
from .models import SearchResult


class Memory:
    def __init__(self, *, llm=None, embedder: Optional[Callable[[str], list[float]]] = None,
                 auto_link: bool = True, **search_kwargs):
        self._triples: list = []
        self._chunks: list = []
        self._of = None
        self._dirty = True
        self.llm = llm
        self.embedder = embedder
        self.auto_link = auto_link
        self.search_kwargs = search_kwargs

    def add_fact(self, s: str, p: str, o: str) -> "Memory":
        """Remember a relation (triple). Nodes are inferred on recall."""
        self._triples.append((s, p, o))
        self._dirty = True
        return self

    def remember(self, text: str, *, id: Optional[str] = None,
                 entities: Optional[list[str]] = None, triples: Optional[list] = None) -> str:
        """Remember a note (text) + optional facts. Returns the note id.

        If ``entities`` is omitted and ``auto_link`` is on, the note is linked to any
        known entity whose label appears in the text.
        """
        cid = id or f"mem{len(self._chunks)}"
        ents = list(entities) if entities is not None else (self._auto_entities(text) if self.auto_link else [])
        self._chunks.append((cid, text, ents))
        if triples:
            self._triples.extend(triples)
        self._dirty = True
        return cid

    def recall(self, query: str) -> SearchResult:
        """Search everything remembered via OmniFuse fusion."""
        if self._dirty or self._of is None:
            self._of = from_triples(self._triples, self._chunks, llm=self.llm,
                                    embedder=self.embedder, **self.search_kwargs)
            self._dirty = False
        return self._of.search(query)

    def stats(self) -> dict:
        return {"facts": len(self._triples), "notes": len(self._chunks)}

    def _auto_entities(self, text: str) -> list[str]:
        labels = [n.label for n in derive_nodes([to_triple(t) for t in self._triples])]
        return [l for l in labels if len(l) >= 2 and l in text]

    # ---- persistence (JSONL) ----
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s, p, o in self._triples:
                f.write(json.dumps({"t": [s, p, o]}, ensure_ascii=False) + "\n")
            for cid, text, ents in self._chunks:
                f.write(json.dumps({"c": [cid, text, list(ents)]}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str, **kwargs) -> "Memory":
        m = cls(**kwargs)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "t" in d:
                    m._triples.append(tuple(d["t"]))
                elif "c" in d:
                    m._chunks.append((d["c"][0], d["c"][1], d["c"][2]))
        m._dirty = True
        return m
