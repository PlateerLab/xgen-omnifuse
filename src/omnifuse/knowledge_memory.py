"""Zero-infrastructure reference provider with immutable, scope-bound snapshots.

This deliberately materializes a small bundle. Large DB implementations implement
knowledge_protocols directly and push filters/pagination into their own indexes.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from threading import RLock

from .backends.memory import InMemoryGraph, InMemoryVector
from .knowledge import (
    ContractError,
    KnowledgeBundle,
    SnapshotConflict,
    digest,
    validate_vector,
)
from .knowledge_protocols import OperationContext, Page, ReadScope
from .models import Chunk, Node


class MemoryKnowledgeProvider:
    def __init__(self, bundle: KnowledgeBundle | dict, *,
                 allowed_resource_ids: frozenset[str] | None = None,
                 authorize: Callable[[], bool] | None = None):
        self._lock = RLock()
        self._bundle = KnowledgeBundle.from_dict(bundle.to_dict() if hasattr(bundle, "to_dict") else bundle)
        self._allowed = allowed_resource_ids
        self._authorize = authorize
        self._snapshots = {self._bundle.snapshot_id: self._bundle.content_digest}
        self._revisions = {}
        self._chunks = {}
        self._immutable = {}
        self._record_identities(self._bundle)

    def _record_identities(self, bundle, *, validate_only=False):
        revisions = {}
        for r in bundle.resources:
            # A move/rename changes hierarchy, not immutable source bytes.
            value = (r.kind, r.content_digest)
            key = (r.id, r.revision)
            if key in self._revisions and self._revisions[key] != value:
                raise ContractError("resource revision reused for different content")
            revisions[key] = value
        chunks = {}
        for c in bundle.chunks:
            value = digest(asdict(c))
            if c.id in self._chunks and self._chunks[c.id] != value:
                raise ContractError("immutable chunk ID reused for different data")
            chunks[c.id] = value
        immutable = {}
        for kind, records in (("profile", bundle.profiles), ("evidence", bundle.evidence), ("fact", bundle.facts)):
            for record in records:
                value = asdict(record)
                if kind == "fact":
                    # New evidence may support the same fact; its meaning stays fixed.
                    value = {k: value[k] for k in ("subject_id", "predicate", "object_id", "literal", "datatype")}
                key = (kind, record.id)
                fingerprint = digest(value)
                if key in self._immutable and self._immutable[key] != fingerprint:
                    raise ContractError(f"immutable {kind} ID reused for different data")
                immutable[key] = fingerprint
        if not validate_only:
            self._revisions.update(revisions)
            self._chunks.update(chunks)
            self._immutable.update(immutable)

    @property
    def snapshot_id(self):
        with self._lock:
            return self._bundle.snapshot_id

    def publish(self, bundle: KnowledgeBundle | dict, *, expected_snapshot: str) -> None:
        """Publish all three components together. Failed writes leave state intact.

        Readers already in a view keep their snapshot. New readers see the complete
        replacement. Hosts needing immediate deletion/revocation during a read must
        revoke the view through authorize(), including before returning results.
        """
        candidate = KnowledgeBundle.from_dict(bundle.to_dict() if hasattr(bundle, "to_dict") else bundle)
        with self._lock:
            if candidate.corpus_id != self._bundle.corpus_id:
                raise ContractError("cannot publish another corpus")
            if candidate.snapshot_id == self._bundle.snapshot_id and candidate.content_digest == self._bundle.content_digest:
                return  # idempotent retry, even if original expected_snapshot is old
            if expected_snapshot != self._bundle.snapshot_id:
                raise SnapshotConflict("publication base snapshot changed")
            if candidate.snapshot_id in self._snapshots:
                raise SnapshotConflict("snapshot ID cannot be reused or rolled back")
            self._record_identities(candidate, validate_only=True)
            self._record_identities(candidate)
            self._snapshots[candidate.snapshot_id] = candidate.content_digest
            self._bundle = candidate

    @contextmanager
    def open_view(self, *, scope=None, snapshot_id=None, context=None):
        scope = scope or ReadScope()
        context = context or OperationContext()
        context.check()
        with self._lock:
            bundle = self._bundle
            if snapshot_id is not None and snapshot_id != bundle.snapshot_id:
                raise SnapshotConflict("requested snapshot is not current")
        view = _MemoryView(bundle, scope, self._allowed, self._authorize, context)
        try:
            view.check()
            yield view
            view.check()
        finally:
            view._closed = True


class _MemoryView:
    def __init__(self, bundle, scope, allowed, authorize, context):
        self.corpus_id, self.snapshot_id = bundle.corpus_id, bundle.snapshot_id
        self._authorize, self._context, self._closed = authorize, context, False
        self.check()
        resources = {r.id: r for r in bundle.resources}
        children = {}
        for r in resources.values():
            children.setdefault(r.parent_id, []).append(r.id)
        selected = set(resources)
        if scope.root_ids:
            if "hierarchy" not in bundle.components:
                raise ContractError("root scope requires hierarchy")
            selected = set()
            frontier = list(scope.root_ids)
            while frontier:
                rid = frontier.pop()
                if rid in selected or rid not in resources:
                    continue
                selected.add(rid)
                if scope.recursive:
                    frontier.extend(children.get(rid, ()))
            if not scope.recursive:
                selected.update(cid for rid in scope.root_ids for cid in children.get(rid, ()))
        if scope.resource_ids is not None:
            selected.intersection_update(scope.resource_ids)
        if allowed is not None:
            selected.intersection_update(allowed)
        self._resources = {rid: resources[rid] for rid in sorted(selected)}
        self._chunks = {c.id: c for c in bundle.chunks if c.resource_id in selected}
        self._evidence = {e.id: e for e in bundle.evidence if e.chunk_id in self._chunks}
        self._facts = [f for f in bundle.facts if (
            set(f.evidence_ids) <= self._evidence.keys() if f.assertion == "inferred"
            else bool(set(f.evidence_ids) & self._evidence.keys())
        )]
        # Remove hidden support IDs from returned facts/entities themselves.
        from dataclasses import replace
        self._facts = [replace(f, evidence_ids=tuple(e for e in f.evidence_ids if e in self._evidence))
                       for f in self._facts]
        visible = {x for f in self._facts for x in (f.subject_id, f.object_id) if x is not None}
        self._entities = {e.id: replace(e, evidence_ids=tuple(i for i in e.evidence_ids if i in self._evidence))
                          for e in bundle.entities if e.id in visible or set(e.evidence_ids) & self._evidence.keys()}
        self._ec = {eid: set() for eid in self._entities}
        for e in self._entities.values():
            self._ec[e.id].update(self._evidence[i].chunk_id for i in e.evidence_ids)
        for f in self._facts:
            for eid in (f.subject_id, f.object_id):
                if eid is not None:
                    self._ec[eid].update(self._evidence[i].chunk_id for i in f.evidence_ids)
        self._ce = {cid: [] for cid in self._chunks}
        for eid, cids in self._ec.items():
            for cid in cids:
                self._ce[cid].append(eid)
        self._profiles = {p.id: p for p in bundle.profiles}
        self._embeddings = [e for e in bundle.embeddings if e.chunk_id in self._chunks]
        self._label_index = InMemoryGraph([Node(e.id, e.label, e.kind) for e in self._entities.values()], [])
        self._text_index = InMemoryVector([Chunk(c.id, c.text, title=c.title) for c in self._chunks.values()])
        self._scope_digest = digest([self.snapshot_id, sorted(selected)])
        self.coverage = {
            "resources": len(self._resources), "chunks": len(self._chunks),
            "embedding_chunks": {pid: len({e.chunk_id for e in self._embeddings if e.profile_id == pid})
                                 for pid in self._profiles},
            "entities": len(self._entities), "facts": len(self._facts),
        }
        self.capabilities = frozenset(bundle.components) | (
            {"lexical"} if any(c.text or c.title for c in self._chunks.values()) else set()
        )
        self._resource_index = None
        self.hierarchy = self if "hierarchy" in self.capabilities else None
        self.embeddings = self if "embeddings" in self.capabilities else None
        self.graph = self if "graph" in self.capabilities else None
        self.content = self

    def check(self):
        if self._closed:
            raise RuntimeError("knowledge view is closed")
        self._context.check()
        if self._authorize is not None and not self._authorize():
            raise PermissionError("knowledge access revoked")

    def _read(self):
        self.check()
        self._context.consume()

    def _page(self, items, cursor, limit, key):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ContractError("page limit must be 1..1000")
        tag = digest([self._scope_digest, key])
        offset = 0
        if cursor is not None:
            try:
                token, raw = cursor.split(":")
                offset = int(raw)
                if token != tag or offset < 0 or offset > len(items):
                    raise ValueError()
            except (ValueError, AttributeError):
                raise ContractError("cursor belongs to another snapshot/scope/query") from None
        end = offset + limit
        return Page(tuple(deepcopy(items[offset:end])), f"{tag}:{end}" if end < len(items) else None, len(items))

    def get_resources(self, ids):
        self._read()
        return deepcopy([self._resources[i] for i in ids if i in self._resources])

    def search_resources(self, query, *, limit=20):
        self._read()
        if self._resource_index is None:
            rows = []
            for r in sorted(self._resources.values(), key=lambda r: r.id):
                names, current = [], r
                while current is not None:
                    names.append(current.name)
                    current = self._resources.get(current.parent_id)
                rows.append(Chunk(r.id, " / ".join(reversed(names)), title=r.name))
            self._resource_index = InMemoryVector(rows)
        return [(deepcopy(self._resources[c.id]), score)
                for c, score in self._resource_index.search(query, limit=limit)]

    def resolve_path(self, parts, *, parent_id=None):
        self._read()
        if isinstance(parts, str) or any(not isinstance(p, str) or not p for p in parts):
            raise ContractError("path must be a sequence of literal path components")
        parents = {parent_id}
        matches = []
        for part in parts:
            matches = [r for r in self._resources.values() if r.parent_id in parents and r.name == part]
            parents = {r.id for r in matches}
        return deepcopy(sorted(matches, key=lambda r: r.id))

    def children(self, parent_id, *, cursor=None, limit=100):
        self._read()
        items = sorted((r for r in self._resources.values() if r.parent_id == parent_id), key=lambda r: r.id)
        return self._page(items, cursor, limit, ["children", parent_id])

    def ancestors(self, resource_id):
        self._read()
        out = []
        r = self._resources.get(resource_id)
        while r is not None and r.parent_id in self._resources:
            r = self._resources[r.parent_id]
            out.append(r)
        return deepcopy(out)

    def descendants(self, root_id, *, cursor=None, limit=100):
        self._read()
        found = set()
        frontier = [root_id] if root_id in self._resources else []
        by_parent = {}
        for r in self._resources.values():
            by_parent.setdefault(r.parent_id, []).append(r.id)
        while frontier:
            rid = frontier.pop()
            for child in by_parent.get(rid, []):
                found.add(child)
                frontier.append(child)
        return self._page([self._resources[i] for i in sorted(found)], cursor, limit, ["descendants", root_id])

    def get_chunks(self, ids):
        self._read()
        return deepcopy([self._chunks[i] for i in ids if i in self._chunks])

    def get_evidence(self, ids):
        self._read()
        return deepcopy([self._evidence[i] for i in ids if i in self._evidence])

    def search_text(self, query, *, limit):
        self._read()
        return [(c.id, s) for c, s in self._text_index.search(query, limit=limit)]

    def profiles(self):
        self._read()
        return deepcopy(list(self._profiles.values()))

    def search_vector(self, values, *, profile_id, limit):
        self._read()
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ContractError("unknown embedding profile")
        validate_vector(values, profile)
        scored = []
        for e in self._embeddings:
            if e.profile_id != profile_id:
                continue
            if profile.metric == "cosine":
                # Scaling before normalization prevents overflow/underflow for
                # finite but very large or very small embedding coordinates.
                qs, es = max(abs(x) for x in values), max(abs(x) for x in e.values)
                qscaled, escaled = [x / qs for x in values], [x / es for x in e.values]
                qnorm, enorm = math.hypot(*qscaled), math.hypot(*escaled)
                score = sum((a / qnorm) * (b / enorm) for a, b in zip(qscaled, escaled))
            elif profile.metric == "dot":
                score = sum(a*b for a, b in zip(values, e.values))
            else:
                score = -math.sqrt(sum((a-b)**2 for a, b in zip(values, e.values)))
            if not math.isfinite(score):
                raise ContractError("embedding score exceeds finite numeric range")
            scored.append((e.chunk_id, score))
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:limit]

    def search_entities(self, query, *, limit):
        self._read()
        return [(deepcopy(self._entities[n.id]), score) for n, score in self._label_index.search_labels(query, limit=limit)]

    def get_entities(self, ids):
        self._read()
        return deepcopy([self._entities[i] for i in ids if i in self._entities])

    def neighbors(self, entity_id, *, direction="both", limit=100):
        self._read()
        if direction not in {"out", "in", "both"}:
            raise ContractError("invalid graph direction")
        return deepcopy([f for f in self._facts if (
            (direction != "in" and f.subject_id == entity_id) or
            (direction != "out" and f.object_id == entity_id)
        )][:limit])

    def class_members(self, class_id, *, cursor=None, limit=100):
        self._read()
        ids = {f.subject_id for f in self._facts if f.object_id == class_id and
               f.predicate in {"instanceOf", "type", "rdf:type", "subClassOf"}}
        return self._page([self._entities[i] for i in sorted(ids)], cursor, limit, ["class_members", class_id])

    def entities_for_chunks(self, chunk_ids):
        self._read()
        return {cid: list(self._ce.get(cid, [])) for cid in chunk_ids}

    def entity_chunk_counts(self, entity_ids):
        self._read()
        return {eid: len(self._ec.get(eid, ())) for eid in entity_ids}

    def chunks_for_entities(self, entity_ids, *, limit=1000):
        self._read()
        return sorted({cid for eid in entity_ids for cid in self._ec.get(eid, ())})[:limit]
