"""FusekiGraph — GraphStore backed by any SPARQL 1.1 endpoint (Apache Jena Fuseki, …).

Stdlib-only (urllib) → no extra dependency. The *same* OmniFuse algorithm that runs on
InMemoryGraph runs unchanged here, proving the logic is backend-agnostic. Label search
uses portable ``FILTER(CONTAINS(...))`` so it works on ANY SPARQL store (not only
jena-text); pair with a real VectorStore, or pass an empty one for graph-only mode.
"""
from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from typing import Optional

from ..models import Node
from ..text import tokenize

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
_OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"


class FusekiGraph:
    def __init__(self, query_url: str, graph_uri: Optional[str] = None, *,
                 user: Optional[str] = None, password: Optional[str] = None, timeout: float = 30):
        self.query_url = query_url
        self.graph_uri = graph_uri
        self.timeout = timeout
        self._auth = (
            "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
            if user and password else None
        )

    def _g(self, body: str) -> str:
        return f"GRAPH <{self.graph_uri}> {{ {body} }}" if self.graph_uri else body

    def _query(self, sparql: str) -> dict:
        data = urllib.parse.urlencode({"query": sparql}).encode()
        req = urllib.request.Request(self.query_url, data=data, headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        })
        if self._auth:
            req.add_header("Authorization", self._auth)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def _local(uri: str) -> str:
        return uri.split("#")[-1].split("/")[-1]

    @staticmethod
    def _val(b: dict, key: str) -> str:
        return b.get(key, {}).get("value", "")

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        terms = [t for t in tokenize(query) if len(t) >= 2][:8]
        if not terms:
            return []
        filt = " || ".join(f'CONTAINS(LCASE(STR(?l)), "{t}")' for t in terms)
        body = (f'?n <{_RDFS_LABEL}> ?l . OPTIONAL {{ ?n <{_RDF_TYPE}> ?ty . FILTER(?ty = <{_OWL_CLASS}>) }} '
                f'FILTER({filt})')
        q = f"SELECT DISTINCT ?n ?l ?ty WHERE {{ {self._g(body)} }} LIMIT {limit}"
        out = []
        for b in self._query(q).get("results", {}).get("bindings", []):
            nid, lab = self._val(b, "n"), self._val(b, "l")
            kind = "class" if self._val(b, "ty") else "instance"
            # rank by how many query terms the label contains
            ll = lab.lower()
            score = float(sum(t in ll for t in terms)) or 1.0
            out.append((Node(nid, lab, kind), score))
        out.sort(key=lambda x: -x[1])
        return out

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        body = (f'?i (<{_RDF_TYPE}>|<{_RDFS_SUBCLASS}>) <{class_id}> . '
                f'OPTIONAL {{ ?i <{_RDFS_LABEL}> ?l }}')
        q = f"SELECT DISTINCT ?i ?l WHERE {{ {self._g(body)} }} LIMIT {limit}"
        out = []
        for b in self._query(q).get("results", {}).get("bindings", []):
            iid = self._val(b, "i")
            out.append(Node(iid, self._val(b, "l") or self._local(iid), "instance"))
        return out

    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]:
        body = (f'{{ BIND(<{node_id}> AS ?s) ?s ?p ?o }} UNION {{ BIND(<{node_id}> AS ?o) ?s ?p ?o }} '
                f'OPTIONAL {{ ?s <{_RDFS_LABEL}> ?sl }} OPTIONAL {{ ?o <{_RDFS_LABEL}> ?ol }} '
                f'FILTER(?p != <{_RDF_TYPE}> && ?p != <{_RDFS_LABEL}>)')
        q = f"SELECT ?s ?sl ?p ?o ?ol WHERE {{ {self._g(body)} }} LIMIT {limit}"
        out = []
        for b in self._query(q).get("results", {}).get("bindings", []):
            sl = self._val(b, "sl") or self._local(self._val(b, "s"))
            ol = self._val(b, "ol") or (self._val(b, "o") if b.get("o", {}).get("type") == "literal"
                                        else self._local(self._val(b, "o")))
            out.append((sl, self._local(self._val(b, "p")), ol))
        return out

    def neighbor_ids(self, node_id: str, *, limit: int = 100) -> list[str]:
        body = (f'{{ BIND(<{node_id}> AS ?s) ?s ?p ?o }} UNION {{ BIND(<{node_id}> AS ?o) ?s ?p ?o }} '
                f'FILTER(?p != <{_RDF_TYPE}> && ?p != <{_RDFS_LABEL}>) '
                f'BIND(IF(?s = <{node_id}>, ?o, ?s) AS ?n) FILTER(isIRI(?n))')
        q = f"SELECT DISTINCT ?n WHERE {{ {self._g(body)} }} LIMIT {limit}"
        return [self._val(b, "n") for b in self._query(q).get("results", {}).get("bindings", []) if self._val(b, "n")]

    def count_class(self, class_id: str) -> int:
        body = f'?i (<{_RDF_TYPE}>|<{_RDFS_SUBCLASS}>) <{class_id}>'
        q = f"SELECT (COUNT(DISTINCT ?i) AS ?c) WHERE {{ {self._g(body)} }}"
        bs = self._query(q).get("results", {}).get("bindings", [])
        return int(bs[0]["c"]["value"]) if bs else 0

    def get_node(self, node_id: str) -> Optional[Node]:
        q = f"SELECT ?l WHERE {{ {self._g(f'<{node_id}> <{_RDFS_LABEL}> ?l')} }} LIMIT 1"
        bs = self._query(q).get("results", {}).get("bindings", [])
        return Node(node_id, (bs[0]["l"]["value"] if bs else self._local(node_id)))
