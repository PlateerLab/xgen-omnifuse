"""Same OmniFuse algorithm, but over a real Apache Jena Fuseki store (graph-only).

    python examples/fuseki_example.py
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnifuse import InMemoryVector, OmniFuse  # noqa: E402
from omnifuse.backends.fuseki import FusekiGraph  # noqa: E402

QUERY_URL = "http://localhost:23030/xgen/query"
GRAPH = "https://w3id.org/xgen/collection/jeju_bank_ecfd6347-6c5e-4ada-a140-239ce7577e0a"

graph = FusekiGraph(QUERY_URL, GRAPH, user="admin", password="admin123")
of = OmniFuse(graph, InMemoryVector([]), llm=__import__("omnifuse").EchoLLM())  # graph-only

r = of.search("담보")
print("RELATIONS:", r.relations[:8])
print("CLASS SEED:", r.class_seed[:200])
print("EVIDENCE NODES:", r.evidence_nodes[:12])
