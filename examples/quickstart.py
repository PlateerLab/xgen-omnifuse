"""OmniFuse quickstart — zero infra, zero API key.

    python examples/quickstart.py
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서도 한글 출력
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, Node, Triple, build_inmemory  # noqa: E402

nodes = [
    Node("c_reg", "규정", kind="class"),
    Node("n_audit", "감사규정", kind="instance"),
    Node("n_race76", "경마시행규정 제76조", kind="instance"),
    Node("n_info", "정보화업무처리규정", kind="instance"),
    Node("n_illegal", "불법도박", kind="instance"),
]
triples = [
    Triple("n_audit", "instanceOf", "c_reg"),
    Triple("n_race76", "instanceOf", "c_reg"),
    Triple("n_info", "instanceOf", "c_reg"),
    Triple("n_illegal", "관련규정", "n_race76"),
]
chunks = [
    Chunk("ch1", "불법도박(불법사설경마)은 한국마사회법 제48조에 따라 단속되며 경마시행규정 제76조가 적용된다.",
          entities=["n_illegal", "n_race76"]),
    Chunk("ch2", "감사규정은 내부 감사의 절차와 권한을 규정한다.", entities=["n_audit"]),
    Chunk("ch3", "정보화업무처리규정은 정보시스템 운영 기준을 정한다.", entities=["n_info"]),
]

of = build_inmemory(nodes, triples, chunks)   # InMemoryGraph + InMemoryVector + EchoLLM
r = of.search("불법도박 관련 규정")

print("Q:", r.question)
print("\nANSWER:\n", r.answer)
print("\nRELATIONS:", r.relations)
print("EVIDENCE NODES (cited):", r.evidence_nodes)
print("CLASS SEED:", r.class_seed[:120])
print("CHUNKS:", [c.id for c in r.chunks])
