"""Memory demo — remember facts/notes, recall via fusion search. Zero infra.

    python examples/memory_example.py
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Memory  # noqa: E402

m = Memory()
m.add_fact("담보", "instanceOf", "규정")
m.add_fact("감사규정", "instanceOf", "규정")
m.remember("담보 한도는 5억원이며 정규담보비율 60%가 적용된다.",
           triples=[("담보", "한도", "5억"), ("담보", "정규담보비율", "60%")])
m.remember("감사규정은 내부 감사 절차를 정한다.")   # auto-links to 감사규정

print("stats:", m.stats())
print("\n[recall: 담보 한도]\n", m.recall("담보 한도").answer[:200])
print("\n[recall: 규정 전부]", m.recall("규정 전부").class_seed[:90])
