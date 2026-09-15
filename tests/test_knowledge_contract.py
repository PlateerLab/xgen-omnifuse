"""Consumer conformance without importing/installing the build package."""
import hashlib
import importlib.resources
import json
import subprocess
import sys
from pathlib import Path

from omnifuse import KnowledgeSearch, MemoryKnowledgeProvider
from omnifuse.knowledge import KnowledgeBundle


def test_packaged_contract_digests():
    package = importlib.resources.files("omnifuse")
    lock = json.loads(package.joinpath("knowledge.lock.json").read_text())
    for filename, key in (("knowledge.py", "records_sha256"), ("knowledge.schema.json", "schema_sha256")):
        assert hashlib.sha256(package.joinpath(filename).read_bytes()).hexdigest() == lock[key]


def test_independent_build_artifact_is_searchable():
    fixture = Path(__file__).with_name("fixtures-knowledge-v1.json")
    data = KnowledgeBundle.load(fixture)
    result = KnowledgeSearch(MemoryKnowledgeProvider(data)).search("Red")
    assert result.facts and result.citations
    assert result.citations[0].revision == "v1"
    assert "xgen_ontology" not in sys.modules


def test_standalone_consumer_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fixture = Path(__file__).with_name("fixtures-knowledge-v1.json")
    code = f'''
import sys
sys.path.insert(0, {str(root / "src")!r})
from omnifuse import KnowledgeSearch, MemoryKnowledgeProvider
from omnifuse.knowledge import KnowledgeBundle
bundle = KnowledgeBundle.load({str(fixture)!r})
result = KnowledgeSearch(MemoryKnowledgeProvider(bundle)).search("Red")
assert result.citations[0].revision == "v1"
assert "xgen_ontology" not in sys.modules
'''
    subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, check=True)
