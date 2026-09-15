"""Cross-process package conformance using two independently installed wheels.

python tools/check_package_pair.py --producer /tmp/build-env/bin/python --consumer /tmp/search-env/bin/python
Neither virtual environment may contain the other library.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

PRODUCER = '''
import importlib.util, sys
assert importlib.util.find_spec("omnifuse") is None
from xgen_ontology import build_knowledge
from xgen_ontology.knowledge import KnowledgeBundle, Resource, SourceChunk, EmbeddingProfile, ChunkEmbedding
source = KnowledgeBundle("pair", "source-v1",
    resources=(Resource("root", "v1", "root", "directory"), Resource("file", "v1", "colors.csv", parent_id="root")),
    chunks=(SourceChunk("chunk:v1", "file", "v1", "csv-v1", "id,name\\n1,Red\\n2,Blue"),),
    profiles=(EmbeddingProfile("profile-v1", "test-encoder", "fixed-v1", 2),),
    embeddings=(ChunkEmbedding("chunk:v1", "profile-v1", (1., .5)),),
    components=("hierarchy", "content", "embeddings"))
built = build_knowledge(source, snapshot_id="built-v1")
built.dump(sys.argv[1])
assert "xgen_ontology.search.oneshot" not in sys.modules
print("build-only: graph artifact saved")
'''
CONSUMER = '''
import importlib.util, sys
assert importlib.util.find_spec("xgen_ontology") is None
from omnifuse import KnowledgeSearch, MemoryKnowledgeProvider, ReadScope
from omnifuse.knowledge import KnowledgeBundle
bundle = KnowledgeBundle.load(sys.argv[1])
result = KnowledgeSearch(MemoryKnowledgeProvider(bundle), query_encoder=lambda q: [1., .5],
    query_profile=bundle.profiles[0]).search("Red", scope=ReadScope(root_ids=("root",)))
assert result.retrieval_mode == "hybrid"
assert result.facts and result.citations
assert result.citations[0].resource_id == "file"
assert result.citations[0].revision == "v1"
assert result.ranked_chunks[0][0].id == "chunk:v1"
print("search-only: hierarchy + embeddings + graph + exact source revision verified")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--consumer", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "knowledge.json")
        for executable, code in ((args.producer, PRODUCER), (args.consumer, CONSUMER)):
            subprocess.run([executable, "-I", "-c", code, path], cwd=directory, check=True)


if __name__ == "__main__":
    main()
