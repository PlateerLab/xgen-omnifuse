"""Search an external bundle: python examples/knowledge_search.py knowledge.json."""
import sys

from omnifuse import KnowledgeSearch, MemoryKnowledgeProvider, ReadScope
from omnifuse.knowledge import KnowledgeBundle

bundle = KnowledgeBundle.load(sys.argv[1])
engine = KnowledgeSearch(MemoryKnowledgeProvider(bundle), mode="lexical")
result = engine.search("Red", scope=ReadScope(root_ids=("root",)))
print(result.result.relations)
for citation in result.citations:
    print(citation.resource_id, citation.revision, citation.locator)
