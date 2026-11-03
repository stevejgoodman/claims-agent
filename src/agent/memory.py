"""The agent's long-term memory: an in-memory langgraph store with semantic search, and the langmem tools over it."""

from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

NAMESPACE = ("fact",)
SEARCH_TOOL = "search_memory"  # the name langmem gives its search tool; run_claim reads what this tool returned
EMBEDDING_MODEL = "openai:text-embedding-3-small"
EMBEDDING_DIMS = 1536


def build_store(knowledge: list[dict]) -> InMemoryStore:
    """A store holding the knowledge facts, indexed for semantic search."""
    store = InMemoryStore(index={"embed": init_embeddings(EMBEDDING_MODEL), "dims": EMBEDDING_DIMS})
    for mem_id, entry in enumerate(knowledge):
        store.put((*NAMESPACE, entry["topic"]), mem_id, {"text": entry["fact"]})
    return store


def memory_tools() -> list:
    return [create_manage_memory_tool(namespace=NAMESPACE), create_search_memory_tool(namespace=NAMESPACE)]
