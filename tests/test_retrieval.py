
import pytest

from retrieval.retriever import Retriever


class FakeEmbedder:
    """Provide predictable embeddings without loading a real model."""

    def embed_query(self, question: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    """Return predefined search results for testing."""

    def __init__(self, results: list[dict]):
        self.results = results
        self.last_query_embedding = None
        self.last_n_results = None

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        self.last_query_embedding = query_embedding
        self.last_n_results = n_results
        return self.results


@pytest.fixture
def sample_results():
    return [
        {
            "id": "chunk-1",
            "text": "Relevant product specification.",
            "metadata": {"source": "technical.pdf"},
            "distance": 0.10,
        },
        {
            "id": "chunk-2",
            "text": "Possibly relevant information.",
            "metadata": {"source": "policy.pdf"},
            "distance": 0.30,
        },
        {
            "id": "chunk-3",
            "text": "Unrelated information.",
            "metadata": {"source": "other.pdf"},
            "distance": 0.60,
        },
    ]


def test_retrieve_filters_results_by_distance(sample_results):
    vectorstore = FakeVectorStore(sample_results)
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vectorstore=vectorstore,
        top_k=5,
        similarity_threshold=0.35,
    )

    results = retriever.retrieve("What is the product specification?")

    assert [result["id"] for result in results] == [
        "chunk-1",
        "chunk-2",
    ]
    assert all(result["distance"] <= 0.35 for result in results)


def test_retrieve_returns_empty_list_when_no_results_match():
    vectorstore = FakeVectorStore([
        {
            "id": "chunk-1",
            "text": "Unrelated information.",
            "metadata": {},
            "distance": 0.80,
        }
    ])

    retriever = Retriever(
        embedder=FakeEmbedder(),
        vectorstore=vectorstore,
        similarity_threshold=0.35,
    )

    assert retriever.retrieve("Question about a product") == []


def test_retrieve_rejects_empty_question():
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vectorstore=FakeVectorStore([]),
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.retrieve("")


def test_retrieve_rejects_whitespace_question():
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vectorstore=FakeVectorStore([]),
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.retrieve("   ")


def test_retrieve_passes_question_embedding_and_top_k(sample_results):
    vectorstore = FakeVectorStore(sample_results)
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vectorstore=vectorstore,
        top_k=3,
        similarity_threshold=1.0,
    )

    retriever.retrieve("Explain the product.")

    assert vectorstore.last_query_embedding == [0.1, 0.2, 0.3]
    assert vectorstore.last_n_results == 3
