
from embeddings.embedder import Embedder
from vectorstore.chroma_manager import ChromaManager
from config.settings import TOP_K, SIMILARITY_THRESHOLD


class Retriever:
    """Retrieve relevant document chunks for a user question."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vectorstore: ChromaManager | None = None,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if similarity_threshold < 0:
            raise ValueError("similarity_threshold cannot be negative.")

        self.embedder = embedder or Embedder()
        self.vectorstore = vectorstore or ChromaManager()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold



    def retrieve(self, question: str) -> list[dict]:
        """Retrieve relevant chunks and filter weak matches."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        query_embedding = self.embedder.embed_query(question.strip())

        results = self.vectorstore.search(
            query_embedding=query_embedding,
            n_results=self.top_k,
        )

        filtered_results = [
            result
            for result in results
            if result.get("distance") is not None
            and result["distance"] <= self.similarity_threshold
        ]

        return filtered_results
