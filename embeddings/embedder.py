from fastembed import TextEmbedding

from config.settings import EMBEDDING_MODEL


class Embedder:
    """
    Wrapper around FastEmbed.

    The rest of CompanyIQ interacts with this class instead of
    directly depending on FastEmbed internals.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple document chunks.
        """
        if not texts:
            return []

        embeddings = self.model.embed(texts)

        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single user query.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")

        embedding = next(
            self.model.embed([text])
        )

        return embedding.tolist()
