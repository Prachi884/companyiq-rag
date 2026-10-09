
from pathlib import Path

import chromadb

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
)


class ChromaManager:
    """Manage CompanyIQ's persistent ChromaDB vector database."""

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.persist_directory = str(Path(persist_directory))
        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def list_ids(self) -> list[str]:
        """Return IDs of stored chunks."""
        result = self.collection.get()
        return result["ids"]

    def delete_all(self) -> None:
        """Remove all chunks from the collection."""
        ids = self.list_ids()

        if ids:
            self.collection.delete(ids=ids)


    def add_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Store document chunks, embeddings, and metadata."""
        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "IDs, texts, embeddings, and metadatas must have equal lengths."
            )

        if not ids:
            return

        if any(not text.strip() for text in texts):
            raise ValueError("Chunk texts cannot be empty.")

        if len(set(ids)) != len(ids):
            raise ValueError("Chunk IDs must be unique within a batch.")

        safe_metadatas = [
            {
                key: value
                for key, value in metadata.items()
                if value is not None
            }
            for metadata in metadatas
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=safe_metadatas,
        )


    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Search for the most similar stored chunks."""
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if n_results <= 0:
            raise ValueError("n_results must be greater than 0.")

        total_chunks = self.count()
        if total_chunks == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total_chunks),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for i, chunk_id in enumerate(result["ids"][0]):
            matches.append({
                "id": chunk_id,
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
            })

        return matches


    def list_documents(self) -> list[dict]:
        """Return unique documents represented in the collection."""
        result = self.collection.get(
            include=["metadatas"]
        )

        documents = {}
        for metadata in result["metadatas"]:
            document_id = metadata.get("document_id")
            document_name = metadata.get("document_name")

            key = document_id or document_name
            if not key:
                continue

            if key not in documents:
                documents[key] = {
                    "document_id": document_id,
                    "document_name": document_name,
                    "document_type": metadata.get("document_type"),
                    "chunk_count": 0,
                }

            documents[key]["chunk_count"] += 1

        return list(documents.values())



    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document ID."""
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        result = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        chunk_ids = result["ids"]

        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

        return len(chunk_ids)
