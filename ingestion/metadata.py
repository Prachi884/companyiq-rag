from pathlib import Path
import hashlib


def generate_document_id(file_path: str) -> str:
    """
    Generate a deterministic document ID from the file contents.

    The same file will always produce the same ID.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    file_hash = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            file_hash.update(block)

    return file_hash.hexdigest()


def create_chunk_metadata(
    file_path: str,
    document_type: str,
    chunk_id: int,
    page_number: int | None = None,
    section: str | None = None,
) -> dict:
    """
    Create metadata for a document chunk.

    Metadata is based only on information actually available
    from the document-processing pipeline.
    """
    path = Path(file_path)

    return {
        "document_id": generate_document_id(file_path),
        "document_name": path.name,
        "document_type": document_type,
        "page_number": page_number,
        "chunk_id": chunk_id,
        "section": section,
    }
