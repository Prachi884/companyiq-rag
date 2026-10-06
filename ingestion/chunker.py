from config.settings import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    The chunker tries to split at sensible boundaries such as:
    - paragraph breaks
    - line breaks
    - spaces

    This helps keep related information together.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = text.strip()

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # If this is not the final chunk, look for a sensible
        # boundary before the maximum chunk size.
        if end < text_length:
            boundary = text.rfind("\n\n", start, end)

            if boundary <= start:
                boundary = text.rfind("\n", start, end)

            if boundary <= start:
                boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - chunk_overlap

        # Prevent the loop from getting stuck.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks
