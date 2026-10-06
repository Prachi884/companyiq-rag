from pathlib import Path

import pymupdf
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.
    """
    documents = []

    pdf = pymupdf.open(file_path)

    try:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    {
                        "text": text,
                        "page_number": page_number,
                    }
                )
    finally:
        pdf.close()

    return documents


def load_docx(file_path: str) -> list[dict]:
    """
    Extract text from a DOCX document.

    DOCX files do not have reliable page information at this stage,
    so page_number is set to None.
    """
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    if not full_text:
        return []

    return [
        {
            "text": full_text,
            "page_number": None,
        }
    ]


def load_txt(file_path: str) -> list[dict]:
    """
    Extract text from a TXT file.
    """
    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "page_number": None,
        }
    ]


def load_document(file_path: str) -> list[dict]:
    """
    Load a supported document based on its file extension.

    Supported formats:
    - PDF
    - DOCX
    - TXT
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )
