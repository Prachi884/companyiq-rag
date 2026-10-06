import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving its meaning.

    The cleaner:
    - normalizes line endings
    - removes excessive spaces
    - removes excessive blank lines
    - trims whitespace from each line
    - preserves paragraph structure
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in lines]

    # Collapse repeated spaces and tabs within a line
    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line)

    # Rebuild the text
    text = "\n".join(cleaned_lines)

    # Collapse more than two consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove whitespace around the entire document
    return text.strip()
