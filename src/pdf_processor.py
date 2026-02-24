from pypdf import PdfReader
import io


def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file object (e.g., Streamlit UploadedFile or file-like)."""
    if isinstance(file, (str, bytes)):
        reader = PdfReader(io.BytesIO(file) if isinstance(file, bytes) else file)
    else:
        reader = PdfReader(file)

    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n\n".join(text_parts)


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for embedding and retrieval."""
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks
