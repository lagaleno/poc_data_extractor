# app/utils/pdf_loader.py
import io
from typing import Optional

def extract_text_from_pdf(file_bytes: bytes, max_pages: Optional[int] = None) -> str:
    try:
        import pdfplumber
    except ImportError:
        # deixa a mensagem para a UI decidir como exibir
        return ""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for p in pages:
            try:
                text_chunks.append(p.extract_text() or "")
            except Exception:
                pass
    return "\n".join(text_chunks)
