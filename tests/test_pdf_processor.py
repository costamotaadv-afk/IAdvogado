import pytest
import io
from src.pdf_processor import extract_text_from_pdf, split_text_into_chunks

def test_extract_text_from_pdf():
    # Cria um PDF em memória simples usando reportlab (se disponível) ou mock
    # Como não temos reportlab no requirements, vamos mockar o PdfReader
    from unittest.mock import patch, MagicMock
    
    with patch('src.pdf_processor.PdfReader') as MockReader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Texto de teste do PDF."
        
        mock_reader_instance = MockReader.return_value
        mock_reader_instance.pages = [mock_page]
        
        dummy_pdf = io.BytesIO(b"dummy pdf content")
        text = extract_text_from_pdf(dummy_pdf)
        
        assert "Texto de teste do PDF." in text

def test_split_text_into_chunks():
    text = "A" * 2500
    chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200)
    
    assert len(chunks) > 1
    assert len(chunks[0]) <= 1000
    assert len(chunks[1]) <= 1000
