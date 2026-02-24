import io
from pypdf import PdfReader
from typing import List

def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    """
    Extrai o texto de um arquivo PDF.
    
    Args:
        pdf_file (io.BytesIO): O arquivo PDF em formato de bytes.
        
    Returns:
        str: O texto extraído do PDF.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Erro ao processar o PDF: {str(e)}")

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Divide o texto em pedaços menores para processamento pelo LLM.
    
    Args:
        text (str): O texto completo.
        chunk_size (int): Tamanho máximo de cada pedaço.
        chunk_overlap (int): Sobreposição entre os pedaços.
        
    Returns:
        List[str]: Lista de pedaços de texto.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks
