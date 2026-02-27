import io
import os
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_file(file_obj, file_name: str) -> str:
    """
    Extrai o texto de vários formatos de arquivo.
    
    Args:
        file_obj: O arquivo em formato de bytes (io.BytesIO).
        file_name (str): O nome do arquivo com a extensão.
        
    Returns:
        str: O texto extraído do arquivo.
    """
    ext = file_name.split('.')[-1].lower()
    text = ""
    
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            
        if ext == 'pdf':
            reader = PdfReader(file_obj)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        elif ext in ['txt', 'csv']:
            text = file_obj.read().decode('utf-8', errors='ignore')
            
        elif ext == 'docx':
            import docx
            doc = docx.Document(file_obj)
            text = "\n".join([para.text for para in doc.paragraphs])
            
        elif ext in ['xlsx', 'xls']:
            import pandas as pd
            df = pd.read_excel(file_obj)
            text = df.to_string()
            
        elif ext == 'rtf':
            from striprtf.striprtf import rtf_to_text
            rtf_content = file_obj.read().decode('utf-8', errors='ignore')
            text = rtf_to_text(rtf_content)
            
        elif ext in ['png', 'jpeg', 'jpg', 'webp', 'heic']:
            from PIL import Image
            import pytesseract
            
            # Para HEIC, precisamos de uma biblioteca extra
            if ext == 'heic':
                from pillow_heif import register_heif_opener
                register_heif_opener()
                
            img = Image.open(file_obj)
            text = pytesseract.image_to_string(img, lang='por') # Assume português
            
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")
            
        return text
        
    except ImportError as e:
        raise ImportError(f"Falta uma biblioteca para processar o arquivo {ext}. Erro: {str(e)}")
    except Exception as e:
        raise ValueError(f"Erro ao processar o arquivo {file_name}: {str(e)}")

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Divide o texto em pedaços menores para processamento pelo LLM.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks
