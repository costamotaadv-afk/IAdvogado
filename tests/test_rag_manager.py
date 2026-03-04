import pytest
from src.rag_manager import RAGManager
from unittest.mock import patch, MagicMock

@patch('src.rag_manager.OpenAIEmbeddings')
@patch('src.rag_manager.Chroma')
def test_rag_manager_initialization(MockChroma, MockEmbeddings):
    manager = RAGManager(persist_directory="./test_db")
    assert manager.persist_directory == "./test_db"
    MockChroma.assert_called_once()

@patch('src.rag_manager.OpenAIEmbeddings')
@patch('src.rag_manager.Chroma')
def test_add_documents(MockChroma, MockEmbeddings):
    manager = RAGManager(persist_directory="./test_db")
    manager.vectorstore.add_documents.reset_mock()
    
    texts = ["Texto 1", "Texto 2"]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}]
    
    manager.add_documents(texts, metadatas)
    
    # Verifica se o método add_documents do Chroma foi chamado
    manager.vectorstore.add_documents.assert_called_once()
    # O persist não é chamado explicitamente pelo método add_documents

@patch('src.rag_manager.OpenAIEmbeddings')
@patch('src.rag_manager.Chroma')
def test_search_similar(MockChroma, MockEmbeddings):
    manager = RAGManager(persist_directory="./test_db")
    
    # Mock do retorno da busca
    mock_doc = MagicMock()
    mock_doc.page_content = "Resultado da busca"
    manager.vectorstore.similarity_search.return_value = [mock_doc]
    
    results = manager.search_similar("consulta", k=1)
    
    assert len(results) == 1
    assert results[0].page_content == "Resultado da busca"
    manager.vectorstore.similarity_search.assert_called_with("consulta", k=8)
