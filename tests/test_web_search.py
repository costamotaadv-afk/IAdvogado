import pytest
from src.web_search import search_jurisprudence
from unittest.mock import patch

@patch('src.web_search.DuckDuckGoSearchRun')
def test_search_jurisprudence(MockSearch):
    mock_instance = MockSearch.return_value
    mock_instance.run.return_value = "Decisão do TCU sobre dispensa de licitação."
    
    result = search_jurisprudence("dispensa licitação")
    
    assert "Decisão do TCU" in result
    # Verifica se foi chamada 3 vezes (uma para cada camada de busca)
    assert mock_instance.run.call_count == 3
