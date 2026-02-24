import pytest
from src.web_search import search_jurisprudence
from unittest.mock import patch

@patch('src.web_search.DuckDuckGoSearchRun')
def test_search_jurisprudence(MockSearch):
    mock_instance = MockSearch.return_value
    mock_instance.run.return_value = "Decisão do TCU sobre dispensa de licitação."
    
    result = search_jurisprudence("TCU dispensa licitação")
    
    assert "Decisão do TCU" in result
    mock_instance.run.assert_called_once_with("TCU dispensa licitação")
