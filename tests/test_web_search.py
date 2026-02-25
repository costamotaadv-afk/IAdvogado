import pytest
from src.web_search import search_jurisprudence
from unittest.mock import patch, MagicMock

@patch('src.web_search.DDGS')
def test_search_jurisprudence(MockDDGS):
    # Mock do context manager do DDGS
    mock_instance = MagicMock()
    MockDDGS.return_value.__enter__.return_value = mock_instance
    
    # Simula retorno de resultados da busca
    mock_results = [{
        'title': 'Decisão TCU',
        'href': 'http://tcu.gov.br',
        'body': 'Acórdão sobre dispensa.'
    }]
    
    # Configura o retorno do método .text()
    mock_instance.text.return_value = mock_results
    
    result = search_jurisprudence("dispensa licitação")
    
    assert "Decisão TCU" in result
    assert "http://tcu.gov.br" in result
    
    # Verifica se foi chamado 3 vezes (uma para cada estratégia)
    assert mock_instance.text.call_count == 3
