import pytest
from streamlit.testing.v1 import AppTest

def test_app_title():
    """Testa se o título do aplicativo está correto."""
    at = AppTest.from_file("app.py").run(timeout=10)
    assert not at.exception
    assert len(at.title) > 0, "Nenhum título encontrado na interface."
    assert at.title[0].value == "⚖️ Assistente de Pareceres Jurídicos - Licitações e Contratos"

def test_sidebar_elements():
    """Testa se os elementos da barra lateral estão presentes."""
    at = AppTest.from_file("app.py").run(timeout=10)
    assert not at.exception
    assert len(at.sidebar.header) > 0, "Nenhum header encontrado na sidebar."
    assert at.sidebar.header[0].value == "⚙️ Configurações"
    assert len(at.sidebar.text_input) > 0, "Nenhum campo de texto encontrado na sidebar."
    assert at.sidebar.text_input[0].label == "OpenAI API Key"
