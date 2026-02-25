import pytest
from src.opinion_generator import generate_legal_opinion
from unittest.mock import patch, MagicMock

@patch('src.opinion_generator.ChatOpenAI')
def test_generate_legal_opinion(MockChatOpenAI):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Parecer Jurídico: Favorável."
    mock_llm.invoke.return_value = mock_response
    
    # O ChatOpenAI é instanciado dentro da função, então precisamos mockar a classe
    MockChatOpenAI.return_value = mock_llm
    
    # Mockando o comportamento da chain (prompt | llm)
    mock_chain = MagicMock()
    # Como a função agora usa stream(), precisamos mockar o retorno do stream
    mock_chunk = MagicMock()
    mock_chunk.content = "Parecer Jurídico: Favorável."
    mock_chain.stream.return_value = [mock_chunk]
    
    with patch('src.opinion_generator.ChatPromptTemplate.from_messages') as mock_prompt:
        # Fazendo com que a operação prompt | llm retorne nossa mock_chain
        mock_prompt.return_value.__or__.return_value = mock_chain
        
        pdf_text = "Edital de licitação para compra de computadores."
        rag_context = "Art. 75 da Lei 14.133/2021."
        web_context = "Acórdão 1234/2023 TCU."
        user_query = "Analise a legalidade do aviso."

        stream_result = generate_legal_opinion(pdf_text, rag_context, web_context, user_query)
        
        # Junta os chunks do stream para verificar o resultado final
        result = "".join([chunk.content for chunk in stream_result])
        
        assert "Parecer Jurídico: Favorável." in result
    # mock_llm.invoke.assert_called_once() # A chain é invocada, não o llm diretamente

@patch('src.opinion_generator.ChatOpenAI')
def test_generate_legal_opinion_system_prompt_rules(MockChatOpenAI):
    mock_llm = MagicMock()
    MockChatOpenAI.return_value = mock_llm
    
    mock_chain = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Teste"
    mock_chain.stream.return_value = [mock_chunk]
    
    with patch('src.opinion_generator.ChatPromptTemplate.from_messages') as mock_prompt:
        mock_prompt.return_value.__or__.return_value = mock_chain
        
        generate_legal_opinion("", "", "", "")
        
        # Verifica se from_messages foi chamado
        mock_prompt.assert_called_once()
        
        # Pega os argumentos passados para from_messages
        messages = mock_prompt.call_args[0][0]
        system_message = messages[0][1]
        
        # Verifica se as regras de ausência de documentos estão no prompt
        assert "REGRA ESPECIAL PARA AUSÊNCIA DE DOCUMENTOS:" in system_message
        assert "8. Apêndice — Matriz de Evidências e Índice de Anexos" in system_message
        assert "8.1 Matriz de evidências (NEGATIVA — “deveria existir”)" in system_message
        assert "8.2 Índice de Anexos (Vazio)" in system_message
        assert "Cenário A — Lacuna de documentação essencial (Lacuna Real)" in system_message
        assert "Cenário B — Falha do sistema de anexos/biblioteca (Lacuna de acesso/indexação)" in system_message
        assert "7.1 Regularização do acervo (Lista mínima de documentos por tipo de caso)" in system_message
        assert "2.2 Método aplicado e Log de Busca Interna" in system_message
