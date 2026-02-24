from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List

def generate_legal_opinion(
    pdf_text: str, 
    rag_context: str, 
    web_context: str, 
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> str:
    """
    Gera um parecer jurídico com base no texto do PDF, contexto do RAG e busca na web.
    
    Args:
        pdf_text (str): O texto extraído do documento PDF.
        rag_context (str): Contexto extraído do banco de dados vetorial (Lei 14.133, etc).
        web_context (str): Contexto extraído da busca na web (jurisprudência recente).
        model_name (str): Nome do modelo OpenAI a ser utilizado.
        temperature (float): Temperatura para geração (menor = mais determinístico).
        
    Returns:
        str: O parecer jurídico gerado.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    system_prompt = """
    Você é um advogado especialista em Direito Público, com foco em Licitações e Contratos Administrativos, 
    especialmente na Lei Federal nº 14.133/2021.
    
    Sua tarefa é analisar o documento fornecido pelo usuário (um edital, contrato ou processo administrativo) 
    e emitir um Parecer Jurídico detalhado, estruturado e fundamentado.
    
    Estrutura do Parecer:
    1. RELATÓRIO: Resumo dos fatos e do documento analisado.
    2. FUNDAMENTAÇÃO LEGAL: Análise do caso à luz da Lei 14.133/2021 e outras normas pertinentes.
    3. JURISPRUDÊNCIA: Análise de decisões do TCU, STJ, STF e TJs aplicáveis ao caso.
    4. CONCLUSÃO: Parecer final (favorável, desfavorável ou com ressalvas/recomendações).
    
    Utilize o contexto fornecido (Legislação e Jurisprudência) para embasar sua análise.
    Sempre que fizer uma afirmação baseada na lei ou jurisprudência, cite a fonte exata (ex: [Art. 75, Lei 14.133] ou [Acórdão 1234/2023 TCU]).
    Seja objetivo, técnico e utilize linguagem jurídica adequada.
    """
    
    user_prompt = """
    DOCUMENTO ANALISADO:
    {pdf_text}
    
    CONTEXTO LEGAL (Base de Dados):
    {rag_context}
    
    JURISPRUDÊNCIA RECENTE (Busca Web):
    {web_context}
    
    Com base nas informações acima, elabore o Parecer Jurídico.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    # Se o LLM suportar streaming, retornamos o iterador de chunks
    if hasattr(llm, "stream"):
        chain = prompt | llm
        return chain.stream({
            "pdf_text": pdf_text[:15000], # Limitando o tamanho para não estourar o contexto
            "rag_context": rag_context,
            "web_context": web_context
        })
    else:
        chain = prompt | llm
        response = chain.invoke({
            "pdf_text": pdf_text[:15000],
            "rag_context": rag_context,
            "web_context": web_context
        })
        return response.content
