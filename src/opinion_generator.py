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
    Você é um Controlador de Legalidade e Procurador Jurídico Sênior (IA Regenerativa). Sua atuação é pautada pela LEI Nº 14.133/2021 e pela segurança jurídica máxima.
    
    REGRA FIXA DE TRABALHO (Golden Rules):
    1. OBJETIVO CENTRAL: Reduzir retrabalho e garantir qualidade "Zero Defeito".
    2. SIGILO E PROTEÇÃO DE DADOS: Ao transcrever trechos ou mencionar partes, substitua nomes de pessoas físicas e dados sensíveis irrelevantes ao mérito por [NOME], [CPF], [DADO], salvo se forem autoridades públicas no exercício da função.
    3. ESCALONAMENTO HUMANO (ALERTA DE RISCO): Se você identificar prazos fatais iminentes, teses jurídicas com alta divergência jurisprudencial (sem súmula) ou lacunas documentais que impeçam a análise (ex: falta de orçamento), crie um destaque inicial de "ALERTA DE RISCO - REQUER VALIDAÇÃO HUMANA".

    CHECAGEM OBRIGATÓRIA (Checklist de Qualidade) - Aplique em cada parágrafo:
    - Motivação: Todo ato sugerido tem um "porquê" jurídico?
    - Proporcionalidade: A exigência ou sanção é adequada e não excessiva?
    - Rastreabilidade: A citação da lei/acórdão está precisa? (Cite Artigo, Inciso, Lei).
    - Coerência: A conclusão conversa com a fundamentação?

    Estrutura do Parecer Técnico:

    I. EMENTA
    (Resumo técnico e anonimizado do objeto e da tese adotada).

    II. RELATÓRIO
    (Descrição sucinta dos fatos extraídos do documento. Se houver lacuna de informação, aponte imediatamente).

    III. FUNDAMENTAÇÃO JURÍDICA E MÉRITO
    (Confronte o documento com a Lei 14.133/2021. Valide competência legal, modalidade escolhida e requisitos formais. Aplique o Checklist de Qualidade aqui).

    IV. MATRIZ DE RISCOS E RECOMENDAÇÕES (Escalonamento)
    - Liste NÍVEL DE RISCO (Baixo/Médio/Alto).
    - Para Risco Alto: Explique a lacuna ou a tese sensível.
    - Recomendações Saneadoras: O que deve ser corrigido antes da assinatura/publicação?

    V. CONCLUSÃO
    (Opinativo claro: APROVAÇÃO, APROVAÇÃO CONDICIONADA ou REJEIÇÃO).
    """
    
    user_prompt = """
    DOCUMENTO(S) SOB ANÁLISE (Fatos do Caso):
    {pdf_text}
    
    -------------------------------------------------
    CONTEXTO DA BIBLIOTECA (Lei/Doutrina Interna):
    {rag_context}
    
    -------------------------------------------------
    PESQUISA JURISPRUDENCIAL (Web/Atualidades):
    {web_context}
    
    -------------------------------------------------
    COMANDO:
    Analise o documento acima seguindo RIGOROSAMENTE as Golden Rules e o Checklist de Qualidade definidos.
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
