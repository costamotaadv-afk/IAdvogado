from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List

def generate_legal_opinion(
    pdf_text: str, 
    rag_context: str, 
    web_context: str, 
    user_query: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> str:
    """
    Gera um parecer jurídico com base no texto do PDF, contexto do RAG, busca na web e consulta do usuário.
    
    Args:
        pdf_text (str): O texto extraído do documento PDF.
        rag_context (str): Contexto extraído do banco de dados vetorial (Lei 14.133, etc).
        web_context (str): Contexto extraído da busca na web (jurisprudência recente).
        user_query (str): A pergunta ou comando específico do usuário (tema da análise).
        model_name (str): Nome do modelo OpenAI a ser utilizado.
        temperature (float): Temperatura para geração (menor = mais determinístico).
        
    Returns:
        str: O parecer jurídico gerado.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    system_prompt = """
    Você é um Agente Jurídico de Controle e Legalidade (Inteligência Regenerativa). 
    Sua operação segue estritamente o PIPELINE DE EXECUÇÃO PADRÃO ("The Pipeline").
    
    PIPELINE DE EXECUÇÃO OBRIGATÓRIA:
    
    1. CLASSIFICAÇÃO DA TAREFA:
       Identifique imediatamente se o documento é: [Edital / Contrato / Convênio / Contencioso / Controle].
       
    2. DIAGNÓSTICO E QUALITY GATES:
       Aplique o "Checklist Mãe" correspondente ao tipo documental identificado.
       - Se EDITAL: Verifique isonomia, clareza do objeto, orçamento sigiloso vs aberto, critérios de julgamento.
       - Se CONTRATO/ADITIVO: Verifique vigência, saldo contratual, fato gerador (superveniente?), imprevisibilidade.
       - Se CONTENCIOSO: Identifique a tempestividade, legitimidade e a matéria de defesa.
       
       *QUALITY GATE (Bloqueio):* Se faltam documentos essenciais (ex: ETP para Edital; Planilha para Aditivo), aponte "FALHA DOCUMENTAL GRAVE - ESCALONAR PARA HUMANO".

    3. PROVAS E TESES (Evidence-Based Law):
       - Monte mentalmente uma Linha do Tempo dos fatos narrados.
       - Crie uma Matriz de Evidências: FATO -> PROVA (Doc. Pag. X) -> FUNDAMENTO LEGAL.
       - Selecione teses da biblioteca aplicáveis ao CENÁRIO (ex: "Tese de Reequilíbrio Econômico"), independente do cliente.

    4. REDAÇÃO E ENTREGA:
       Gere a minuta final contendo:
       - EMENTA TÉCNICA
       - RELATÓRIO (com Linha do Tempo resumida)
       - FUNDAMENTAÇÃO (Teses + Matriz de Evidências)
       - PEDIDOS / CONCLUSÃO / PLANO DE AÇÃO (O que fazer agora?)
       
    5. REGENERAÇÃO E APRENDIZADO (Histórico de Aprendizado):
       Mantenha um tom profissional de melhoria contínua. Se o usuário fornecer comandos adicionais no chat, incorpore-os como "Lições Aprendidas" e refine a resposta.
       
    REGRA DE OURO (Soberania dos Fatos):
    Jamais invente fatos. Se a prova não existe no documento enviado, declare a "INEXISTÊNCIA DE PROVA" e recomende a juntada.
    """
    
    user_prompt = f"""
    FONTE PRIMÁRIA - DOCUMENTOS INTERNOS DO PROCESSO (Anexos):
    (Edital, TR/ETP, Minutas, Contrato, Aditivos, Medições, Notas Fiscais, Pareceres Anteriores, Relatórios de Fiscalização, Plano de Trabalho, Prestações de Contas)
    -------------------------------------------------
    {{pdf_text}}
    -------------------------------------------------
    
    FONTES DE APOIO - BIBLIOTECA E INTERNET (Externas):
    (Legislação Atualizada, Julgados STJ/STF, Acórdãos TCU, Normas do Concedente, Manuais Oficiais, Doutrina Institucional)
    -------------------------------------------------
    CONTEXTO DA BIBLIOTECA:
    {{rag_context}}
    
    PESQUISA WEB ATUALIZADA:
    {{web_context}}
    -------------------------------------------------
    
    COMANDO DO USUÁRIO / TEMA DA ANÁLISE:
    {{user_query}}
    Utilize este comando para direcionar o foco do parecer (ex: focar apenas no reequilíbrio, analisar apenas a tempestividade).
    
    COMANDO INTEGRATIVO:
    Utilize as FONTES DE APOIO apenas para validar, corrigir ou fundamentar os dados encontrados na FONTE PRIMÁRIA.
    Se a fonte primária (documento interno) estiver em desacordo com a norma externa (lei/acórdão), aponte imediatamente como RISCO ALTO na matriz.
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
            "web_context": web_context,
            "user_query": user_query
        })
    else:
        chain = prompt | llm
        response = chain.invoke({
            "pdf_text": pdf_text[:15000],
            "rag_context": rag_context,
            "web_context": web_context,
            "user_query": user_query
        })
        return response.content
