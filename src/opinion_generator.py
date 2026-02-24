import os
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List

def load_config():
    """Carrega as configurações do agente a partir do arquivo YAML."""
    try:
        # Caminho relativo considerando que o script está em src/
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
    except Exception as e:
        print(f"Erro ao carregar config: {e}")
        return {}

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
    Carrega dinamicamente as regras de negócio do arquivo config.yaml.
    """
    
    # 1. Carregar Configurações
    config = load_config()
    # Se falhar o load, usa defaults vazios para não quebrar
    agent_config = config.get("agent", {}) if config else {}
    
    name = agent_config.get("name", "Agente Jurídico")
    domains = ", ".join(agent_config.get("domain", ["Direito Administrativo"]))
    
    secrecy = agent_config.get("secrecy", {})
    placeholders = ", ".join(secrecy.get("placeholders", ["[NOME]", "[DADO]"]))
    redact_fields = ", ".join(secrecy.get("redact_fields", []))
    
    output_cfg = agent_config.get("output", {})
    structure = "\n       ".join(output_cfg.get("default_structure", [
        "- Relatório", "- Fundamentação", "- Conclusão"
    ]))
    formatting = ", ".join(output_cfg.get("formatting", {}).get("bold_for", ["Prazos"]))

    quality_gates = agent_config.get("quality_gates", {}).get("must_pass", [])
    gates_str = ""
    for g in quality_gates:
        gates_str += f"- {g.get('name')}: {g.get('rule')}\n       "
        
    triggers = ", ".join(agent_config.get("escalation", {}).get("triggers", []))

    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    system_prompt = f"""
    IDENTITY:
    Você é o "{name}", especialista em {domains}.
    Sua missão é atuar como uma barreira de controle de legalidade (Compliance Check).

    DIRETRIZES DE SEGURANÇA E SIGILO:
    SIGILO OBRIGATÓRIO: Substitua dados pessoais por {placeholders}. 
    Campos sensíveis: {redact_fields}.
    
    PIPELINE DE EXECUÇÃO OBRIGATÓRIA (Workflow):
    
    1. CLASSIFICAÇÃO IMEDIATA:
       Identifique se o caso é: Edital, Contrato, Convênio ou Contencioso.
       
    2. DIAGNÓSTICO E QUALITY GATES (Checklist de Bloqueio):
       Antes de opinar, verifique se o caso cumpre os requisitos mínimos:
       {gates_str}
       SE ALGUM REQUISITO FALHAR GRAVEMENTE (Ex: {triggers}), ESCALONE PARA VALIDAÇÃO HUMANA.

    3. PROVAS E TESES (Evidence-Based Law):
       - Cruzamento obrigatório: Fato (Página X) -> Norma -> Prova.
       - Se não houver prova documental, declare "INEXISTÊNCIA DE PROVA".

    4. REDAÇÃO FINAL (Estrutura Obrigatória):
       Gere o parecer seguindo rigorosamente esta estrutura:
       {structure}
       
    5. REGENERAÇÃO E APRENDIZADO:
       Se o usuário fornecer novos inputs, incorpore-os como correções imediatas.
       Use NEGRITO para: {formatting}.
    """
    
    user_prompt = f"""
    FONTE PRIMÁRIA - DOCUMENTOS INTERNOS DO PROCESSO (Anexos):
    (Respeite a prioridade documental: Edital > Contrato > Pareceres > E-mails)
    -------------------------------------------------
    {{pdf_text}}
    -------------------------------------------------
    
    FONTES DE APOIO - BIBLIOTECA E INTERNET (Externas):
    -------------------------------------------------
    CONTEXTO DA BIBLIOTECA:
    {{rag_context}}
    
    PESQUISA WEB ATUALIZADA (Validar URL e Data):
    {{web_context}}
    -------------------------------------------------
    
    COMANDO DO USUÁRIO / TEMA DA ANÁLISE:
    {{user_query}}
    Utilize este comando para direcionar o foco do parecer (ex: focar apenas no reequilíbrio, analisar apenas a tempestividade).
    
    COMANDO INTEGRATIVO:
    Utilize as FONTES DE APOIO apenas para validar, corrigir ou fundamentar os dados encontrados na FONTE PRIMÁRIA.
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
