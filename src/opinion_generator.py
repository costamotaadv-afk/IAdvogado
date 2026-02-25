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

    # Configuração de Regeneração (Correção Automática)
    regen_config = config.get("regen", {})
    regen_instruction = ""
    if regen_config.get("enabled"):
        updates = ", ".join(regen_config.get("post_delivery_updates", []))
        regen_instruction = f"""
    6. REGENERAÇÃO E AUTO-CORREÇÃO (Loop de Aprendizado):
       - ATENÇÃO: Se esta for uma solicitação de correção (feedback), trate como uma NOVA VERSÃO (v+1).
       - Aplique as seguintes ações de melhoria: {updates}.
       - Ao final, liste explicitamente o que foi corrigido em relação à versão anterior.
        """

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
       
       REGRA ESPECIAL PARA AUSÊNCIA DE DOCUMENTOS:
       Se a FONTE PRIMÁRIA (Documentos Internos) estiver vazia ou não contiver documentos essenciais (edital, contrato, etc.), os tópicos 1, 2, 3, 4, 5, 6, 7 e 8 DEVEM ser EXATAMENTE os seguintes textos:
       
       "1. Sumário Executivo
       Foi solicitada análise de documentação anexada e da biblioteca do projeto, com eventual complementação por pesquisa externa. Após varredura, não foram localizados documentos essenciais (ex.: edital, Termo de Referência, ETP, pesquisa de preços, minuta/contrato, atas, parecer jurídico, publicações, planilhas, mapa de rotas, etc.), o que impede conclusão técnica sobre regularidade, proporcionalidade das exigências, estimativa de preços e viabilidade do objeto.
       
       Conclusão: a análise não é possível em profundidade por lacuna documental. Recomenda-se regularização imediata do acervo e reprocessamento do parecer após juntada dos documentos mínimos, com rastreabilidade por arquivo/página;"
       
       "2. Escopo, Método e Linha do Tempo (auditável)
       
       2.1 Escopo
       Licitações/contratos/convênios: checagem de conformidade documental mínima, riscos de impugnação, inexequibilidade, nulidade, e risco de controle externo.
       
       2.2 Método aplicado e Log de Busca Interna
       - Palavras-chave utilizadas: [LISTAR PALAVRAS-CHAVE USADAS NA BUSCA, ex: "edital", "TR", "contrato"]
       - Locais de busca: Anexos fornecidos (Fonte Primária) e Biblioteca do projeto (RAG)
       - Arquivos encontrados: 0 (zero) documentos essenciais localizados.
       Busca externa (internet): consulta a referências normativas gerais e orientações públicas (sem substituir documentos do processo).
       
       2.3 Linha do tempo
       Data da solicitação: [DADO]
       Prazo para entrega/decisão: [DADO]
       Constatação central: ausência de acervo mínimo para instrução e análise."
       
       "3. Diagnóstico por Quality Gates (com ressalvas)
       
       Motivação: não verificável (sem ETP/TR/justificativa juntada).
       Proporcionalidade: não verificável (sem edital/TR e sem exigências transcritas).
       Documentação: insuficiente (lacuna de peças essenciais).
       Rastreabilidade: inexistente no parecer (não há referência a arquivos/páginas; não há índice de anexos).
       Coerência: não verificável (sem descrição técnica do objeto e critério de julgamento)."
       
       "4. Fundamentação Normativa (base mínima, sem presumir regime do certame)
       
       Sem acesso aos documentos, não é possível afirmar se o procedimento tramita sob Lei nº 14.133/2021 ou sob regime anterior/transitório. Ainda assim, há regra geral: processos licitatórios/contratuais exigem instrução formal e motivada, com documentação mínima que suporte necessidade, preço estimado, regras do certame e gestão do contrato.
       
       - Lei nº 14.133/2021 (Normas gerais de licitações e contratos).
       - Lei nº 8.666/1993 (regime anterior, aplicável conforme transição e contexto do ente).
       
       Ajuste técnico importante: há risco de invalidação/anulação por deficiência de instrução."
       
       "5. Teses aplicáveis (por cenário)
       
       Cenário A — Lacuna de documentação essencial (Lacuna Real)
       Tese técnica: Sem instrução mínima (ETP/TR, edital/minuta, estimativa de preços e anexos do objeto), o processo fica vulnerável a questionamentos de legalidade, nulidade, impugnações e apontamentos de controle externo, porque não se consegue demonstrar:
       - necessidade e adequação do objeto,
       - critério de julgamento e regras do certame,
       - estimativa e aceitabilidade de preços,
       - fiscalização/gestão contratual.
       
       Cenário B — Falha do sistema de anexos/biblioteca (Lacuna de acesso/indexação)
       Tese técnica: Pode haver documentação existente, mas inacessível ao agente por erro de indexação/nomeação/permissão/pasta. Nesse caso, o problema é de governança informacional, e a ação é saneamento do repositório."
       
       "6. Análise de Risco e Evidências
       
       Risco Alto (se a lacuna for real):
       - Controle externo: glosa, determinações, recomendações, responsabilização por falha de instrução.
       - Impugnação e judicialização: ausência de TR/edital/planilhas aumenta probabilidade de suspensão.
       - Inexequibilidade/sobrepreço: sem pesquisa/memória de cálculo, risco de contratar mal.
       - Execução e fiscalização: sem definição técnica, risco de prestação inadequada e litígios.
       
       Risco Moderado (se a lacuna for de acesso/indexação):
       - Atraso e retrabalho; decisões baseadas em informação incompleta."
       
       "7. Pedidos / Recomendações (objetivas e executáveis)
       
       7.1 Regularização do acervo (Lista mínima de documentos por tipo de caso)
       Solicitar/Anexar, no mínimo, conforme o caso:
       
       Para Editais/Licitações:
       - Termo de Referência e/ou ETP (ou justificativa formal equivalente).
       - Edital (ou minuta do instrumento convocatório) e anexos.
       - Pesquisa de preços + memória de cálculo (metodologia e fontes).
       - Parecer jurídico (ou manifestação jurídica), quando aplicável.
       
       Para Contratos/Aditivos:
       - Minuta contratual/Contrato assinado.
       - Termos Aditivos (se existirem) e suas justificativas.
       - Documentos de habilitação e regularidade fiscal da contratada.
       - Publicações (aviso, extrato) e atas.
       
       Para Convênios:
       - Plano de Trabalho aprovado.
       - Termo de Convênio assinado.
       - Comprovação de regularidade do convenente.
       - Pareceres técnicos e jurídicos de aprovação.
       
       7.2 Governança do repositório (para o agente achar)
       - Padronizar nomes: A-001_TR.pdf, A-002_Edital.pdf, A-003_ETP.pdf, A-004_PesquisaPrecos.xlsx, etc.
       - Gerar um index.yaml/index.csv com arquivo + assunto + data + páginas-chave.
       
       7.3 Reprocessamento
       - Após juntada, reexecutar o parecer para produzir: Matriz de evidências, índice de anexos e checklist do edital/TR com pontos de risco."
       
       "8. Apêndice — Matriz de Evidências e Índice de Anexos (negativos, para auditoria)
       
       8.1 Matriz de evidências (NEGATIVA — “deveria existir”)
       | Item a verificar      | Evidência esperada            | Status     |
       | --------------------- | ----------------------------- | ---------- |
       | Necessidade/motivação | ETP/TR/justificativa          | **LACUNA** |
       | Definição do objeto   | TR + anexos técnicos          | **LACUNA** |
       | Estimativa de preços  | pesquisa + memória de cálculo | **LACUNA** |
       | Regras do certame     | edital + anexos               | **LACUNA** |
       | Gestão/fiscalização   | cláusulas e rotinas           | **LACUNA** |
       | Publicidade/atos      | publicações/atas              | **LACUNA** |
       | Parecer jurídico      | peça de controle              | **LACUNA** |
       
       8.2 Índice de Anexos (Vazio)
       - Nenhum anexo essencial foi identificado ou processado com sucesso para esta análise."
       
    5. APRENDIZADO CONTÍNUO:
       Use NEGRITO para: {formatting}.

    {regen_instruction}
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
