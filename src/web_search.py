from langchain_community.tools import DuckDuckGoSearchRun
import time

def search_jurisprudence(topic: str) -> str:
    """
    Realiza buscas segmentadas na web para simular uma pesquisa jurídica completa (Google-like).
    Busca separadamente por Legislação, TCU, Tribunais Superiores e Doutrina.
    """
    search_tool = DuckDuckGoSearchRun()
    combined_results = []
    
    # 1. Camada de Legislação e Normas Oficiais (Base Legal)
    official_normative_strategy = {
        "source": "📜 LEGISLAÇÃO, MANUAIS E PORTAIS OFICIAIS (Planalto, Transparência)",
        "query": f"site:planalto.gov.br OR site:gov.br OR site:portaltransparencia.gov.br {topic} (\"lei 14.133\" OR \"instrução normativa\" OR \"manual\")"
    }

    # 2. Camada de Jurisprudência e Controle Externo (Precedentes)
    jurisprudence_strategy = {
        "source": "⚖️ JURISPRUDÊNCIA E CONTROLE (TCU, STJ, STF)",
        "query": f"site:tcu.gov.br OR site:stj.jus.br OR site:stf.jus.br {topic} (\"acórdão\" OR \"enunciado\" OR \"tese fixada\")"
    }

    # 3. Camada de Doutrina e Notas Técnicas (Instituições)
    doctrine_strategy = {
        "source": "🎓 DOUTRINA E NOTAS TÉCNICAS (Institucional)",
        "query": f"site:advocaciageral.gov.br OR site:enap.gov.br OR site:cnj.jus.br {topic} (\"parecer referencial\" OR \"nota técnica\" OR \"orientação normativa\")"
    }
    
    search_strategies = [official_normative_strategy, jurisprudence_strategy, doctrine_strategy]
    
    combined_results.append(f"🔎 RELATÓRIO DE PESQUISA NA WEB SOBRE: '{topic}'\n")

    for strategy in search_strategies:
        try:
            # Adiciona um pequeno delay para não bloquear a busca
            time.sleep(1) 
            result = search_tool.run(strategy["query"])
            
            # Formata o resultado para ficar legível no parecer
            combined_results.append(f"\n--- {strategy['source']} ---")
            combined_results.append(result if result else "Nenhum resultado relevante encontrado nesta fonte.")
            
        except Exception as e:
            combined_results.append(f"\n--- {strategy['source']} ---\nErro na busca: {str(e)}")

    return "\n".join(combined_results)
