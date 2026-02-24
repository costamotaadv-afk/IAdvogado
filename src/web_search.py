from langchain_community.tools import DuckDuckGoSearchRun
import time

def search_jurisprudence(topic: str) -> str:
    """
    Realiza buscas segmentadas na web para simular uma pesquisa jurídica completa (Google-like).
    Busca separadamente por Legislação, TCU, Tribunais Superiores e Doutrina.
    """
    search_tool = DuckDuckGoSearchRun()
    combined_results = []
    
    # 1. Camada de Notícias e Contexto (Clima dos Tribunais)
    news_strategy = {
        "source": "📰 NOTÍCIAS JURÍDICAS E CONTEXTO (Migalhas, Conjur, Jota)",
        "query": f"site:migalhas.com.br OR site:conjur.com.br OR site:jota.info {topic} (\"decisão recente\" OR \"últimas notícias\")"
    }

    # 2. Camada de Fontes Oficiais (Lei Seca e Andamento)
    official_strategy = {
        "source": "⚖️ LEGISLAÇÃO E FONTES OFICIAIS (Planalto, Senado, Jus.br)",
        "query": f"site:planalto.gov.br OR site:senado.leg.br OR site:jus.br {topic} lei 14.133"
    }

    # 3. Camada Técnica e Doutrinária (Acórdãos e Teses)
    technical_strategy = {
        "source": "🎓 DOUTRINA E JURISPRUDÊNCIA TÉCNICA (Acórdãos e Teses)",
        "query": f"site:stj.jus.br OR site:stf.jus.br OR site:tcu.gov.br {topic} (\"acórdão\" OR \"tese fixada\" OR \"voto relator\")"
    }
    
    search_strategies = [news_strategy, official_strategy, technical_strategy]
    
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
