from duckduckgo_search import DDGS
import time

def search_jurisprudence(topic: str) -> str:
    """
    Realiza buscas segmentadas na web para simular uma pesquisa jurídica completa (Google-like).
    Busca separadamente por Legislação, TCU, Tribunais Superiores e Doutrina.
    Retorna explicitamente URLs e Títulos para metadados.
    """
    combined_results = []
    
    # Estratégias de Busca Segmentada
    strategies = [
        {
            "name": "📜 LEGISLAÇÃO E PORTAIS OFICIAIS",
            "query": f"{topic} site:planalto.gov.br OR site:gov.br (\"lei 14.133\" OR \"instrução normativa\")"
        },
        {
            "name": "⚖️ JURISPRUDÊNCIA E CONTROLE (TCU/STJ)",
            "query": f"{topic} site:tcu.gov.br OR site:stj.jus.br OR site:stf.jus.br (\"acórdão\" OR \"enunciado\")"
        },
        {
            "name": "🎓 DOUTRINA INSTITUCIONAL (AGU/ENAP)",
            "query": f"{topic} site:advocaciageral.gov.br OR site:enap.gov.br (\"parecer\" OR \"nota técnica\")"
        }
    ]
    
    combined_results.append(f"🔎 RELATÓRIO DE PESQUISA AVANÇADA PARA: '{topic}'\n")

    with DDGS() as ddgs:
        for strategy in strategies:
            try:
                # Delay anti-bloqueio
                time.sleep(1)
                
                results = list(ddgs.text(strategy["query"], region='br-pt', safesearch='off', max_results=3))
                
                header = f"\n--- {strategy['name']} ---"
                combined_results.append(header)
                
                if not results:
                    combined_results.append("Nenhum resultado relevante encontrado.")
                    continue
                    
                for i, r in enumerate(results, 1):
                    # Formata cada resultado com Título, URL e Texto (snippet)
                    title = r.get('title', 'Sem título')
                    link = r.get('href', 'URL indisponível')
                    body = r.get('body', '')
                    
                    entry = (
                        f"\n[RESULTADO {i}]"
                        f"\nTITULO: {title}"
                        f"\nFONTE/URL: {link}"
                        f"\nRESUMO: {body}"
                        f"\n-------------------------------------------------"
                    )
                    combined_results.append(entry)
                    
            except Exception as e:
                combined_results.append(f"\nErro na busca de {strategy['name']}: {str(e)}")

    return "\n".join(combined_results)
