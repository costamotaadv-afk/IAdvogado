from langchain_community.tools import DuckDuckGoSearchRun
from typing import List

def search_jurisprudence(query: str, max_results: int = 3) -> str:
    """
    Busca jurisprudência recente na web (TCU, STJ, STF, TJs).
    
    Args:
        query (str): A consulta de busca (ex: "jurisprudência TCU lei 14.133 dispensa de licitação").
        max_results (int): Número máximo de resultados.
        
    Returns:
        str: Resultados da busca formatados como texto.
    """
    try:
        search = DuckDuckGoSearchRun()
        # DuckDuckGoSearchRun retorna uma string com os resultados
        results = search.run(query)
        return results
    except Exception as e:
        return f"Erro ao buscar jurisprudência na web: {str(e)}"
