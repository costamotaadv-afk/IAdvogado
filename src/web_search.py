import logging
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def search_jurisprudence(query: str, max_results: int = 5) -> str:
    """Search for jurisprudence online using DuckDuckGo and return a formatted context string."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return "Não foi possível realizar a busca na web no momento."

    if not results:
        return "Nenhuma jurisprudência encontrada para o tema pesquisado."

    lines = []
    for item in results:
        title = item.get("title", "Sem título")
        body = item.get("body", "")
        href = item.get("href", "")
        lines.append(f"**{title}**\n{body}\nFonte: {href}")

    return "\n\n---\n\n".join(lines)
