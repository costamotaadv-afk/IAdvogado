def classify_legal_case(query: str) -> str:
    """
    Classifica automaticamente o tipo de procedimento jurídico
    com base na consulta do usuário.
    """

    query = query.lower()

    if "dispensa" in query:
        return "dispensa"

    if "inexigibilidade" in query:
        return "inexigibilidade"

    if "reequilíbrio" in query or "reequilibrio" in query:
        return "reequilibrio"

    if "aditivo" in query:
        return "aditivo"

    if "registro de preços" in query or "registro de precos" in query or "srp" in query:
        return "srp"

    if "pregão" in query or "pregao" in query:
        return "pregao"

    if "emergencial" in query:
        return "emergencial"

    return "licitacao"
