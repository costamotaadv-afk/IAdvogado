import os
from openai import OpenAI


def generate_legal_opinion(pdf_text: str, rag_context: str, web_context: str) -> str:
    """Generate a formal legal opinion using OpenAI based on document text and retrieved context."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = (
        "Você é um advogado publicista especializado em licitações e contratos administrativos, "
        "com profundo conhecimento da Lei Federal nº 14.133/2021 (Nova Lei de Licitações), "
        "jurisprudência do TCU, STJ, STF e Tribunais de Justiça estaduais. "
        "Redija pareceres jurídicos formais, fundamentados e completos, em língua portuguesa."
    )

    user_prompt = f"""Com base nas informações abaixo, elabore um Parecer Jurídico formal e fundamentado.

## Documento Analisado
{pdf_text[:4000]}

## Contexto da Base de Conhecimento (RAG)
{rag_context[:3000]}

## Jurisprudência e Doutrina Encontradas na Web
{web_context[:3000]}

## Instruções para o Parecer
- Utilize estrutura formal: Ementa, Relatório, Fundamentação e Conclusão.
- Cite os dispositivos legais da Lei 14.133/2021 aplicáveis.
- Referencie jurisprudência do TCU, STJ ou STF quando pertinente.
- Conclua com uma recomendação jurídica clara e objetiva.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    return response.choices[0].message.content
