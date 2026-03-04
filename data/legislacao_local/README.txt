BASE LOCAL DE LEGISLACAO E JURISPRUDENCIA

Esta pasta guarda a base fixa para consulta local.

Estrutura sugerida (base.json):
- lei_14133: itens da Lei 14.133/2021
- decretos: decretos relacionados
- sumulas_tcu: sumulas do TCU
- acordaos_tcu: acordaos do TCU
- jurisprudencia: STJ, STF, TRFs e TJs
- ingestao_pendente: itens aguardando confirmacao

Formato sugerido para itens:
{
  "id": "string",
  "tipo": "lei|decreto|sumula|acordao|jurisprudencia",
  "orgao": "TCU|STJ|STF|TRF|TJ|OUTRO",
  "numero": "string",
  "data": "YYYY-MM-DD",
  "ementa": "texto",
  "fonte_url": "https://...",
  "status": "pending_review|approved",
  "added_by": "user|web"
}

Recomendacao:
- Itens coletados na web devem entrar como pending_review.
- Somente confirmar apos validar ementa, data e fonte oficial.
