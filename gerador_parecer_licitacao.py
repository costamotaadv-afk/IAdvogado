import datetime


def _value(dados, key, default=""):
    value = dados.get(key)
    return value if value else default


def gerar_parecer(dados):
    ente = _value(dados, "ente", "")
    secretaria = _value(dados, "secretaria", "XXXXX")
    processo = _value(dados, "processo", "00000/2026")
    interessado = _value(dados, "interessado", "Interessado")
    assunto = _value(dados, "assunto", "Assunto")
    objeto = _value(dados, "objeto", "objeto da contratacao")
    procurador = _value(dados, "procurador", "Nome do Procurador")
    cargo = _value(dados, "cargo", "Cargo")
    secretaria_assinatura = _value(dados, "secretaria_assinatura", secretaria)
    cidade = _value(dados, "cidade", "Cidade")
    data = _value(dados, "data", datetime.date.today().strftime("%d/%m/%Y"))

    ente_line = f"ESTADO / MUNICIPIO {ente}" if ente else "ESTADO / MUNICIPIO"
    secretaria_line = f"SECRETARIA {secretaria}" if secretaria else "SECRETARIA"
    cabecalho = f"""
{ente_line}
{secretaria_line}
PROCURADORIA / ASSESSORIA JURÍDICA

Processo nº: {processo}
Interessado: {interessado}
Assunto: {assunto}

PARECER JURÍDICO
"""

    relatorio = f"""
RELATÓRIO

Trata-se de processo administrativo encaminhado a esta Assessoria Jurídica para análise da regularidade jurídica do procedimento referente à {objeto}.

Conforme documentação constante nos autos, a unidade demandante apresentou justificativa administrativa para a contratação, bem como documentação técnica pertinente, incluindo:

- Estudo Técnico Preliminar
- Termo de Referência
- Pesquisa de preços
- Indicação de dotação orçamentária

É o relatório.
"""

    fundamentos = f"""
FUNDAMENTAÇÃO JURÍDICA

A Constituição Federal estabelece em seu art. 37, inciso XXI, que as obras, serviços, compras e alienações da Administração Pública devem ser precedidas de licitação, ressalvados os casos previstos em lei.

Atualmente, a matéria encontra-se disciplinada pela Lei nº 14.133/2021, que instituiu o novo regime jurídico das licitações e contratos administrativos.

Nos termos do art. 11 da referida lei, o processo licitatório possui como objetivos assegurar a seleção da proposta mais vantajosa para a Administração Pública, garantir tratamento isonômico entre os licitantes e evitar contratações com sobrepreço.

Analisando os documentos constantes nos autos, verifica-se que o procedimento administrativo apresenta, em regra, os elementos formais exigidos pela legislação vigente.
"""

    conclusao = f"""
CONCLUSÃO

Diante do exposto, esta Assessoria Jurídica opina pela possibilidade jurídica do prosseguimento do procedimento administrativo referente à {objeto}, desde que observadas as recomendações eventualmente constantes neste parecer.

Encaminhem-se os autos à autoridade competente para as providências administrativas cabíveis.

{cidade}, {data}.

{procurador}
{cargo}
Assessoria Jurídica
Secretaria {secretaria_assinatura}
"""

    parecer = cabecalho + relatorio + fundamentos + conclusao

    return parecer


if __name__ == "__main__":
    dados = {
        "ente": "",
        "secretaria": "SECRETARIA X",
        "processo": "00000/2026",
        "interessado": "Secretaria X",
        "assunto": "Contratação de empresa",
        "objeto": "contratação de empresa especializada em serviços",
        "procurador": "Nome do Procurador",
        "cargo": "Cargo",
        "cidade": "Maceio",
        "data": "",
        "secretaria_assinatura": "SECRETARIA X",
    }

    parecer = gerar_parecer(dados)

    with open("parecer_gerado.txt", "w", encoding="utf-8") as f:
        f.write(parecer)

    print("Parecer gerado com sucesso.")
