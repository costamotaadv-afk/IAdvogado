import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def load_template(case_type: str) -> str:
    """
    Carrega template jurídico apropriado
    """

    file_path = os.path.join(TEMPLATE_DIR, f"{case_type}.txt")

    if not os.path.exists(file_path):
        file_path = os.path.join(TEMPLATE_DIR, "licitacao.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
