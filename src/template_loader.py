from pathlib import Path
from typing import Dict

TEMPLATE_FILES: Dict[str, str] = {
    "dispensa": "dispensa.txt",
    "inexigibilidade": "inexigibilidade.txt",
    "reequilibrio": "reequilibrio.txt",
    "pregao": "pregao.txt",
    "srp": "srp.txt",
}


def load_template(case_type: str) -> str:
    filename = TEMPLATE_FILES.get(case_type)
    if not filename:
        return ""

    base_dir = Path(__file__).resolve().parent
    template_path = base_dir / "templates" / filename
    if not template_path.exists():
        return ""

    return template_path.read_text(encoding="utf-8")
