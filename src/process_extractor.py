import re
from typing import Dict, Optional

FIELD_PATTERNS = {
    "objeto": [
        r"objeto\s*[:\-]\s*(.+)",
        r"objeto\s+da\s+contratacao\s*[:\-]\s*(.+)",
    ],
    "modalidade": [
        r"modalidade\s*[:\-]\s*(.+)",
        r"modalidade\s+da\s+licitacao\s*[:\-]\s*(.+)",
    ],
    "valor": [
        r"valor\s*(?:total)?\s*[:\-]\s*([^\n]+)",
        r"valor\s+estimado\s*[:\-]\s*([^\n]+)",
    ],
    "prazo": [
        r"prazo\s*(?:de)?\s*(?:execucao|vigencia)?\s*[:\-]\s*([^\n]+)",
        r"prazo\s+de\s+vigencia\s*[:\-]\s*([^\n]+)",
    ],
    "fundamento_legal": [
        r"fundamento\s+legal\s*[:\-]\s*(.+)",
        r"fundamentacao\s+legal\s*[:\-]\s*(.+)",
    ],
}


def _find_first_match(text: str, patterns) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_process_fields(text: str) -> Dict[str, Optional[str]]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    results: Dict[str, Optional[str]] = {}
    for field, patterns in FIELD_PATTERNS.items():
        results[field] = _find_first_match(normalized, patterns)
    return results
