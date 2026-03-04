import os
from pathlib import Path
from docx import Document

TEMPLATE_DIRS = [
    Path(r"c:\Users\Keynes\Meu Aplicativo\src\templates"),
    Path(r"c:\Users\Keynes\Meu Aplicativo\biblioteca_pareceres_14133"),
]

def to_rtf(text: str) -> str:
    # Minimal RTF wrapper to make a Word-readable .doc file.
    esc = (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    esc = esc.replace("\n", r"\par\n")
    return r"{\rtf1\ansi\deff0 " + esc + r"}"

def write_docx(text: str, out_path: Path) -> None:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(out_path)

def write_doc(text: str, out_path: Path) -> None:
    rtf = to_rtf(text)
    out_path.write_text(rtf, encoding="utf-8")

def process_txt(txt_path: Path) -> None:
    text = txt_path.read_text(encoding="utf-8")
    docx_path = txt_path.with_suffix(".docx")
    doc_path = txt_path.with_suffix(".doc")

    write_docx(text, docx_path)
    write_doc(text, doc_path)


def main() -> None:
    for root in TEMPLATE_DIRS:
        if not root.exists():
            continue
        for txt_path in root.glob("*.txt"):
            process_txt(txt_path)

if __name__ == "__main__":
    main()
