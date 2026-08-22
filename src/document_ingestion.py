from pathlib import Path
import json
import re

from pypdf import PdfReader

from document_registry import DOCUMENT_REGISTRY


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "documents"


def extract_text_pdf(pdf_path):
    # Extrai a camada textual página a página
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text.strip(),
        })

    return pages


def has_useful_text(pages, min_chars=100):
    # Confirma se há conteúdo textual suficiente
    total_chars = sum(len(page["text"]) for page in pages)
    return total_chars >= min_chars


def extract_sidecar_text(pdf_path):
    # Para PDFs escaneados, usa um TXT previamente validado
    txt_path = pdf_path.with_suffix(".txt")

    if not txt_path.exists():
        return None

    content = txt_path.read_text(encoding="utf-8").strip()

    if not content:
        return None

    # Preserva as páginas marcadas no arquivo TXT
    pattern = r"=== PÁGINA (\d+) ===\n"
    parts = re.split(pattern, content)

    pages = []

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        text = parts[i + 1].strip()

        pages.append({
            "page": page_number,
            "text": text,
        })

    return pages or [{
        "page": None,
        "text": content,
    }]


def build_document_record(family, pdf_path):
    # Primeiro tenta a camada textual original do PDF
    pages = extract_text_pdf(pdf_path)
    method = "pypdf"

    # PDF escaneado: usa transcrição validada em arquivo sidecar
    if not has_useful_text(pages):
        sidecar_pages = extract_sidecar_text(pdf_path)

        if sidecar_pages:
            pages = sidecar_pages
            method = "validated_sidecar"
        else:
            method = "no_text"

    return {
        "family": family,
        "document": pdf_path.name,
        "extraction_method": method,
        "pages": pages,
    }


def main():
    print("=" * 90)
    print("FIESC - INGESTÃO DOS DOCUMENTOS TÉCNICOS")
    print("=" * 90)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_documents = {}
    unique_documents = {}

    # Evita processar o mesmo PDF várias vezes
    for family, pdf_path in DOCUMENT_REGISTRY.items():
        unique_documents.setdefault(
            pdf_path.name,
            {
                "path": pdf_path,
                "families": [],
            },
        )

        unique_documents[pdf_path.name]["families"].append(family)

    for document_name, config in unique_documents.items():
        pdf_path = config["path"]
        families = config["families"]

        record = build_document_record(
            families[0],
            pdf_path,
        )

        record["families"] = families
        processed_documents[document_name] = record

        text_chars = sum(
            len(page["text"])
            for page in record["pages"]
        )

        print(f"\nDocumento : {document_name}")
        print("Famílias  : " + ", ".join(families))
        print(f"Método    : {record['extraction_method']}")
        print(f"Páginas   : {len(record['pages'])}")
        print(f"Caracteres: {text_chars:,}")

    output_path = OUTPUT_DIR / "documents_raw.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            processed_documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nArquivo gerado:")
    print(output_path)

    print("\n" + "=" * 90)
    print("INGESTÃO CONCLUÍDA")
    print("=" * 90)


if __name__ == "__main__":
    main()
