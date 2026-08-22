from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "documents"
    / "documents_raw.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "documents"
    / "document_chunks.json"
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # Divide texto preservando pequena sobreposição
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        # Evita cortar no meio de uma palavra
        if end < len(text):
            last_space = chunk.rfind(" ")

            if last_space > chunk_size * 0.6:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks

def main():
    print("=" * 90)
    print("FIESC - CHUNKING DOS DOCUMENTOS")
    print("=" * 90)

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        documents = json.load(file)

    chunks = []

    for document_name, document in documents.items():
        families = document["families"]

        document_count = 0

        for page in document["pages"]:
            page_number = page["page"]
            text = page["text"]

            page_chunks = split_text(text)

            for position, chunk_text in enumerate(
                page_chunks,
                start=1,
            ):
                chunk_id = (
                    f"{document_name}"
                    f"_p{page_number}"
                    f"_c{position}"
                )

                chunks.append({
                    "chunk_id": chunk_id,
                    "document": document_name,
                    "page": page_number,
                    "families": families,
                    "text": chunk_text,
                })

                document_count += 1

        print(
            f"{document_name:<10} "
            f"-> {document_count:>3} chunks"
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTotal de chunks:")
    print(f"{len(chunks):,}")
    print("\nArquivo gerado:")
    print(OUTPUT_PATH)
    print("\n" + "=" * 90)
    print("CHUNKING CONCLUÍDO")
    print("=" * 90)

if __name__ == "__main__":
    main()