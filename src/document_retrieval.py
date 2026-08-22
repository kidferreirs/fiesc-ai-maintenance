from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "documents"
    / "document_chunks.json"
)

TOP_K = 5

def load_chunks():
    # Carrega os chunks já preparados
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def build_index(chunks):
    # Vetorização textual simples e local
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix

def retrieve(
    query,
    chunks,
    vectorizer,
    matrix,
    family=None,
    top_k=TOP_K,
):
    # Vetoriza a consulta
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    results = []

    for index, score in enumerate(scores):
        chunk = chunks[index]

        # Restringe aos documentos autorizados da família
        if family is not None:
            if family not in chunk["families"]:
                continue

        results.append({
            **chunk,
            "score": float(score),
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


def print_results(query, family, results):
    print("\n" + "=" * 90)
    print("CONSULTA")
    print("=" * 90)
    print(f"Pergunta : {query}")
    print(f"Família  : {family}")
    print("\nRESULTADOS")

    for position, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 90)
        print(f"Posição   : {position}")
        print(f"Score     : {result['score']:.4f}")
        print(f"Documento : {result['document']}")
        print(f"Página    : {result['page']}")
        print(f"Chunk     : {result['chunk_id']}")

        preview = result["text"][:500]

        print("\nTrecho:")
        print(preview)

def main():
    print("=" * 90)
    print("FIESC - TESTE DE RECUPERAÇÃO DOCUMENTAL")
    print("=" * 90)

    chunks = load_chunks()

    print(f"\nChunks carregados: {len(chunks)}")

    vectorizer, matrix = build_index(chunks)

    # Teste 1 - Rolamento pista interna
    query = (
        "Quais são os sintomas e como diagnosticar "
        "defeito na pista interna do rolamento?"
    )

    results = retrieve(
        query=query,
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
        family="rolamento_inner",
    )

    print_results(
        query,
        "rolamento_inner",
        results,
    )

    # Teste 2 - Correia
    query = (
        "O que deve ser verificado quando existe "
        "problema de tensão ou desgaste da correia?"
    )

    results = retrieve(
        query=query,
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
        family="correia",
    )

    print_results(
        query,
        "correia",
        results,
    )

    # Teste 3 - Família sem documentação
    query = "Como corrigir uma falha de falta de fase?"

    results = retrieve(
        query=query,
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
        family="falta_fase",
    )

    print_results(
        query,
        "falta_fase",
        results,
    )

    print("\n" + "=" * 90)
    print("TESTE CONCLUÍDO")
    print("=" * 90)

if __name__ == "__main__":
    main()