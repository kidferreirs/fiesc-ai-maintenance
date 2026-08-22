from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "documents"
    / "document_chunks.json"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

def load_chunks():
    # Carrega a base documental preparada
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def build_index(chunks):
    # Modelo multilíngue leve, adequado ao português
    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return model, embeddings

def retrieve(
    query,
    family,
    chunks,
    model,
    embeddings,
    top_k=TOP_K,
):
    # Gera embedding semântico da consulta
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    results = []

    for index, score in enumerate(scores):
        chunk = chunks[index]

        # Guardrail: pesquisa apenas documentos da família
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
    print("CONSULTA SEMÂNTICA")
    print("=" * 90)
    print(f"Pergunta : {query}")
    print(f"Família  : {family}")

    if not results:
        print("\nNenhuma documentação autorizada encontrada.")
        return

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
        print("\nTrecho:")
        print(result["text"][:500])

def main():
    print("=" * 90)
    print("FIESC - BUSCA SEMÂNTICA DOCUMENTAL")
    print("=" * 90)

    chunks = load_chunks()

    print(f"\nChunks carregados: {len(chunks)}")
    print(f"Modelo           : {MODEL_NAME}")

    model, embeddings = build_index(chunks)

    # Consulta usando palavras diferentes das do documento
    query = (
        "Como identificar dano no anel interno "
        "do rolamento através da vibração?"
    )

    results = retrieve(
        query=query,
        family="rolamento_inner",
        chunks=chunks,
        model=model,
        embeddings=embeddings,
    )

    print_results(
        query,
        "rolamento_inner",
        results,
    )

    # Consulta sobre correia
    query = (
        "A transmissão está escorregando e "
        "a correia oscila durante a operação. "
        "O que devo verificar?"
    )

    results = retrieve(
        query=query,
        family="correia",
        chunks=chunks,
        model=model,
        embeddings=embeddings,
    )

    print_results(
        query,
        "correia",
        results,
    )

    # Família sem documento
    query = "Como corrigir falta de fase?"

    results = retrieve(
        query=query,
        family="falta_fase",
        chunks=chunks,
        model=model,
        embeddings=embeddings,
    )

    print_results(
        query,
        "falta_fase",
        results,
    )

if __name__ == "__main__":
    main()