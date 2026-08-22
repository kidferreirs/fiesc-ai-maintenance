from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "documents"
    / "document_chunks.json"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SEMANTIC_WEIGHT = 0.60
LEXICAL_WEIGHT = 0.40
TOP_K = 5

def load_chunks():
    # Carrega a base documental
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def build_indexes(chunks):
    texts = [chunk["text"] for chunk in chunks]

    # Índice lexical
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    # Índice semântico
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return (
        vectorizer,
        tfidf_matrix,
        model,
        embeddings,
    )

def normalize_scores(scores):
    # Coloca os scores em escala comparável
    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )

def retrieve(
    query,
    family,
    chunks,
    vectorizer,
    tfidf_matrix,
    model,
    embeddings,
    top_k=TOP_K,
):
    # Similaridade lexical
    query_tfidf = vectorizer.transform([query])

    lexical_scores = cosine_similarity(
        query_tfidf,
        tfidf_matrix,
    )[0]

    # Similaridade semântica
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    # Normaliza antes de combinar
    lexical_scores = normalize_scores(
        lexical_scores
    )

    semantic_scores = normalize_scores(
        semantic_scores
    )

    hybrid_scores = (
        SEMANTIC_WEIGHT * semantic_scores
        + LEXICAL_WEIGHT * lexical_scores
    )

    results = []

    for index, score in enumerate(hybrid_scores):
        chunk = chunks[index]

        # Guardrail documental
        if family not in chunk["families"]:
            continue

        results.append({
            **chunk,
            "hybrid_score": float(score),
            "semantic_score": float(
                semantic_scores[index]
            ),
            "lexical_score": float(
                lexical_scores[index]
            ),
        })

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return results[:top_k]

def print_results(query, family, results):
    print("\n" + "=" * 90)
    print("CONSULTA HÍBRIDA")
    print("=" * 90)
    print(f"Pergunta : {query}")
    print(f"Família  : {family}")

    if not results:
        print(
            "\nNenhuma documentação autorizada encontrada."
        )
        return

    for position, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 90)
        print(f"Posição        : {position}")
        print(
            f"Score híbrido  : "
            f"{result['hybrid_score']:.4f}"
        )
        print(
            f"Score semântico: "
            f"{result['semantic_score']:.4f}"
        )
        print(
            f"Score lexical  : "
            f"{result['lexical_score']:.4f}"
        )
        print(f"Documento      : {result['document']}")
        print(f"Página         : {result['page']}")
        print(f"Chunk          : {result['chunk_id']}")
        print("\nTrecho:")
        print(result["text"][:500])

def main():
    print("=" * 90)
    print("FIESC - BUSCA HÍBRIDA DOCUMENTAL")
    print("=" * 90)

    chunks = load_chunks()

    print(f"\nChunks carregados : {len(chunks)}")
    print(f"Peso semântico    : {SEMANTIC_WEIGHT:.2f}")
    print(f"Peso lexical      : {LEXICAL_WEIGHT:.2f}")
    (
        vectorizer,
        tfidf_matrix,
        model,
        embeddings,
    ) = build_indexes(chunks)

    tests = [
        (
            "Como identificar dano no anel interno "
            "do rolamento através da vibração?",
            "rolamento_inner",
        ),
        (
            "Quais são os sintomas e como diagnosticar "
            "defeito na pista interna do rolamento?",
            "rolamento_inner",
        ),
        (
            "A transmissão está escorregando e a correia "
            "oscila durante a operação. O que devo verificar?",
            "correia",
        ),
        (
            "Como corrigir uma falha de falta de fase?",
            "falta_fase",
        ),
    ]

    for query, family in tests:
        results = retrieve(
            query=query,
            family=family,
            chunks=chunks,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            model=model,
            embeddings=embeddings,
        )

        print_results(
            query,
            family,
            results,
        )
    print("\n" + "=" * 90)
    print("TESTE CONCLUÍDO")
    print("=" * 90)

if __name__ == "__main__":
    main()