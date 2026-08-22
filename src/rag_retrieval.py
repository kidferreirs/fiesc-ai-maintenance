from pathlib import Path
import json

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

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 5
RRF_K = 60

# Consulta técnica controlada por família
FAMILY_QUERIES = {
    "rolamento_inner": (
        "rolamento pista interna BPFI diagnóstico "
        "vibração sintomas causas inspeção correção"
    ),

    "rolamento_outer": (
        "rolamento pista externa BPFO diagnóstico "
        "vibração sintomas causas inspeção correção"
    ),

    "rolamento_ball": (
        "rolamento elementos rolantes esfera BSF "
        "diagnóstico vibração sintomas correção"
    ),

    "rolamento_combination": (
        "falhas em rolamentos BPFO BPFI BSF FTF "
        "diagnóstico vibração inspeção correção"
    ),

    "desalinhado": (
        "desalinhamento diagnóstico vibração "
        "inspeção alinhamento correção validação"
    ),

    "desbalanceado": (
        "desbalanceamento diagnóstico vibração "
        "1x RPM balanceamento correção validação"
    ),

    "correia": (
        "correia tensão escorregamento oscilação "
        "desgaste diagnóstico inspeção correção"
    ),

    "polia": (
        "polia excentricidade desgaste vibração "
        "diagnóstico inspeção correção"
    ),

    "cocked_rotor": (
        "cocked rotor rotor inclinado vibração axial "
        "1x RPM 2x RPM diagnóstico correção"
    ),
}

def load_chunks():
    # Carrega os chunks documentais
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def get_family_chunks(chunks, family):
    # Guardrail: somente documentação autorizada
    return [
        chunk
        for chunk in chunks
        if family in chunk["families"]
    ]

def build_query(family, user_query=None):
    # Usa termos controlados e, se existir, contexto adicional
    base_query = FAMILY_QUERIES.get(family)

    if not base_query:
        return None

    if user_query:
        return f"{base_query} {user_query}"

    return base_query

def reciprocal_rank_fusion(
    lexical_order,
    semantic_order,
):
    # Combina posições dos dois rankings
    scores = {}

    for rank, index in enumerate(
        lexical_order,
        start=1,
    ):
        scores[index] = scores.get(index, 0) + (
            1 / (RRF_K + rank)
        )

    for rank, index in enumerate(
        semantic_order,
        start=1,
    ):
        scores[index] = scores.get(index, 0) + (
            1 / (RRF_K + rank)
        )

    return scores

def retrieve(
    family,
    chunks,
    model,
    user_query=None,
    top_k=TOP_K,
):
    # Filtra antes de calcular qualquer ranking
    family_chunks = get_family_chunks(
        chunks,
        family,
    )

    if not family_chunks:
        return []

    query = build_query(
        family,
        user_query,
    )

    if not query:
        return []

    texts = [
        chunk["text"]
        for chunk in family_chunks
    ]

    # Ranking lexical
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    lexical_scores = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    lexical_order = lexical_scores.argsort()[::-1]

    # Ranking semântico
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    semantic_order = semantic_scores.argsort()[::-1]

    # Reciprocal Rank Fusion
    rrf_scores = reciprocal_rank_fusion(
        lexical_order,
        semantic_order,
    )

    final_order = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    results = []

    for index in final_order[:top_k]:
        chunk = family_chunks[index]

        results.append({
            **chunk,
            "rrf_score": rrf_scores[index],
            "lexical_score": float(
                lexical_scores[index]
            ),
            "semantic_score": float(
                semantic_scores[index]
            ),
        })

    return results

def print_results(family, results):
    print("\n" + "=" * 90)
    print(f"FAMÍLIA: {family}")
    print("=" * 90)

    if not results:
        print(
            "Nenhuma documentação autorizada encontrada."
        )
        return

    for position, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 90)
        print(f"Posição    : {position}")
        print(
            f"RRF        : "
            f"{result['rrf_score']:.6f}"
        )
        print(
            f"Lexical    : "
            f"{result['lexical_score']:.4f}"
        )
        print(
            f"Semântico  : "
            f"{result['semantic_score']:.4f}"
        )
        print(
            f"Documento  : "
            f"{result['document']}"
        )
        print(
            f"Página     : "
            f"{result['page']}"
        )
        print(
            f"Chunk      : "
            f"{result['chunk_id']}"
        )
        print("\nTrecho:")
        print(result["text"][:500])

def main():
    print("=" * 90)
    print("FIESC - RETRIEVAL FINAL DO RAG")
    print("=" * 90)

    chunks = load_chunks()

    model = SentenceTransformer(
        MODEL_NAME
    )

    tests = [
        (
            "rolamento_inner",
            "Como identificar dano no anel interno?",
        ),
        (
            "correia",
            "A correia está escorregando e oscilando.",
        ),
        (
            "falta_fase",
            "Como corrigir?",
        ),
    ]

    for family, query in tests:
        results = retrieve(
            family=family,
            chunks=chunks,
            model=model,
            user_query=query,
        )

        print_results(
            family,
            results,
        )

    print("\n" + "=" * 90)
    print("TESTE CONCLUÍDO")
    print("=" * 90)

if __name__ == "__main__":
    main()