from pathlib import Path
import json

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from rag_retrieval import (
    MODEL_NAME,
    load_chunks,
)
from rag_response import generate_rag_response
from database import (
    init_db,
    save_analysis,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "banner_normalized.csv"
)

TOP_K = 5

# Gate calibrado no experimento de confiança
MIN_HISTORICAL_SUPPORT = 1.0
MAX_MEAN_DISTANCE = 1.0

FEATURES = [
    "z_rms_velocity_mm_s",
    "x_rms_velocity_mm_s",
    "temperature_c",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_kurtosis",
    "x_kurtosis",
    "z_crest_factor",
    "x_crest_factor",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
]

STATE_FAMILIES = {
    "normal",
    "baseline",
    "teste",
    "acelerando",
    "motor_desligado",
}

def load_history():
    # Carrega histórico já normalizado
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["created_at"],
    )

def validate_event(event):
    # Valida os campos necessários para a análise
    required = FEATURES + ["rpm"]

    missing = [
        field
        for field in required
        if field not in event
    ]

    if missing:
        return {
            "valid": False,
            "message": (
                "Evento inválido. Campos ausentes: "
                + ", ".join(missing)
            ),
        }

    invalid = []

    for field in required:
        try:
            float(event[field])
        except (TypeError, ValueError):
            invalid.append(field)

    if invalid:
        return {
            "valid": False,
            "message": (
                "Evento inválido. Campos não numéricos: "
                + ", ".join(invalid)
            ),
        }

    return {"valid": True}

def find_similar_events(
    event,
    history,
    top_k=TOP_K,
):
    rpm = float(event["rpm"])

    # RPM é tratado como contexto operacional
    rpm_history = history[
        history["rpm"] == rpm
    ].copy()

    # Evita leakage quando o próprio evento já existe no histórico
    event_id = event.get("id")

    if event_id is not None:
        rpm_history = rpm_history[
            rpm_history["id"] != event_id
        ].copy()

    if len(rpm_history) <= top_k:
        return {
            "status": "error",
            "message": (
                "Não existem eventos históricos suficientes "
                f"para o regime de {rpm:g} RPM."
            ),
        }

    scaler = RobustScaler()

    historical_features = rpm_history[FEATURES]
    historical_scaled = scaler.fit_transform(
        historical_features
    )

    event_df = pd.DataFrame(
        [
            {
                feature: float(event[feature])
                for feature in FEATURES
            }
        ]
    )

    event_scaled = scaler.transform(
        event_df
    )

    model = NearestNeighbors(
        n_neighbors=top_k,
        metric="euclidean",
        n_jobs=-1,
    )

    model.fit(historical_scaled)

    distances, indices = model.kneighbors(
        event_scaled
    )

    neighbors = rpm_history.iloc[
        indices[0]
    ].copy()

    neighbors["distance"] = distances[0]

    family_counts = (
        neighbors["fault_family"]
        .value_counts()
    )

    candidate_family = family_counts.index[0]
    candidate_count = int(
        family_counts.iloc[0]
    )

    historical_support = (
        candidate_count / len(neighbors)
    )

    # Distância média dos vizinhos recuperados
    mean_distance = float(
        neighbors["distance"].mean()
    )

    return {
        "status": "ok",
        "candidate_family": candidate_family,
        "historical_support": historical_support,
        "mean_distance": mean_distance,
        "neighbors": neighbors,
    }

def serialize_neighbors(neighbors):
    # Mantém somente informações úteis para auditoria
    serialized = []

    for _, row in neighbors.iterrows():
        serialized.append(
            {
                "id": int(row["id"]),
                "created_at": str(row["created_at"]),
                "fault_original": row["fault_original"],
                "fault_family": row["fault_family"],
                "rpm": float(row["rpm"]),
                "distance": float(row["distance"]),
            }
        )

    return serialized


def build_historical_summary(history, family, rpm):
    # Resume a presença histórica da família candidata
    family_history = history[
        history["fault_family"] == family
    ].copy()

    same_rpm = family_history[
        family_history["rpm"] == rpm
    ].copy()

    if family_history.empty:
        return {
            "total_occurrences": 0,
            "same_rpm_occurrences": 0,
            "first_occurrence": None,
            "last_occurrence": None,
            "frequency_per_day": 0.0,
            "context_rpm": rpm,
        }

    dates = pd.to_datetime(
        family_history["created_at"],
        errors="coerce",
        utc=True,
    ).dropna()

    first_occurrence = dates.min()
    last_occurrence = dates.max()

    if len(dates) > 0:
        span_days = max(
            (last_occurrence - first_occurrence).total_seconds()
            / 86400,
            1.0,
        )
        frequency_per_day = len(family_history) / span_days
    else:
        frequency_per_day = 0.0

    return {
        "total_occurrences": int(len(family_history)),
        "same_rpm_occurrences": int(len(same_rpm)),
        "first_occurrence": (
            first_occurrence.isoformat()
            if pd.notna(first_occurrence)
            else None
        ),
        "last_occurrence": (
            last_occurrence.isoformat()
            if pd.notna(last_occurrence)
            else None
        ),
        "frequency_per_day": round(
            float(frequency_per_day),
            2,
        ),
        "context_rpm": float(rpm),
    }

def analyze_event(
    event,
    history,
    chunks,
    embedding_model,
):
    # 1. Validação
    validation = validate_event(event)

    if not validation["valid"]:
        return {
            "status": "invalid_input",
            "message": validation["message"],
        }

    # 2. Busca histórica
    similarity = find_similar_events(
        event,
        history,
    )

    if similarity["status"] != "ok":
        return similarity

    family = similarity["candidate_family"]
    neighbors = similarity["neighbors"]

    historical_summary = build_historical_summary(
        history=history,
        family=family,
        rpm=float(event["rpm"]),
    )

    base_result = {
        "event_id": event.get("id"),
        "rpm": float(event["rpm"]),
        "candidate_family": family,
        "historical_support": round(
            similarity["historical_support"],
            4,
        ),
        "mean_distance": round(
            similarity["mean_distance"],
            4,
        ),
        "historical_summary": historical_summary,
        "similar_events": serialize_neighbors(
            neighbors
        ),
    }

    # 3. Confiança mínima antes de gerar recomendação
    if (
        similarity["historical_support"] < MIN_HISTORICAL_SUPPORT
        or similarity["mean_distance"] > MAX_MEAN_DISTANCE
    ):
        return {
            "status": "abstain",
            **base_result,
            "message": (
                "A evidência histórica não atingiu o nível mínimo "
                "definido para gerar uma recomendação automática. "
                "O evento deve ser encaminhado para avaliação técnica."
            ),
            "recommendation": None,
            "sources": [],
            "abstain_reason": "low_confidence",
        }

    # 4. Estados não são tratados como falhas
    if family in STATE_FAMILIES:
        return {
            "status": "state",
            **base_result,
            "message": (
                "Os eventos históricos mais semelhantes "
                f"indicam o estado '{family}'. "
                "Nenhuma recomendação corretiva foi gerada."
            ),
        }

    # 5. Pergunta controlada para o RAG
    question = (
        "Com base exclusivamente na documentação técnica "
        "disponível, quais evidências devem ser verificadas "
        "e quais ações de inspeção ou correção são indicadas "
        f"para a família candidata '{family}'?"
    )

    # 6. RAG + OpenAI
    rag_result = generate_rag_response(
        family=family,
        question=question,
        chunks=chunks,
        model=embedding_model,
    )

    if rag_result["status"] == "abstain":
        return {
            "status": "abstain",
            **base_result,
            "message": (
                "Não existe documentação técnica autorizada para "
                f"a família '{family}'. Registre um novo documento "
                "técnico para o defeito antes de solicitar uma "
                "recomendação automática."
            ),
            "recommendation": None,
            "sources": [],
            "abstain_reason": "no_documentation",
        }

    if rag_result["status"] == "error":
        return {
            "status": "error",
            **base_result,
            "message": rag_result["message"],
        }

    # 7. Resultado final
    return {
        "status": "ready",
        **base_result,
        "recommendation": rag_result["answer"],
        "sources": rag_result[
            "retrieved_sources"
        ],
    }

def print_result(result):
    print("\n" + "=" * 90)
    print("FIESC - PIPELINE DE MANUTENÇÃO PRESCRITIVA")
    print("=" * 90)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

def main():
    print("=" * 90)
    print("FIESC - PIPELINE END-TO-END")
    print("=" * 90)
    
    # Inicializa persistência local
    init_db()

    print("\nCarregando histórico...")
    history = load_history()

    print(
        f"Eventos históricos: "
        f"{len(history):,}"
    )

    print("Carregando documentação...")
    chunks = load_chunks()

    print(
        f"Chunks documentais: "
        f"{len(chunks)}"
    )

    print("Carregando modelo de embeddings...")

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    # Evento do case usado apenas como demonstração.
    # O campo fault não participa da inferência.
    test_event = {
        "id": 114387,
        "created_at": (
            "2026-06-01 "
            "21:32:53.911176+00:00"
        ),
        "z_rms_velocity_mm_s": 1.517,
        "temperature_c": 24.69,
        "x_rms_velocity_mm_s": 2.0,
        "z_peak_acceleration_g": 0.484,
        "x_peak_acceleration_g": 0.631,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 0.09,
        "x_rms_acceleration_g": 0.114,
        "z_kurtosis": 2.392,
        "x_kurtosis": 2.77,
        "z_crest_factor": 3.747,
        "x_crest_factor": 4.269,
        "z_high_freq_rms_accel_g": 0.129,
        "x_high_freq_rms_accel_g": 0.147,
        "rpm": 1000.0,
    }

    print("\nAnalisando evento...")

    result = analyze_event(
        event=test_event,
        history=history,
        chunks=chunks,
        embedding_model=embedding_model,
    )
    
    # Registra a análise para auditoria
    analysis_id = save_analysis(result)

    print(
        f"\nAnálise registrada no banco "
        f"com ID: {analysis_id}"
    )

    print_result(result)

if __name__ == "__main__":
    main()