from pathlib import Path
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "banner_normalized.csv"
)

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

# Configuração do experimento
TOP_K = 5
SAMPLE_SIZE = 4000
RANDOM_STATE = 42

def evaluate_event(event, history):
    rpm = float(event["rpm"])

    # Mantém eventos com o mesmo RPM
    rpm_history = history[
        history["rpm"] == rpm
    ].copy()

    # Evita comparar o evento com ele mesmo
    rpm_history = rpm_history[
        rpm_history["id"] != event["id"]
    ].copy()

    if len(rpm_history) <= TOP_K:
        return None

    # Usa a mesma normalização do pipeline
    scaler = RobustScaler()

    X_history = scaler.fit_transform(
        rpm_history[FEATURES]
    )

    event_df = pd.DataFrame(
        [
            {
                feature: float(event[feature])
                for feature in FEATURES
            }
        ]
    )

    X_event = scaler.transform(
        event_df
    )

    # Busca os eventos mais próximos
    model = NearestNeighbors(
        n_neighbors=TOP_K,
        metric="euclidean",
        n_jobs=-1,
    )

    model.fit(X_history)

    distances, indices = model.kneighbors(
        X_event
    )

    neighbors = rpm_history.iloc[
        indices[0]
    ].copy()

    neighbors["distance"] = distances[0]

    counts = (
        neighbors["fault_family"]
        .value_counts()
    )

    # Família predominante entre os vizinhos
    candidate_family = counts.index[0]
    candidate_count = int(counts.iloc[0])

    # Suporte da família candidata no Top-K
    support = (
        candidate_count / TOP_K
    )

    true_family = event["fault_family"]

    return {
        "event_id": int(event["id"]),
        "true_family": true_family,
        "candidate_family": candidate_family,
        "correct": candidate_family == true_family,
        "support": support,
        "nearest_distance": float(
            neighbors["distance"].iloc[0]
        ),
        "mean_distance": float(
            neighbors["distance"].mean()
        ),
        "max_distance": float(
            neighbors["distance"].max()
        ),
    }

def print_summary(results):
    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("RESUMO GERAL")
    print("=" * 90)

    print(
        f"Eventos avaliados : "
        f"{len(df):,}"
    )

    print(
        f"Acerto da família : "
        f"{df['correct'].mean() * 100:.2f}%"
    )

    print("\nDistâncias médias por resultado:")

    # Compara distâncias entre acertos e erros
    summary = (
        df.groupby("correct")[
            [
                "nearest_distance",
                "mean_distance",
                "max_distance",
                "support",
            ]
        ]
        .mean()
    )

    print(summary)

    print("\n" + "=" * 90)
    print("ACERTO POR SUPORTE")
    print("=" * 90)

    # Mede o acerto para cada nível de suporte
    support_summary = (
        df.groupby("support")
        .agg(
            events=("correct", "size"),
            accuracy=("correct", "mean"),
            nearest_distance=(
                "nearest_distance",
                "mean",
            ),
            mean_distance=(
                "mean_distance",
                "mean",
            ),
        )
        .reset_index()
    )

    support_summary["accuracy"] *= 100

    print(
        support_summary.to_string(
            index=False,
            formatters={
                "support": "{:.2f}".format,
                "accuracy": "{:.2f}%".format,
                "nearest_distance": "{:.4f}".format,
                "mean_distance": "{:.4f}".format,
            },
        )
    )

    print("\n" + "=" * 90)
    print("TESTE DE GATES")
    print("=" * 90)

    # Testa limites possíveis para o confidence gate
    gates = [
        (0.60, 0.50),
        (0.60, 0.75),
        (0.60, 1.00),
        (0.80, 0.50),
        (0.80, 0.75),
        (0.80, 1.00),
        (1.00, 0.50),
        (1.00, 0.75),
        (1.00, 1.00),
    ]

    rows = []

    for min_support, max_mean_distance in gates:
        accepted = df[
            (df["support"] >= min_support)
            & (
                df["mean_distance"]
                <= max_mean_distance
            )
        ]

        coverage = (
            len(accepted) / len(df)
            if len(df)
            else 0
        )

        accuracy = (
            accepted["correct"].mean()
            if len(accepted)
            else 0
        )

        rows.append(
            {
                "min_support": min_support,
                "max_mean_distance": (
                    max_mean_distance
                ),
                "accepted": len(accepted),
                "coverage": coverage,
                "accuracy": accuracy,
            }
        )

    gates_df = pd.DataFrame(rows)

    print(
        gates_df.to_string(
            index=False,
            formatters={
                "min_support": "{:.2f}".format,
                "max_mean_distance": "{:.2f}".format,
                "coverage": "{:.2%}".format,
                "accuracy": "{:.2%}".format,
            },
        )
    )

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = (
        reports_dir
        / "confidence_analysis.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\nResultados salvos em:")
    print(output_path)

def main():
    print("=" * 90)
    print("FIESC - ANÁLISE DE CONFIANÇA DA SIMILARIDADE")
    print("=" * 90)

    # Carrega o histórico normalizado
    history = pd.read_csv(
        DATA_PATH,
        parse_dates=["created_at"],
    )

    # Usa uma amostra para manter o experimento rápido
    sample_size = min(
        SAMPLE_SIZE,
        len(history),
    )

    sample = history.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    results = []

    # Avalia cada evento sem usar seu rótulo na inferência
    for position, (_, event) in enumerate(
        sample.iterrows(),
        start=1,
    ):
        result = evaluate_event(
            event,
            history,
        )

        if result is not None:
            results.append(result)

        if position % 250 == 0:
            print(
                f"Processados: "
                f"{position:,}/{sample_size:,}"
            )

    print_summary(results)

if __name__ == "__main__":
    main()