from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "banner_normalized.csv"

K_VALUES = [3, 5, 10, 20]
SAMPLE_SIZE = 3000

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


def calculate_metrics(df, sample, neighbor_lists, k):
    purities = []
    majority_hits = []
    topk_hits = []

    for row_position, neighbors in enumerate(neighbor_lists):
        neighbor_indices = neighbors[:k]

        real_fault = sample.iloc[row_position]["fault_family"]
        neighbor_faults = df.loc[neighbor_indices, "fault_family"]

        # Percentual dos vizinhos com o mesmo rótulo
        purities.append((neighbor_faults == real_fault).mean())

        # Verifica se a maioria coincide com o rótulo real
        majority_fault = neighbor_faults.mode().iloc[0]
        majority_hits.append(majority_fault == real_fault)

        # Verifica se ao menos um vizinho possui o mesmo rótulo
        topk_hits.append(real_fault in neighbor_faults.values)

    return {
        "purity": np.mean(purities),
        "majority_accuracy": np.mean(majority_hits),
        "topk_hit": np.mean(topk_hits),
    }


def build_global_neighbors(df, scaler, use_rpm=False):
    features = FEATURES.copy()

    if use_rpm:
        features.append("rpm")

    X = df[features]
    X_scaled = scaler.fit_transform(X)

    # Um único índice atende todos os valores de K
    model = NearestNeighbors(
        n_neighbors=max(K_VALUES) + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    model.fit(X_scaled)

    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=42,
    )

    query = X_scaled[sample.index]
    _, indices = model.kneighbors(query)

    # O primeiro vizinho é o próprio registro
    neighbor_lists = [
        df.index[row[1:]].tolist()
        for row in indices
    ]

    return sample.reset_index(drop=True), neighbor_lists


def build_rpm_filter_neighbors(df, scaler):
    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=42,
    )

    neighbor_map = {}

    # Cria scaler e índice somente uma vez para cada RPM
    for rpm_value, group in df.groupby("rpm"):
        if len(group) <= max(K_VALUES):
            continue

        group_scaler = scaler.__class__()
        X_group = group[FEATURES]
        X_scaled = group_scaler.fit_transform(X_group)

        model = NearestNeighbors(
            n_neighbors=max(K_VALUES) + 1,
            metric="euclidean",
            n_jobs=-1,
        )
        model.fit(X_scaled)

        sample_group = sample[sample["rpm"] == rpm_value]

        if sample_group.empty:
            continue

        query_scaled = group_scaler.transform(sample_group[FEATURES])
        _, local_indices = model.kneighbors(query_scaled)

        for sample_index, row in zip(sample_group.index, local_indices):
            global_indices = group.index[row].tolist()

            # Remove o próprio evento
            global_indices = [
                idx for idx in global_indices
                if idx != sample_index
            ][:max(K_VALUES)]

            if len(global_indices) == max(K_VALUES):
                neighbor_map[sample_index] = global_indices

    valid_indexes = [
        idx for idx in sample.index
        if idx in neighbor_map
    ]

    valid_sample = sample.loc[valid_indexes].reset_index(drop=True)
    neighbor_lists = [
        neighbor_map[idx]
        for idx in valid_indexes
    ]

    return valid_sample, neighbor_lists


def add_results(results, name, df, sample, neighbor_lists):
    for k in K_VALUES:
        metrics = calculate_metrics(
            df,
            sample,
            neighbor_lists,
            k,
        )

        results.append({
            "config": name,
            "k": k,
            **metrics,
        })


def main():
    print("=" * 90)
    print("FIESC - SIMILARIDADE POR FAMÍLIA DE FALHA")
    print("=" * 90)

    df = pd.read_csv(DATA_PATH)
    results = []

    configurations = [
        ("Standard", StandardScaler(), False),
        ("Standard + RPM feature", StandardScaler(), True),
        ("Robust", RobustScaler(), False),
        ("Robust + RPM feature", RobustScaler(), True),
    ]

    # Busca global: cada configuração é calculada apenas uma vez
    for name, scaler, use_rpm in configurations:
        print(f"Processando: {name}...")

        sample, neighbors = build_global_neighbors(
            df,
            scaler,
            use_rpm=use_rpm,
        )

        add_results(
            results,
            name,
            df,
            sample,
            neighbors,
        )

    # RPM como contexto operacional
    print("Processando: Robust + RPM filter...")

    sample, neighbors = build_rpm_filter_neighbors(
        df,
        RobustScaler(),
    )

    add_results(
        results,
        "Robust + RPM filter",
        df,
        sample,
        neighbors,
    )

    results_df = pd.DataFrame(results)

    print(
        "\n"
        f"{'Configuração':<28}"
        f"{'K':>5}"
        f"{'Pureza':>12}"
        f"{'Maioria':>12}"
        f"{'Top-K hit':>12}"
    )
    print("-" * 69)

    for _, row in results_df.iterrows():
        print(
            f"{row['config']:<28}"
            f"{int(row['k']):>5}"
            f"{row['purity'] * 100:>11.2f}%"
            f"{row['majority_accuracy'] * 100:>11.2f}%"
            f"{row['topk_hit'] * 100:>11.2f}%"
        )

    # Salva os resultados para o relatório
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = reports_dir / "similarity_family_experiments.csv"
    results_df.to_csv(output_path, index=False)

    print("\nResultados salvos em:")
    print(output_path)

    print("\n" + "=" * 90)
    print("EXPERIMENTOS CONCLUÍDOS")
    print("=" * 90)

if __name__ == "__main__":
    main()