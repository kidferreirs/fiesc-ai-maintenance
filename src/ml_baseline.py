from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "banner_normalized.csv"

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
    "rpm",
]

TARGET = "fault_family"

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{name}")
    print("-" * 80)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(f"Accuracy    : {accuracy * 100:.2f}%")
    print(f"Macro F1    : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1 : {weighted_f1 * 100:.2f}%")

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

def random_split(df):
    # Baseline aleatório para comparação
    train = df.sample(
        frac=0.8,
        random_state=42,
    )

    test = df.drop(train.index)
    return train, test

def temporal_split(df):
    # Simula treino com dados passados e teste com dados futuros
    df = (
        df
        .sort_values("created_at")
        .reset_index(drop=True)
    )

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test

def temporal_closed_set(train, test):
    # Mantém no teste apenas famílias já conhecidas no treino
    known_families = set(train[TARGET].unique())

    closed_test = test[
        test[TARGET].isin(known_families)
    ].copy()

    removed = len(test) - len(closed_test)
    return closed_test, removed

def build_models():
    # Baseline linear
    linear_model = Pipeline([
        ("scaler", RobustScaler()),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss",
                max_iter=1000,
                tol=1e-3,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # Modelo não linear
    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return {
        "SGD Logistic": linear_model,
        "Random Forest": random_forest,
    }


def run_split_experiment(split_name, train, test):
    print("\n" + "=" * 90)
    print(f"VALIDAÇÃO: {split_name}")
    print("=" * 90)
    print(f"Treino: {len(train):,}")
    print(f"Teste : {len(test):,}")

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    results = []

    for name, model in build_models().items():
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        result["split"] = split_name
        result["train_size"] = len(train)
        result["test_size"] = len(test)
        results.append(result)

    return results

def main():
    print("=" * 90)
    print("FIESC - BASELINE SUPERVISIONADO DE MACHINE LEARNING")
    print("=" * 90)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["created_at"],
    )

    df = df[df[TARGET] != "unknown"].copy()

    print(f"\nRegistros utilizados: {len(df):,}")
    print(f"Famílias avaliadas   : {df[TARGET].nunique()}")

    results = []

    # Random split
    train_random, test_random = random_split(df)

    results.extend(
        run_split_experiment(
            "Random Split 80/20",
            train_random,
            test_random,
        )
    )

    # Temporal open-set
    train_temporal, test_temporal = temporal_split(df)

    results.extend(
        run_split_experiment(
            "Temporal Open-Set 80/20",
            train_temporal,
            test_temporal,
        )
    )

    # Temporal closed-set
    closed_test, removed = temporal_closed_set(
        train_temporal,
        test_temporal,
    )

    print("\nEventos removidos do closed-set por serem classes inéditas:")
    print(f"{removed:,}")
    print(
        "Percentual do teste original: "
        f"{removed / len(test_temporal) * 100:.2f}%"
    )

    results.extend(
        run_split_experiment(
            "Temporal Closed-Set 80/20",
            train_temporal,
            closed_test,
        )
    )

    # Resumo
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("RESUMO")
    print("=" * 90)
    print(
        results_df[
            [
                "split",
                "model",
                "train_size",
                "test_size",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "macro_f1": "{:.4f}".format,
                "weighted_f1": "{:.4f}".format,
            },
        )
    )

    # Salva resultados
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = reports_dir / "ml_baseline_results.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\nResultados salvos em:")
    print(output_path)
    print("\n" + "=" * 90)
    print("EXPERIMENTO CONCLUÍDO")
    print("=" * 90)

if __name__ == "__main__":
    main()