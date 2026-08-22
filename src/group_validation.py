from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "banner_normalized.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "group_validation_results.csv"

TARGET = "fault_family"

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

def evaluate_model(train, test):
    # Modelo igual ao baseline anterior
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

def main():
    print("=" * 90)
    print("FIESC - VALIDAÇÃO POR GRUPOS")
    print("=" * 90)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["created_at"],
    )

    # O rótulo original funciona como aproximação da condição/ensaio
    group_column = "fault"

    print(f"\nRegistros : {len(df):,}")
    print(f"Famílias  : {df[TARGET].nunique()}")
    print(f"Grupos    : {df[group_column].nunique()}")

    # Divide grupos inteiros entre treino e teste
    splitter = GroupShuffleSplit(
        n_splits=5,
        test_size=0.20,
        random_state=42,
    )

    results = []

    print("\nExecutando 5 divisões independentes por grupo...\n")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(
            df,
            y=df[TARGET],
            groups=df[group_column],
        ),
        start=1,
    ):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        train_groups = set(train[group_column])
        test_groups = set(test[group_column])

        # Confirma que nenhum grupo vazou entre treino e teste
        overlap = train_groups & test_groups

        train_families = set(train[TARGET])
        test_families = set(test[TARGET])

        unseen_families = test_families - train_families

        # Closed-set para avaliar somente famílias conhecidas
        closed_test = test[
            test[TARGET].isin(train_families)
        ].copy()

        print("-" * 90)
        print(f"FOLD {fold}")
        print("-" * 90)
        print(f"Treino                 : {len(train):,}")
        print(f"Teste total            : {len(test):,}")
        print(f"Grupos treino          : {len(train_groups)}")
        print(f"Grupos teste           : {len(test_groups)}")
        print(f"Sobreposição de grupos : {len(overlap)}")
        print(f"Famílias inéditas      : {len(unseen_families)}")

        if unseen_families:
            print(
                "Classes inéditas       : "
                + ", ".join(sorted(unseen_families))
            )

        removed = len(test) - len(closed_test)

        print(f"Teste closed-set       : {len(closed_test):,}")
        print(f"Removidos              : {removed:,}")

        # Evita avaliação vazia
        if len(closed_test) == 0:
            print("Sem amostras closed-set neste fold.")
            continue

        metrics = evaluate_model(
            train,
            closed_test,
        )

        print(f"Accuracy               : {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro F1               : {metrics['macro_f1'] * 100:.2f}%")
        print(f"Weighted F1            : {metrics['weighted_f1'] * 100:.2f}%")

        results.append({
            "fold": fold,
            "train_size": len(train),
            "test_size": len(test),
            "closed_test_size": len(closed_test),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap": len(overlap),
            "unseen_families": len(unseen_families),
            "removed_open_set": removed,
            **metrics,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("RESULTADO MÉDIO")
    print("=" * 90)

    if not results_df.empty:
        print(
            f"Accuracy média    : "
            f"{results_df['accuracy'].mean() * 100:.2f}%"
        )

        print(
            f"Macro F1 médio    : "
            f"{results_df['macro_f1'].mean() * 100:.2f}%"
        )

        print(
            f"Weighted F1 médio : "
            f"{results_df['weighted_f1'].mean() * 100:.2f}%"
        )

        print(
            f"Desvio Accuracy   : "
            f"{results_df['accuracy'].std() * 100:.2f} pp"
        )

        OUTPUT_PATH.parent.mkdir(exist_ok=True)

        results_df.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print("\nResultados salvos em:")
        print(OUTPUT_PATH)

    print("\n" + "=" * 90)
    print("VALIDAÇÃO CONCLUÍDA")
    print("=" * 90)

if __name__ == "__main__":
    main()