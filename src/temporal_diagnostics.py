from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "banner_normalized.csv"
TARGET = "fault_family"

def main():
    print("=" * 90)
    print("FIESC - DIAGNÓSTICO DA VALIDAÇÃO TEMPORAL")
    print("=" * 90)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["created_at"],
    )

    # Ordena os eventos para simular treino com passado e teste com futuro
    df = df.sort_values("created_at").reset_index(drop=True)

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    cutoff = test["created_at"].min()
    
    # 1. Corte temporal
    print("\n[1] CORTE TEMPORAL")
    print(f"Treino           : {len(train):,}")
    print(f"Teste            : {len(test):,}")
    print(f"Último treino    : {train['created_at'].max()}")
    print(f"Primeiro teste   : {test['created_at'].min()}")
    print(f"Data de corte    : {cutoff}")

    train_families = set(train[TARGET].unique())
    test_families = set(test[TARGET].unique())
    
    # 2. Verifica classes novas no período futuro
    print("\n[2] COBERTURA DE CLASSES")
    print(f"Famílias no treino: {len(train_families)}")
    print(f"Famílias no teste : {len(test_families)}")

    unseen = sorted(test_families - train_families)

    if unseen:
        print("\nFamílias presentes no teste e AUSENTES no treino:")
        for family in unseen:
            print(f"  - {family}")
    else:
        print("\nTodas as famílias do teste também existem no treino.")

    # 3. Distribuição no treino
    print("\n[3] DISTRIBUIÇÃO NO TREINO")

    train_counts = train[TARGET].value_counts()

    for family, count in train_counts.items():
        pct = count / len(train) * 100
        print(f"{family:<25} {count:>8,}  {pct:>6.2f}%")

    # 4. Distribuição no teste
    print("\n[4] DISTRIBUIÇÃO NO TESTE")

    test_counts = test[TARGET].value_counts()

    for family, count in test_counts.items():
        pct = count / len(test) * 100
        print(f"{family:<25} {count:>8,}  {pct:>6.2f}%")

    # 5. Mede mudança de distribuição entre passado e futuro
    print("\n[5] COMPARAÇÃO DE DISTRIBUIÇÃO")

    families = sorted(train_families | test_families)
    comparison = []

    for family in families:
        train_count = (train[TARGET] == family).sum()
        test_count = (test[TARGET] == family).sum()

        train_pct = train_count / len(train) * 100
        test_pct = test_count / len(test) * 100

        comparison.append({
            "family": family,
            "train_count": train_count,
            "test_count": test_count,
            "train_percent": train_pct,
            "test_percent": test_pct,
            "difference_pp": test_pct - train_pct,
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df["abs_difference"] = (
        comparison_df["difference_pp"].abs()
    )
    comparison_df = comparison_df.sort_values(
        "abs_difference",
        ascending=False,
    )

    print(
        comparison_df[
            [
                "family",
                "train_count",
                "test_count",
                "train_percent",
                "test_percent",
                "difference_pp",
            ]
        ].to_string(
            index=False,
            formatters={
                "train_percent": "{:.2f}%".format,
                "test_percent": "{:.2f}%".format,
                "difference_pp": "{:+.2f} pp".format,
            },
        )
    )

    # Guarda os resultados para documentação da POC
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = reports_dir / "temporal_distribution.csv"
    comparison_df.to_csv(output_path, index=False)

    print("\nResultado salvo em:")
    print(output_path)
    print("\n" + "=" * 90)
    print("DIAGNÓSTICO CONCLUÍDO")
    print("=" * 90)

if __name__ == "__main__":
    main()