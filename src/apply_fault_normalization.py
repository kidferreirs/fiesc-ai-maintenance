from pathlib import Path

import pandas as pd

from fault_normalization import normalize_fault

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "banner.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

def main():
    print("=" * 90)
    print("FIESC - APLICAÇÃO DA NORMALIZAÇÃO DE FAULT")
    print("=" * 90)

    df = pd.read_csv(DATA_PATH)

    normalized = df["fault"].apply(normalize_fault)

    normalized_df = pd.DataFrame(normalized.tolist())

    df["fault_original"] = normalized_df["original"]
    df["fault_family"] = normalized_df["family"]
    df["condition_type"] = normalized_df["condition_type"]
    df["documented"] = normalized_df["documented"]

    print("\n[1] FAMÍLIAS IDENTIFICADAS")
    family_counts = (
        df["fault_family"]
        .value_counts()
        .rename_axis("fault_family")
        .reset_index(name="count")
    )

    family_counts["percent"] = (
        family_counts["count"] / len(df) * 100
    )

    print(
        family_counts.to_string(
            index=False,
            formatters={"percent": "{:.2f}%".format}
        )
    )

    print("\n[2] TIPOS DE CONDIÇÃO")
    print(
        df["condition_type"]
        .value_counts()
        .to_string()
    )

    print("\n[3] COBERTURA DOCUMENTAL")
    problem_df = df[df["condition_type"] == "problem"]

    documented_counts = (
        problem_df["documented"]
        .value_counts()
        .rename(index={True: "documentado", False: "sem_documento"})
    )

    print(documented_counts.to_string())

    print("\n[4] RÓTULOS NÃO RECONHECIDOS")
    unknown = (
        df[df["fault_family"] == "unknown"]["fault"]
        .value_counts()
    )

    if unknown.empty:
        print("Nenhum rótulo desconhecido.")
    else:
        print(unknown.to_string())

    # Salva dataset enriquecido
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_data = PROCESSED_DIR / "banner_normalized.csv"
    df.to_csv(output_data, index=False)

    # Salva resumo das famílias
    REPORTS_DIR.mkdir(exist_ok=True)

    family_report = REPORTS_DIR / "fault_family_summary.csv"
    family_counts.to_csv(family_report, index=False)

    print("\nArquivos gerados:")
    print(output_data)
    print(family_report)
    print("\n" + "=" * 90)
    print("NORMALIZAÇÃO CONCLUÍDA")
    print("=" * 90)

if __name__ == "__main__":
    main()