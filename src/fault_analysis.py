from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "banner.csv"
REPORTS_DIR = BASE_DIR / "reports"

def main():
    print("=" * 90)
    print("FIESC - ANÁLISE DOS RÓTULOS HISTÓRICOS (FAULT)")
    print("=" * 90)

    df = pd.read_csv(DATA_PATH)

    # Garante que o rótulo seja tratado como texto
    df["fault"] = df["fault"].astype(str).str.strip()

    print(f"\nRegistros analisados : {len(df):,}")
    print(f"Rótulos distintos    : {df['fault'].nunique():,}")

    # Frequência dos rótulos
    print("\n[1] DISTRIBUIÇÃO COMPLETA DOS RÓTULOS")

    fault_counts = (
        df["fault"]
        .value_counts()
        .rename_axis("fault")
        .reset_index(name="count")
    )

    fault_counts["percent"] = (
        fault_counts["count"] / len(df) * 100
    )

    print(
        fault_counts.to_string(
            index=False,
            formatters={"percent": "{:.2f}%".format}
        )
    )

    # RPM por rótulo
    print("\n[2] REGIMES DE RPM POR RÓTULO")

    rpm_summary = (
        df.groupby("fault")["rpm"]
        .agg(
            registros="size",
            rpm_distintos="nunique",
            rpm_min="min",
            rpm_max="max",
        )
        .reset_index()
        .sort_values(
            ["registros", "fault"],
            ascending=[False, True]
        )
    )

    rpm_values = (
        df.groupby("fault")["rpm"]
        .apply(
            lambda values: ", ".join(
                str(int(v)) if float(v).is_integer() else str(v)
                for v in sorted(values.unique())
            )
        )
        .rename("rpm_values")
        .reset_index()
    )

    rpm_summary = rpm_summary.merge(
        rpm_values,
        on="fault",
        how="left"
    )

    print(rpm_summary.to_string(index=False))
    
    # Período temporal por rótulo
    print("\n[3] JANELA TEMPORAL POR RÓTULO")

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True
    )

    temporal_summary = (
        df.groupby("fault")["created_at"]
        .agg(
            primeira_ocorrencia="min",
            ultima_ocorrencia="max",
        )
        .reset_index()
    )

    temporal_summary["duracao_horas"] = (
        (
            temporal_summary["ultima_ocorrencia"]
            - temporal_summary["primeira_ocorrencia"]
        )
        .dt.total_seconds()
        / 3600
    )

    print(temporal_summary.to_string(index=False))

    # Possíveis variantes pelo nome
    print("\n[4] RÓTULOS COM SUFIXOS NUMÉRICOS")

    suffix_mask = df["fault"].str.contains(
        r"_\d+$",
        regex=True,
        na=False
    )

    suffix_faults = sorted(
        df.loc[suffix_mask, "fault"].unique()
    )

    print(f"Quantidade: {len(suffix_faults)}")

    for fault in suffix_faults:
        print(f"  {fault}")

    # Remove apenas o sufixo para investigação.
    # O fault original continua preservado.
    df["fault_base_candidate"] = (
        df["fault"]
        .str.replace(
            r"_\d+$",
            "",
            regex=True
        )
    )

    candidate_summary = (
        df.groupby("fault_base_candidate")
        .agg(
            registros=("fault", "size"),
            variantes=("fault", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["variantes", "registros"],
            ascending=[False, False]
        )
    )

    candidate_summary = candidate_summary[
        candidate_summary["variantes"] > 1
    ]

    print("\n[5] POSSÍVEIS FAMÍLIAS TEXTUAIS")
    print(candidate_summary.to_string(index=False))

    # Exporta para análise e documentação
    REPORTS_DIR.mkdir(exist_ok=True)

    fault_report = (
        fault_counts
        .merge(rpm_summary, on="fault", how="left")
        .merge(temporal_summary, on="fault", how="left")
    )

    fault_report.to_csv(
        REPORTS_DIR / "fault_analysis.csv",
        index=False
    )

    candidate_summary.to_csv(
        REPORTS_DIR / "fault_family_candidates.csv",
        index=False
    )

    print("\nArquivos gerados:")
    print(REPORTS_DIR / "fault_analysis.csv")
    print(REPORTS_DIR / "fault_family_candidates.csv")

    print("\n" + "=" * 90)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 90)

if __name__ == "__main__":
    main()