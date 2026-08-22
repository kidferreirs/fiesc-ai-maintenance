from pathlib import Path

import pandas as pd

# CONFIGURAÇÃO DE CAMINHOS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "banner.csv"

def main():
    print("=" * 70)
    print("FIESC - AUDITORIA INICIAL DO DATASET")
    print("=" * 70)

    # CARREGAMENTO DOS DADOS
    df = pd.read_csv(DATA_PATH)
    print("\n[1] DIMENSÕES")
    print(f"Linhas : {df.shape[0]:,}")
    print(f"Colunas: {df.shape[1]}")

    # COLUNAS E TIPOS
    print("\n[2] COLUNAS E TIPOS")
    print(df.dtypes.to_string())

    # VALORES AUSENTES
    print("\n[3] VALORES AUSENTES")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("Nenhum valor ausente encontrado.")
    else:
        print(missing.to_string())

    # DUPLICIDADES
    print("\n[4] DUPLICIDADES")
    print(f"Linhas duplicadas: {df.duplicated().sum():,}")

    if "id" in df.columns:
        print(f"IDs duplicados: {df['id'].duplicated().sum():,}")
    
    # PERÍODO DOS DADOS
    if "created_at" in df.columns:
        print("\n[5] PERÍODO DOS DADOS")

        dates = pd.to_datetime(
            df["created_at"],
            errors="coerce",
            utc=True
        )
        print(f"Datas inválidas: {dates.isna().sum():,}")
        print(f"Data inicial   : {dates.min()}")
        print(f"Data final     : {dates.max()}")

    # DISTRIBUIÇÃO DE FAULT
    if "fault" in df.columns:
        print("\n[6] DISTRIBUIÇÃO DE FAULT")
        print(f"Valores distintos: {df['fault'].nunique()}")
        print("\n20 valores mais frequentes:")
        print(
            df["fault"]
            .value_counts()
            .head(20)
            .to_string()
        )

    # DISTRIBUIÇÃO DE RPM
    if "rpm" in df.columns:
        print("\n[7] DISTRIBUIÇÃO DE RPM")
        print(
            df["rpm"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # ESTATÍSTICAS DAS VARIÁVEIS NUMÉRICAS
    print("\n[8] ESTATÍSTICAS DAS VARIÁVEIS NUMÉRICAS")

    numeric_df = df.select_dtypes(include="number")

    print(
        numeric_df
        .describe()
        .T
        .round(4)
        .to_string()
    )
   
    #ANÁLISE DE CORRELAÇÃO
    # IMPORTANTE:
    # uma variável NÃO será removida apenas por apresentar alta
    print("\n[9] CORRELAÇÕES ABSOLUTAS >= 0.95")

    corr_df = numeric_df.drop(
        columns=["id"],
        errors="ignore"
    )

    corr = corr_df.corr().abs()

    pairs = []
    columns = corr.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]

            if value >= 0.95:
                pairs.append(
                    (
                        columns[i],
                        columns[j],
                        value
                    )
                )

    pairs.sort(
        key=lambda item: item[2],
        reverse=True
    )

    if not pairs:
        print("Nenhum par encontrado.")
    else:
        for col1, col2, value in pairs:
            print(
                f"{col1:<32} <-> "
                f"{col2:<32} "
                f"{value:.6f}"
            )

    print("\n" + "=" * 70)
    print("AUDITORIA CONCLUÍDA")
    print("=" * 70)

# ENTRY POINT
if __name__ == "__main__":
    main()