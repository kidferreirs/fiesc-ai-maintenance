from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "banner.csv"

def ratio_stats(df, numerator, denominator, description):
    """Analisa se duas features possuem relação aproximadamente constante."""

    # Evita divisão por zero
    valid = df[denominator] != 0
    ratio = df.loc[valid, numerator] / df.loc[valid, denominator]

    print(f"\n{description}")
    print("-" * 70)
    print(f"Média   : {ratio.mean():.6f}")
    print(f"Mediana : {ratio.median():.6f}")
    print(f"Desvio  : {ratio.std():.6f}")
    print(f"Mínimo  : {ratio.min():.6f}")
    print(f"Máximo  : {ratio.max():.6f}")

def outlier_summary(df, column):
    """Resume valores extremos usando o critério estatístico IQR."""

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    percentage = len(outliers) / len(df) * 100

    print(f"\n{column}")
    print("-" * 70)
    print(f"Q1              : {q1:.6f}")
    print(f"Q3              : {q3:.6f}")
    print(f"IQR             : {iqr:.6f}")
    print(f"Limite inferior : {lower:.6f}")
    print(f"Limite superior : {upper:.6f}")
    print(f"Outliers        : {len(outliers):,} ({percentage:.2f}%)")

def main():
    print("=" * 70)
    print("FIESC - ANÁLISE DAS FEATURES")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    # Conversões de unidade
    print("\n[1] RELAÇÕES ENTRE UNIDADES")

    ratio_stats(
        df,
        "z_rms_velocity_mm_s",
        "z_rms_velocity_in_s",
        "Z RMS: mm/s ÷ in/s"
    )

    ratio_stats(
        df,
        "x_rms_velocity_mm_s",
        "x_rms_velocity_in_s",
        "X RMS: mm/s ÷ in/s"
    )

    # Fahrenheit deve ser função direta de Celsius
    expected_f = (df["temperature_c"] * 9 / 5) + 32
    temp_error = np.abs(df["temperature_f"] - expected_f)

    print("\nTemperatura: erro da conversão C → F")
    print("-" * 70)
    print(f"Erro médio : {temp_error.mean():.6f}")
    print(f"Erro máximo: {temp_error.max():.6f}")

    # Relação Peak / RMS
    print("\n[2] RELAÇÃO PEAK / RMS")

    ratio_stats(
        df,
        "z_peak_velocity_mm_s",
        "z_rms_velocity_mm_s",
        "Z: Peak Velocity ÷ RMS Velocity"
    )

    ratio_stats(
        df,
        "x_peak_velocity_mm_s",
        "x_rms_velocity_mm_s",
        "X: Peak Velocity ÷ RMS Velocity"
    )

    # 3. Valores extremos
    print("\n[3] ANÁLISE DE VALORES EXTREMOS (IQR)")

    outlier_columns = [
        "z_peak_acceleration_g",
        "x_peak_acceleration_g",
        "z_rms_acceleration_g",
        "x_rms_acceleration_g",
        "z_kurtosis",
        "x_kurtosis",
        "z_crest_factor",
        "x_crest_factor",
        "z_high_freq_rms_accel_g",
        "x_high_freq_rms_accel_g",
    ]

    for column in outlier_columns:
        outlier_summary(df, column)

    print("\n" + "=" * 70)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 70)

if __name__ == "__main__":
    main()