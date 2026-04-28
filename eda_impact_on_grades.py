import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = "AI_Student_Life_Pakistan_2026.csv"
TARGET = "Impact_on_Grades"
OUTPUT_DIR = Path("outputs/eda")
SHOW_PLOTS = False


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def save_plot(filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Arquivo nao encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    print_section("1) Estrutura inicial")
    print(f"Shape: {df.shape}")
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nTipos de dados:")
    print(df.dtypes)

    print_section("2) Qualidade dos dados")
    print("Missing values por coluna:")
    print(df.isnull().sum())
    print(f"\nDuplicados: {df.duplicated().sum()}")

    print_section("3) Distribuicao da variavel alvo")
    if TARGET not in df.columns:
        raise KeyError(f"Variavel alvo '{TARGET}' nao existe no dataset.")

    counts = df[TARGET].value_counts()
    props = df[TARGET].value_counts(normalize=True).mul(100).round(2)
    print("Contagem do alvo:")
    print(counts)
    print("\nProporcao do alvo (%):")
    print(props)

    plt.figure(figsize=(8, 4.5))
    sns.countplot(data=df, x=TARGET, order=counts.index)
    plt.title("Distribuicao de Impact_on_Grades")
    plt.xlabel(TARGET)
    plt.ylabel("Frequencia")
    plt.xticks(rotation=15)
    save_plot("01_distribuicao_alvo.png")

    print_section("4) Categoricas x alvo (graficos padrao)")
    cat_cols = [
        "Gender",
        "Education_Level",
        "City",
        "AI_Tool_Used",
        "Purpose",
        "Satisfaction_Level",
    ]

    for col in cat_cols:
        if col not in df.columns:
            print(f"Coluna ausente, pulando: {col}")
            continue

        print(f"\n--- {col} x {TARGET} ---")
        print(pd.crosstab(df[col], df[TARGET]))

        plt.figure(figsize=(9, 4.5))
        sns.countplot(data=df, x=col, hue=TARGET)
        plt.title(f"{col} por {TARGET}")
        plt.xlabel(col)
        plt.ylabel("Frequencia")
        plt.xticks(rotation=20)
        plt.legend(title=TARGET, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_plot(f"02_{col.lower()}_vs_alvo.png")

    print_section("5) Numericas x alvo")
    num_cols = ["Daily_Usage_Hours", "Age"]

    for col in num_cols:
        if col not in df.columns:
            print(f"Coluna ausente, pulando: {col}")
            continue

        print(f"\nResumo estatistico de {col} por {TARGET}:")
        print(df.groupby(TARGET)[col].describe().round(2))

        plt.figure(figsize=(8, 4.5))
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, element="step")
        plt.title(f"Distribuicao de {col} por {TARGET}")
        plt.xlabel(col)
        plt.ylabel("Contagem")
        save_plot(f"03_hist_{col.lower()}_por_alvo.png")

        plt.figure(figsize=(8, 4.5))
        sns.boxplot(data=df, x=TARGET, y=col)
        plt.title(f"{col} por {TARGET}")
        plt.xlabel(TARGET)
        plt.ylabel(col)
        plt.xticks(rotation=15)
        save_plot(f"04_boxplot_{col.lower()}_por_alvo.png")

if __name__ == "__main__":
    main()
