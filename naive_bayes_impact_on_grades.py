from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


DATA_PATH = "AI_Student_Life_Pakistan_2026.csv"
TARGET_COL = "Impact_on_Grades"
RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs/naive_bayes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return clean_text_columns(df)


def build_probability_table(
    feature_names: list[str],
    class_labels: list[str],
    feature_log_prob: np.ndarray,
) -> pd.DataFrame:
    probability_rows = []
    probabilities = np.exp(feature_log_prob)

    for feature_index, feature_name in enumerate(feature_names):
        values = probabilities[:, feature_index]

        row = {
            "feature": feature_name,
            "max_prob": float(values.max()),
            "min_prob": float(values.min()),
            "spread": float(values.max() - values.min()),
        }

        for class_index, class_label in enumerate(class_labels):
            row[f"P(feature=1 | {class_label})"] = float(values[class_index])

        probability_rows.append(row)

    probability_df = (
        pd.DataFrame(probability_rows)
        .sort_values("spread", ascending=False)
        .reset_index(drop=True)
    )

    return probability_df


def train_and_evaluate():
    print_header("1) Leitura dos dados")
    df = load_data()

    print(f"Shape do dataset: {df.shape}")
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nDistribuicao do alvo:")
    print(df[TARGET_COL].value_counts())

    print_header("2) Separacao de X e y")
    drop_cols = [TARGET_COL]

    if "Student_ID" in df.columns:
        drop_cols.append("Student_ID")

    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]

    print("Colunas de entrada (X):", list(X.columns))
    print("Classes do alvo (y):", sorted(y.unique().tolist()))

    print_header("3) Train/Test split estratificado")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    print_header("4) Pre-processamento para Naive Bayes")
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print("Features numericas:", numeric_features)
    print("Features categoricas:", categorical_features)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "binner",
                KBinsDiscretizer(
                    n_bins=3,
                    encode="onehot-dense",
                    strategy="quantile",
                ),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print_header("5) Treino do Naive Bayes")
    model = BernoulliNB(alpha=1.0)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    print_header("6) Avaliacao")
    y_pred = clf.predict(X_test)

    fitted_preprocessor = clf.named_steps["preprocessor"]
    fitted_model = clf.named_steps["model"]

    labels = fitted_model.classes_.tolist()

    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=labels))

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Matriz de confusao - Naive Bayes")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    save_plot("01_confusion_matrix.png")

    print_header("7) Analise das probabilidades condicionais")
    feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    probability_df = build_probability_table(
        feature_names=feature_names,
        class_labels=labels,
        feature_log_prob=fitted_model.feature_log_prob_,
    )

    probability_columns = (
        ["feature", "max_prob", "min_prob", "spread"]
        + [f"P(feature=1 | {label})" for label in labels]
    )

    print("Top 10 features com maior diferenca entre classes:")
    print(probability_df[probability_columns].head(10))

    probability_df.to_csv(
        OUTPUT_DIR / "02_conditional_probabilities.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=probability_df.head(10),
        x="spread",
        y="feature",
        orient="h",
    )
    plt.title("Top 10 - spread das probabilidades condicionais")
    plt.xlabel("Diferença entre maior e menor P(feature=1 | classe)")
    plt.ylabel("Feature")
    save_plot("02_conditional_probabilities_top10.png")

    print_header("8) Probabilidades por classe")
    class_prior_df = pd.DataFrame(
        {
            "class": labels,
            "class_prior": np.exp(fitted_model.class_log_prior_),
        }
    )

    print(class_prior_df)

    class_prior_df.to_csv(
        OUTPUT_DIR / "03_class_priors.csv",
        index=False,
    )

    print_header("9) Resumo interpretativo")
    top_features = probability_df.head(5)

    for _, row in top_features.iterrows():
        feature = row["feature"]
        print(f"Feature: {feature}")

        for class_label in labels:
            column_name = f"P(feature=1 | {class_label})"
            print(f"  {class_label}: {row[column_name]:.3f}")

    print_header("Pipeline finalizado")
    print(f"Artefatos gerados em: {OUTPUT_DIR.resolve()}")

    return clf, df, X.columns.tolist(), numeric_features, categorical_features


def interactive_prediction(
    clf,
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    print("\n" + "=" * 90)
    print("PREVISÃO INTERATIVA DE NOVOS ALUNOS (Naive Bayes)")
    print("Digite os dados do aluno. Para sair, digite 'n' em qualquer campo.")
    print("=" * 90)

    while True:
        novo = {}
        sair = False

        for col in numeric_features:
            while True:
                val = input(f"Digite o valor para {col} (número): ")

                if val.lower() == "n":
                    sair = True
                    break

                try:
                    novo[col] = float(val)
                    break
                except ValueError:
                    print("Valor inválido. Digite um número.")

            if sair:
                break

        if sair:
            print("Saindo da previsão interativa.")
            break

        for col in categorical_features:
            opcoes = sorted(df[col].dropna().unique().tolist())
            opcoes_dict = {str(i + 1): op for i, op in enumerate(opcoes)}
            opcoes_str = ", ".join(
                [f"{i + 1}: {op}" for i, op in enumerate(opcoes)]
            )

            while True:
                print(f"Opções para {col}: {opcoes_str}")
                val = input(f"Digite o número da opção para {col}: ")

                if val.lower() == "n":
                    sair = True
                    break

                if val in opcoes_dict:
                    novo[col] = opcoes_dict[val]
                    break

                print("Opção inválida. Digite o número correspondente a uma das opções.")

            if sair:
                break

        if sair:
            print("Saindo da previsão interativa.")
            break

        novo_df = pd.DataFrame([novo], columns=feature_cols)

        pred = clf.predict(novo_df)[0]
        proba = clf.predict_proba(novo_df)[0]

        classes = clf.named_steps["model"].classes_

        probas_df = pd.DataFrame(
            {
                "classe": classes,
                "probabilidade": proba,
            }
        ).sort_values("probabilidade", ascending=False)

        print(f"\nPrevisão do Naive Bayes para o novo aluno: {pred}")
        print("\nProbabilidades por classe:")
        print(probas_df.to_string(index=False))
        print()


if __name__ == "__main__":
    clf, df, feature_cols, numeric_features, categorical_features = train_and_evaluate()

    interactive_prediction(
        clf=clf,
        df=df,
        feature_cols=feature_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )