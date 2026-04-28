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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


DATA_PATH = "AI_Student_Life_Pakistan_2026.csv"
TARGET_COL = "Impact_on_Grades"
RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs/decision_tree")
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


def original_feature_name(transformed_name: str, original_features: list[str]) -> str:
    base_name = transformed_name.split("__", 1)[-1]

    for feature in sorted(original_features, key=len, reverse=True):
        if base_name == feature or base_name.startswith(f"{feature}_"):
            return feature

    return base_name


def aggregate_tree_importances(
    feature_names: list[str],
    importances: np.ndarray,
    original_features: list[str],
) -> pd.DataFrame:
    grouped: dict[str, float] = {}

    for name, importance in zip(feature_names, importances):
        feature = original_feature_name(name, original_features)
        grouped[feature] = grouped.get(feature, 0.0) + float(importance)

    importance_df = (
        pd.DataFrame(
            {
                "feature": list(grouped.keys()),
                "importance": list(grouped.values()),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


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

    print_header("4) Pre-processamento")
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print("Features numericas:", numeric_features)
    print("Features categoricas:", categorical_features)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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

    print_header("5) Treino da arvore de decisao")
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

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
    plt.title("Matriz de confusao - Decision Tree")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    save_plot("01_confusion_matrix.png")

    print_header("7) Importancia das features")
    feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    transformed_importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": fitted_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    transformed_importance_df.to_csv(
        OUTPUT_DIR / "02_tree_importance_transformed.csv",
        index=False,
    )

    grouped_importance_df = aggregate_tree_importances(
        feature_names=feature_names,
        importances=fitted_model.feature_importances_,
        original_features=X.columns.tolist(),
    )

    print("Top 10 features mais importantes:")
    print(grouped_importance_df.head(10))

    grouped_importance_df.to_csv(
        OUTPUT_DIR / "03_tree_importance_grouped.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=grouped_importance_df.head(10),
        x="importance",
        y="feature",
        orient="h",
    )
    plt.title("Top 10 features - Decision Tree")
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    save_plot("02_tree_importance_top10.png")

    print_header("8) Estrutura da arvore")
    tree_text = export_text(
        fitted_model,
        feature_names=feature_names,
        max_depth=4,
    )

    print(tree_text)

    tree_text_path = OUTPUT_DIR / "04_tree_structure.txt"
    tree_text_path.write_text(tree_text, encoding="utf-8")

    plt.figure(figsize=(22, 12))
    plot_tree(
        fitted_model,
        feature_names=feature_names,
        class_names=[str(label) for label in labels],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    save_plot("05_decision_tree_plot.png")

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
    print("PREVISÃO INTERATIVA DE NOVOS ALUNOS (Árvore de Decisão)")
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

        print(f"\nPrevisão da Árvore de Decisão para o novo aluno: {pred}")
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
