from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "AI_Student_Life_Pakistan_2026.csv"
TARGET_COL = "Impact_on_Grades"
RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs/knn")
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


def main():
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

    print_header("4) Pre-processamento para KNN")
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

    print_header("5) Treino do KNN com GridSearchCV")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "model__n_neighbors": [3, 5, 7, 10, 15, 20],
        "model__weights": ["uniform", "distance"],
    }

    # cv=5: cada combinação é avaliada em 5 folds do treino para estimar
    # a acurácia de forma mais confiável do que usar o conjunto de teste diretamente.
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print(f"Melhor combinação de hiperparâmetros: {search.best_params_}")
    print(f"Acurácia média (CV) com os melhores parâmetros: {search.best_score_:.3f}")

    clf = search.best_estimator_

    print_header("6) Avaliacao")
    y_pred = clf.predict(X_test)
    labels = sorted(y.unique().tolist())

    print("Classification report:")
    print(classification_report(y_test, y_pred))

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
    plt.title("Matriz de confusao - KNN")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    save_plot("01_confusion_matrix.png")

    print_header("7) Importancia global aproximada com permutation importance")
    perm = permutation_importance(
        clf,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("Top 10 features por permutation importance:")
    print(perm_df.head(10))

    perm_df.to_csv(OUTPUT_DIR / "02_permutation_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=perm_df.head(10), x="importance_mean", y="feature", orient="h")
    plt.title("Top 10 - Permutation Importance")
    plt.xlabel("Importance mean")
    plt.ylabel("Feature")
    save_plot("02_permutation_importance_top10.png")

    print_header("8) Interpretabilidade local com LIME")
    fitted_preprocessor = clf.named_steps["preprocessor"]
    fitted_model = clf.named_steps["model"]

    X_train_t = fitted_preprocessor.transform(X_train)
    X_test_t = fitted_preprocessor.transform(X_test)

    feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_train_t),
        feature_names=feature_names,
        class_names=labels,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    idx_to_explain = 0

    exp = lime_explainer.explain_instance(
        data_row=np.array(X_test_t[idx_to_explain]),
        predict_fn=fitted_model.predict_proba,
        num_features=10,
        top_labels=1,
    )

    lime_html_path = OUTPUT_DIR / "03_lime_explanation.html"
    exp.save_to_file(str(lime_html_path))

    print(f"Explicacao LIME salva em: {lime_html_path}")

    print_header("9) Vizinhos mais proximos da instancia explicada")
    n_neighbors = 20

    distances, indices = fitted_model.kneighbors(
        np.array(X_test_t[idx_to_explain]).reshape(1, -1),
        n_neighbors=n_neighbors,
    )

    neighbors_df = pd.DataFrame(
        {
            "neighbor_rank": list(range(1, n_neighbors + 1)),
            "distance": distances[0],
            "train_index": indices[0],
            "neighbor_target": y_train.iloc[indices[0]].values,
        }
    )

    print(neighbors_df)

    neighbors_df.to_csv(OUTPUT_DIR / "04_nearest_neighbors.csv", index=False)

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
    print("PREVISÃO INTERATIVA DE NOVOS ALUNOS (KNN)")
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

        print(f"\nPrevisão do KNN para o novo aluno: {pred}")
        print("\nProbabilidades por classe:")
        print(probas_df.to_string(index=False))
        print()


if __name__ == "__main__":
    clf, df, feature_cols, numeric_features, categorical_features = main()

    interactive_prediction(
        clf=clf,
        df=df,
        feature_cols=feature_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )