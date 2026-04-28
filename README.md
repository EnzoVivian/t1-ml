# T1 – Interpretabilidade de Modelos de Aprendizado de Máquina

## Dataset

**AI Student Life in Pakistan 2026**: dataset do Kaggle (https://www.kaggle.com/datasets/guriya79/how-ai-is-changing-student-life) que registra o uso de ferramentas de IA por estudantes paquistaneses e o impacto percebido nas notas.

| Atributo | Tipo | Descrição |
|---|---|---|
| `Age` | Numérico | Idade do aluno (15–25 anos) |
| `Gender` | Categórico | Gênero (Female / Male) |
| `Education_Level` | Categórico | Nível escolar (School / College / University) |
| `City` | Categórico | Cidade de origem (5 cidades) |
| `AI_Tool_Used` | Categórico | Ferramenta de IA utilizada (ChatGPT, Gemini, Grammarly, Copilot, Notion AI) |
| `Daily_Usage_Hours` | Numérico | Horas diárias de uso da ferramenta de IA |
| `Purpose` | Categórico | Finalidade de uso (Homework, Learning, Research, Coding, Writing) |
| `Satisfaction_Level` | Categórico | Satisfação com a ferramenta (Low / Medium / High) |
| `Impact_on_Grades` | Categórico | **Variável alvo** (Improved / No Change / Slight Decline) |

O dataset original continha 100 instâncias. Para aumentar a confiabilidade das métricas de avaliação, foram geradas 50 instâncias sintéticas adicionais (seed `2026`) respeitando as distribuições originais de cada coluna, totalizando **150 instâncias**. Todas as proporções de classe, faixas de valores numéricos e distribuições categóricas foram preservadas dentro da variação amostral esperada.

O dataset atende aos critérios do trabalho: 8 features (≥ 5), problema de classificação multiclasse com alvo categórico, variáveis numéricas e categóricas, e não é um dataset de exemplo de aula.

---

## Pré-processamento

O mesmo pipeline de pré-processamento base foi aplicado nos três modelos, com variações onde o algoritmo exige.

### Tratamento de valores ausentes

Imputer com estratégia `median` para features numéricas e `most_frequent` para features categóricas. Essa escolha é conservadora: a mediana é robusta a outliers (ao contrário da média), e a moda preserva a distribuição original das categorias.

### Codificação de variáveis categóricas

`OneHotEncoder` com `handle_unknown="ignore"` para todas as features categóricas. A opção `handle_unknown="ignore"` garante que categorias não vistas no treino não causem erro em produção (e na previsão interativa).

### Normalização

`StandardScaler` aplicado nas features numéricas (`Age` e `Daily_Usage_Hours`) para KNN e Árvore de Decisão. O KNN é especialmente sensível à escala, pois calcula distâncias euclidianas e sem normalização, `Age` (escala 15–25) dominaria `Daily_Usage_Hours` (escala 0,5–5,9).

Para o Naive Bayes foi utilizado `KBinsDiscretizer` (3 bins, estratégia quantil) no lugar do StandardScaler, transformando as features numéricas em variáveis binárias por bin. Isso é necessário porque o `BernoulliNB` assume features binárias; discretizar os valores contínuos é a forma correta de usar esse estimador com dados numéricos.

### Divisão treino/teste

Split estratificado 80/20 (`train_test_split` com `stratify=y`, `random_state=42`), resultando em 120 amostras de treino e 30 de teste. A estratificação garante que as proporções das três classes do alvo sejam mantidas em ambos os conjuntos.

---

## Modelos

### Árvore de Decisão (`decision_tree_impact_on_grades.py`)

**Hiperparâmetros escolhidos:** `max_depth=4`, `min_samples_leaf=3`, `class_weight="balanced"`.

- `max_depth=4`: limita a profundidade para evitar overfitting em um dataset pequeno (150 instâncias). Uma árvore mais profunda memorizaria o treino sem generalizar.
- `min_samples_leaf=3`: exige pelo menos 3 amostras em cada folha, reduzindo divisões espúrias criadas por ruído.
- `class_weight="balanced"`: compensa o leve desbalanceamento entre as classes (Improved 35% / No Change 34% / Slight Decline 31%), dando maior peso às classes minoritárias durante o treino.

**Resultados:**

```
              precision    recall  f1-score   support
    Improved       0.50      0.36      0.42        11
   No Change       0.29      0.40      0.33        10
Slight Decline     0.12      0.11      0.12         9
    accuracy                           0.30        30
```

**Interpretabilidade — Importância das features:**

| Feature | Importância |
|---|---|
| Daily_Usage_Hours | 0.413 |
| Age | 0.192 |
| Purpose | 0.108 |
| Satisfaction_Level | 0.095 |
| AI_Tool_Used | 0.090 |
| Education_Level | 0.056 |
| City | 0.045 |
| Gender | 0.000 |

A árvore identifica `Daily_Usage_Hours` como a feature mais discriminante, estudantes com padrões de uso muito baixo ou muito alto tendem a ter resultados distintos. `Age` e `Purpose` aparecem como segundo e terceiro fatores relevantes. `Gender` não contribuiu em nenhuma divisão, sugerindo que o gênero não é preditor do impacto nas notas neste dataset.

A estrutura da árvore (salva em `outputs/decision_tree/04_tree_structure.txt`) e o gráfico (`05_decision_tree_plot.png`) permitem rastrear exatamente qual combinação de valores leva a cada previsão, sendo essa a principal vantagem de interpretabilidade desse modelo.

---

### Naive Bayes (`naive_bayes_impact_on_grades.py`)

**Variante escolhida:** `BernoulliNB` com `alpha=1.0` (suavização de Laplace).

O `BernoulliNB` modela a probabilidade de cada feature binária estar presente ou ausente dado cada classe. Com o pré-processamento de discretização + OHE, cada feature numérica é convertida em 3 colunas binárias (pertence ao bin 0, 1 ou 2), compatíveis com esse estimador. A suavização de Laplace (`alpha=1.0`) evita probabilidades condicionais zero para combinações não observadas no treino.

**Resultados:**

```
              precision    recall  f1-score   support
    Improved       0.40      0.36      0.38        11
   No Change       0.42      0.50      0.45        10
Slight Decline     0.25      0.22      0.24         9
    accuracy                           0.37        30
```

**Interpretabilidade — Probabilidades condicionais:**

O Naive Bayes é intrinsecamente interpretável: para cada feature binária, podemos ler diretamente a probabilidade condicional `P(feature=1 | classe)`. As features com maior `spread` (diferença entre o maior e o menor `P` entre as classes) são as que mais discriminam entre os rótulos.

As features mais discriminantes foram `Daily_Usage_Hours` (especialmente os bins de uso baixo e médio), `AI_Tool_Used` (Gemini e Grammarly), e `Education_Level`. Por exemplo:

- `Daily_Usage_Hours` bin baixo: `P=0.318` para Improved, `P=0.535` para No Change e `P=0.154` para Slight Decline — uso muito baixo está associado à estabilidade nas notas.
- `Daily_Usage_Hours` bin médio: `P=0.364` para Improved, `P=0.256` para No Change e `P=0.487` para Slight Decline — uso médio-alto aparece com mais força no Slight Decline.

As probabilidades a priori das classes ficaram equilibradas (Improved 35%, No Change 34%, Slight Decline 31%), refletindo a distribuição do dataset combinado.

---

### KNN (`knn_impact_on_grades.py`)

**Hiperparâmetros escolhidos via `GridSearchCV`:** `n_neighbors=15`, `weights="uniform"`.

Os valores de `n_neighbors` testados foram `[3, 5, 7, 10, 15, 20]` e `weights` entre `uniform` e `distance`, avaliados com validação cruzada de 5 folds no conjunto de treino. A combinação `n_neighbors=15, weights=uniform` obteve a maior acurácia média de CV (0.358). Um k maior suaviza a fronteira de decisão e reduz overfitting em datasets pequenos; `uniform` foi preferido pelo GridSearchCV indicando que dar peso igual aos 15 vizinhos mais próximos generaliza melhor que ponderar pela distância neste caso.

**Resultados:**

```
              precision    recall  f1-score   support
    Improved       0.38      0.45      0.42        11
   No Change       0.40      0.40      0.40        10
Slight Decline     0.14      0.11      0.12         9
    accuracy                           0.33        30
```

**Interpretabilidade — Permutation Importance e LIME:**

O KNN é o modelo menos interpretável dos três: não há regras explícitas nem probabilidades condicionais que expliquem a previsão. A previsão é feita por votação entre os vizinhos, sendo um modelo baseado em memória.

Para contornar isso foram utilizadas duas técnicas:

**Permutation Importance** (global): permuta aleatoriamente os valores de cada feature e mede a queda de acurácia. Valores negativos indicam que a feature não ajuda (removê-la não piora o modelo). Neste caso, quase todas as features apresentaram importância negativa ou zero, o que reflete a baixa acurácia geral do KNN neste dataset. A feature `AI_Tool_Used` foi a única com importância levemente positiva (`0.012`).

**LIME** (local): explica a previsão para uma instância específica aproximando o comportamento do KNN por um modelo linear interpretável na vizinhança daquela amostra. A explicação é salva em `outputs/knn/03_lime_explanation.html` e mostra quais features empurraram a previsão em direção a cada classe para aquele aluno específico.

A análise dos vizinhos mais próximos (`04_nearest_neighbors.csv`) complementa o LIME mostrando quais amostras do treino influenciaram a decisão.

---

## Comparação e Análise

### Desempenho

| Modelo | Acurácia |
|---|---|
| Naive Bayes | **37%** |
| Árvore de Decisão | 30% |
| KNN | 33% |

O desempenho geral é baixo nos três modelos. Isso indica que as features disponíveis têm poder preditivo limitado sobre o impacto nas notas, possivelmente porque fatores importantes como desempenho escolar anterior, disciplina e hábitos de estudo não estão no dataset. A baixa acurácia é uma limitação real do dataset, não necessariamente dos modelos.

### Os modelos concordaram nas variáveis mais relevantes?

Parcialmente. Árvore de Decisão e Naive Bayes concordaram que `Daily_Usage_Hours` é a feature mais discriminante. O KNN, por ter acurácia muito baixa, não fornece um sinal confiável de importância via permutation importance (a maioria dos valores foi negativa ou nula).

### Os resultados fizeram sentido?

Em parte. A relevância de `Daily_Usage_Hours` é intuitiva: extremos de uso (muito baixo ou muito alto) tendem a ter padrões distintos de impacto. A irrelevância de `Gender` também é coerente, não há razão a priori para que o gênero influencie o impacto de ferramentas de IA nas notas.

A associação de uso médio-alto com `Slight Decline` (identificada pelo Naive Bayes) pode indicar dependência excessiva da ferramenta, mas seria necessário um dataset maior para confirmar esse padrão.

### Limitações de cada modelo em termos de interpretabilidade

| Modelo | Limitação |
|---|---|
| **Árvore de Decisão** | Com `max_depth=4`, pode não capturar interações mais complexas. A importância nativa de features é enviesada para features de alta cardinalidade e pode ser instável em datasets pequenos. |
| **Naive Bayes** | A suposição de independência condicional entre features raramente é verdadeira na prática. O uso de `BernoulliNB` com discretização por bins perde informação sobre a ordem e magnitude dos valores numéricos. |
| **KNN** | Não tem representação interna interpretável. LIME fornece explicações locais (por instância), mas não permite afirmações globais sobre o modelo. A permutation importance foi prejudicada pela baixa acurácia. |

---

## Ferramentas de Interpretabilidade Utilizadas

| Ferramenta | Modelo | O que faz |
|---|---|---|
| **Feature Importance nativa** | Árvore de Decisão | Mede a redução média de impureza (Gini) proporcionada por cada feature em todas as divisões da árvore. |
| **Visualização da árvore** (`plot_tree`, `export_text`) | Árvore de Decisão | Mostra graficamente as regras de decisão aprendidas, permitindo rastrear qualquer previsão. |
| **Probabilidades condicionais** (`feature_log_prob_`) | Naive Bayes | Lê diretamente os parâmetros aprendidos pelo modelo: `P(feature | classe)` para cada combinação. |
| **Permutation Importance** | KNN | Técnica agnóstica de modelo: embaralha cada feature e mede a queda de performance, estimando a importância global. |
| **LIME** | KNN | Técnica agnóstica de modelo: ajusta um modelo linear simples na vizinhança de uma instância para explicar a previsão localmente. |

---

## Como executar

```bash
# Análise exploratória
python eda_impact_on_grades.py

# Modelos (cada um gera artefatos em outputs/<modelo>/)
python decision_tree_impact_on_grades.py
python naive_bayes_impact_on_grades.py
python knn_impact_on_grades.py
```

Dependências: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `lime`.
