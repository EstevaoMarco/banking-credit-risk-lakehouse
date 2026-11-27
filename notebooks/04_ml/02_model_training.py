# Databricks notebook source
# MAGIC %md
# MAGIC # ML 02: treino do modelo
# MAGIC
# MAGIC Treina uma regressão logística para estimar a probabilidade de inadimplência de um empréstimo. Este notebook é executado pelos notebooks 03 e 04 via %run.

# COMMAND ----------

from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dados de treino
# MAGIC
# MAGIC Executa o notebook 01_data_preparation, que entrega **pdf_model**: empréstimos com alvo definido (A = 0, B = 1) e só as features selecionadas.

# COMMAND ----------

# MAGIC %run ./01_data_preparation

# COMMAND ----------

X = pdf_model.drop(columns=["default"])
y = pdf_model["default"]

print(X.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Divisão entre treino e teste
# MAGIC
# MAGIC 80% treino e 20% teste, estratificado pelo alvo, porque a classe de inadimplentes é pequena.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pré-processamento
# MAGIC
# MAGIC - **Numéricas:** imputação pela mediana e padronização
# MAGIC - **Categóricas (gender, frequency):** imputação pela moda e one-hot encoding
# MAGIC
# MAGIC Tudo fica dentro do Pipeline, então o pré-processamento é ajustado só no treino e não vaza informação do teste.

# COMMAND ----------

categorical_features = [
    "gender",
    "frequency"
]

numeric_features = [
    c for c in X.columns
    if c not in categorical_features
]

# COMMAND ----------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# COMMAND ----------

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo
# MAGIC
# MAGIC Regressão logística com **class_weight="balanced"**, para compensar o desbalanceamento entre pagos e não pagos.

# COMMAND ----------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# COMMAND ----------

model_pipeline.fit(X_train, y_train)

print("Modelo treinado com sucesso.")
