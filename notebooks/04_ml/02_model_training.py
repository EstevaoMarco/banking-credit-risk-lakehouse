# Databricks notebook source
from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

df_ml = (
    spark.table("banking_project.gold.loan_risk_features")
    .filter(col("default").isNotNull())
)

pdf_ml = df_ml.toPandas()

# COMMAND ----------

columns_to_drop = [
    "loan_id",
    "account_id",
    "client_id",
    "loan_date",
    "loan_status"
]

pdf_ml = pdf_ml.drop(columns=columns_to_drop)

# COMMAND ----------

X = pdf_ml.drop(columns=["default"])
y = pdf_ml["default"]

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

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