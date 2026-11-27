# Databricks notebook source
from pyspark.sql.functions import col
df_ml = (

    spark.table("banking_project.gold.loan_risk_features")
    .filter(col("default").isNotNull())
)

print(df_ml.count())
display(df_ml.limit(10))

# COMMAND ----------

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

print(pdf_ml.shape)

print(pdf_ml["default"].value_counts())

# COMMAND ----------

selected_features = [
    "loan_amount",
    "loan_duration",
    "monthly_payment",
    "gender",
    "age_at_loan",
    "account_age_days",
    "frequency",

    "average_salary",
    "unemployment_rate_1995",
    "unemployment_rate_1996",
    "entrepreneurs_per_1000",
    "urban_ratio",

    # comportamento recente
    "transaction_count_30d",
    "avg_balance_30d",
    "min_balance_30d",

    # comportamento de médio prazo
    "transaction_count_90d",
    "avg_transaction_amount_90d",
    "cash_flow_90d",
    "avg_balance_90d",
    "min_balance_90d",
    "negative_balance_transactions_90d",
    "credit_debit_ratio_90d_log",

    # comportamento mais estrutural
    "transaction_count_180d",
    "avg_balance_180d",
    "min_balance_180d",

    "default"
]

pdf_model = pdf_ml[selected_features]

print(pdf_model.shape)

# COMMAND ----------

print(pdf_ml.shape)
print(pdf_ml.columns.tolist())

# COMMAND ----------

X = pdf_model.drop(columns=["default"])
y = pdf_model["default"]

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# COMMAND ----------

print(X_train.shape)
print(X_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


categorical_features = [
    "gender",
    "frequency"
]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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