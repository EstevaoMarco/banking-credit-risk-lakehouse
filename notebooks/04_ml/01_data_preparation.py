# Databricks notebook source
# MAGIC %md
# MAGIC # ML 01: preparação dos dados
# MAGIC
# MAGIC Lê a tabela **gold.loan_risk_features** e mantém só os empréstimos com alvo definido (status A e B, contratos encerrados). Os contratos em andamento (C e D) não têm desfecho final e são pontuados no notebook 04_score_active_loans.
# MAGIC
# MAGIC O resultado é o DataFrame **pdf_model**, usado pelo notebook 02_model_training via %run. O split, o pré-processamento e o treino ficam no 02.

# COMMAND ----------

from pyspark.sql.functions import col
df_ml = (

    spark.table("banking_project.gold.loan_risk_features")
    .filter(col("default").isNotNull())
)

print(df_ml.count())
display(df_ml.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conversão para pandas
# MAGIC
# MAGIC A amostra rotulada é pequena (234 linhas), por isso o trabalho segue em pandas e scikit-learn.

# COMMAND ----------

pdf_ml = df_ml.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remoção de identificadores
# MAGIC
# MAGIC IDs, data do empréstimo e status não entram como features. O status já está representado no alvo default.

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

# MAGIC %md
# MAGIC ## Seleção de features
# MAGIC
# MAGIC Subconjunto de features escolhido por grupo: dados do empréstimo e do cliente, contexto do distrito e comportamento da conta em 30, 90 e 180 dias.

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
