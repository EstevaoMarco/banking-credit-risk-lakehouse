# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # ML 04: pontuação dos contratos em andamento (C e D)
# MAGIC
# MAGIC O modelo é treinado só com contratos encerrados (A e B), que têm desfecho conhecido. Aqui ele pontua os contratos ainda em andamento:
# MAGIC - **C:** em andamento, em dia
# MAGIC - **D:** em andamento, em atraso
# MAGIC
# MAGIC Esses contratos não têm rótulo final, mas o status D é um bom sinal de risco. Se o modelo faz sentido, **o score médio de D deve ser maior que o de C**. Essa é uma validação de sanidade fora da amostra de treino.

# COMMAND ----------

# MAGIC %run ./02_model_training

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contratos em andamento

# COMMAND ----------

from pyspark.sql.functions import col

df_active = (
    spark.table("banking_project.gold.loan_risk_features")
    .filter(col("loan_status").isin("C", "D"))
)

pdf_active = df_active.toPandas()

print(pdf_active.shape)
display(pdf_active.groupby("loan_status").size().reset_index(name="count"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pontuação
# MAGIC
# MAGIC São usadas as mesmas colunas do treino (X_train.columns) e o pipeline já ajustado.

# COMMAND ----------

X_active = pdf_active[X_train.columns]

pdf_active["default_score"] = model_pipeline.predict_proba(X_active)[:, 1]
pdf_active["default_pred"] = model_pipeline.predict(X_active)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação: score de D contra C

# COMMAND ----------

display(
    pdf_active
    .groupby("loan_status")["default_score"]
    .describe()
    .reset_index()
)

# COMMAND ----------

display(
    pdf_active
    .groupby("loan_status")["default_pred"]
    .mean()
    .reset_index(name="share_predicted_default")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC-AUC usando D como alvo aproximado
# MAGIC
# MAGIC Mede o quanto o score separa os contratos em atraso (D) dos em dia (C). Acima de 0,5 indica que o modelo ordena o risco na direção certa.

# COMMAND ----------

from sklearn.metrics import roc_auc_score

y_proxy = (pdf_active["loan_status"] == "D").astype(int)

print("ROC-AUC (D vs C):", roc_auc_score(y_proxy, pdf_active["default_score"]))

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

for status in ["C", "D"]:
    plt.hist(
        pdf_active.loc[pdf_active["loan_status"] == status, "default_score"],
        bins=30,
        alpha=0.6,
        density=True,
        label=f"Status {status}"
    )

plt.xlabel("Score de inadimplência")
plt.ylabel("Densidade")
plt.title("Distribuição do score: C (em dia) x D (em atraso)")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação dos scores

# COMMAND ----------

df_scores = spark.createDataFrame(
    pdf_active[["loan_id", "loan_status", "default_score", "default_pred"]]
)

(
    df_scores.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("banking_project.gold.active_loan_scores")
)