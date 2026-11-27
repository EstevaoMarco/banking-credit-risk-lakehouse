# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # ML 03: avaliação do modelo
# MAGIC
# MAGIC Executa o notebook de treino e avalia o modelo com métricas no teste, validação cruzada e curvas ROC e Precision-Recall.

# COMMAND ----------

# MAGIC %run ./02_model_training

# COMMAND ----------

# MAGIC %md
# MAGIC ## Previsões no conjunto de teste

# COMMAND ----------

y_pred = model_pipeline.predict(X_test)

y_prob = model_pipeline.predict_proba(X_test)[:, 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Métricas no teste
# MAGIC
# MAGIC Com poucas linhas de teste (47, sendo 6 inadimplentes), essas métricas variam muito conforme o split.

# COMMAND ----------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação cruzada
# MAGIC
# MAGIC StratifiedKFold com 5 folds. Como a amostra é pequena, a média e o desvio da validação cruzada são a métrica principal do projeto.

# COMMAND ----------

from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

cv_results = cross_validate(
    model_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring
)

# COMMAND ----------

import pandas as pd

cv_summary = pd.DataFrame({
    "accuracy": cv_results["test_accuracy"],
    "precision": cv_results["test_precision"],
    "recall": cv_results["test_recall"],
    "f1": cv_results["test_f1"],
    "roc_auc": cv_results["test_roc_auc"]
})

display(cv_summary)

# COMMAND ----------

display(
    cv_summary.agg(["mean", "std"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Curva ROC

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)

roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Curva Precision-Recall
# MAGIC
# MAGIC Mais informativa que a ROC quando a classe positiva é rara.

# COMMAND ----------

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds_pr = precision_recall_curve(
    y_test,
    y_prob
)

average_precision = average_precision_score(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))
plt.plot(
    recall,
    precision,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()