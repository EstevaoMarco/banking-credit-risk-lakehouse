# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: account
# MAGIC
# MAGIC Limpa a tabela de contas (bronze.account) e grava em **banking_project.silver.account**.
# MAGIC
# MAGIC Transformações:
# MAGIC - **date:** texto no formato YYMMDD vira data (1993-01-01)
# MAGIC - **frequency:** códigos em tcheco traduzidos (POPLATEK MESICNE = monthly, POPLATEK TYDNE = weekly, POPLATEK PO OBRATU = after_transaction)

# COMMAND ----------

df_account = spark.table("banking_project.bronze.account")

display(df_account.limit(10))
df_account.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perfil de qualidade
# MAGIC
# MAGIC Contagem de nulos, linhas duplicadas e chaves repetidas antes de transformar.

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_account.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_account.columns
    ])
)

# COMMAND ----------

total_rows = df_account.count()
distinct_rows = df_account.distinct().count()

print("Total de linhas:", total_rows)
print("Linhas distintas:", distinct_rows)
print("Duplicadas:", total_rows - distinct_rows)

# COMMAND ----------

display(
    df_account
    .groupBy("account_id")
    .count()
    .filter(col("count") > 1)
)

# COMMAND ----------

display(
    df_account
    .groupBy("frequency")
    .count()
    .orderBy("frequency")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformações

# COMMAND ----------

from pyspark.sql.functions import col, when, substring, concat, lit, to_date

df_account_silver = (
    df_account
    .withColumn(
        "date",
        to_date(
            concat(
                lit("19"),
                substring(col("date"), 1, 2),
                substring(col("date"), 3, 2),
                substring(col("date"), 5, 2)
            ),
            "yyyyMMdd"
        )
    )
    .withColumn(
        "frequency",
        when(col("frequency") == "POPLATEK MESICNE", "monthly")
        .when(col("frequency") == "POPLATEK TYDNE", "weekly")
        .when(col("frequency") == "POPLATEK PO OBRATU", "after_transaction")
        .otherwise(col("frequency"))
    )
)

# COMMAND ----------

display(df_account_silver.limit(10))
df_account_silver.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação e conferência
# MAGIC
# MAGIC Grava em Delta e confere o resultado.

# COMMAND ----------

(
    df_account_silver.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("banking_project.silver.account")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM banking_project.silver.account
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*) 
# MAGIC FROM banking_project.silver.account;
