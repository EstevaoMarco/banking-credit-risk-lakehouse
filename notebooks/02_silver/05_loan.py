# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: loan
# MAGIC
# MAGIC Limpa a tabela de empréstimos (bronze.loan) e grava em **banking_project.silver.loan**.
# MAGIC
# MAGIC Transformações:
# MAGIC - **date:** YYMMDD vira data
# MAGIC - **amount, duration, payments:** cast para tipos numéricos
# MAGIC - **status** é mantido como está (A, B, C, D) e vira o alvo na gold

# COMMAND ----------

df_loan = spark.table("banking_project.bronze.loan")

display(df_loan.limit(10))
df_loan.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perfil de qualidade
# MAGIC
# MAGIC Contagem de nulos, linhas duplicadas e chaves repetidas antes de transformar.

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_loan.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_loan.columns
    ])
)

# COMMAND ----------

total_rows = df_loan.count()
distinct_rows = df_loan.distinct().count()

print("Total de linhas:", total_rows)
print("Linhas distintas:", distinct_rows)
print("Duplicadas:", total_rows - distinct_rows)

# COMMAND ----------

display(
    df_loan
    .groupBy("loan_id")
    .count()
    .filter(col("count") > 1)
)

# COMMAND ----------

display(
    df_loan
    .groupBy("status")
    .count()
    .orderBy("status")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformações

# COMMAND ----------

from pyspark.sql.functions import col, substring, concat, lit, to_date

df_loan_silver = (
    df_loan
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
    .withColumn("amount", col("amount").cast("double"))
    .withColumn("duration", col("duration").cast("int"))
    .withColumn("payments", col("payments").cast("double"))
)

# COMMAND ----------

display(df_loan_silver.limit(10))
df_loan_silver.printSchema()

# COMMAND ----------

from pyspark.sql.functions import sum, when, col

display(
    df_loan_silver.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_loan_silver.columns
    ])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação e conferência
# MAGIC
# MAGIC Grava em Delta e confere o resultado.

# COMMAND ----------

(
    df_loan_silver.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("banking_project.silver.loan")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM banking_project.silver.loan
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*)
# MAGIC FROM banking_project.silver.loan;
