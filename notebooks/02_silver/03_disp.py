# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: disp
# MAGIC
# MAGIC Limpa a tabela de disposições (bronze.disp), que liga clientes a contas, e grava em **banking_project.silver.disp**.
# MAGIC
# MAGIC Transformação: **type** em minúsculas (owner ou disponent). Na gold usamos só o owner.

# COMMAND ----------

df_disp = spark.table("banking_project.bronze.disp")

display(df_disp.limit(10))
df_disp.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perfil de qualidade
# MAGIC
# MAGIC Contagem de nulos, linhas duplicadas e chaves repetidas antes de transformar.

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_disp.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_disp.columns
    ])
)

# COMMAND ----------

total_rows = df_disp.count()
distinct_rows = df_disp.distinct().count()

print("Total de linhas:", total_rows)
print("Linhas distintas:", distinct_rows)
print("Duplicadas:", total_rows - distinct_rows)

# COMMAND ----------

display(
    df_disp
    .groupBy("disp_id")
    .count()
    .filter(col("count") > 1)
)

# COMMAND ----------

display(
    df_disp
    .groupBy("type")
    .count()
    .orderBy("type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformações

# COMMAND ----------

from pyspark.sql.functions import col, lower

df_disp_silver = (
    df_disp
    .withColumn("type", lower(col("type")))
)

# COMMAND ----------

display(df_disp_silver.limit(10))
df_disp_silver.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação e conferência
# MAGIC
# MAGIC Grava em Delta e confere o resultado.

# COMMAND ----------

(
    df_disp_silver.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("banking_project.silver.disp")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM banking_project.silver.disp
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*)
# MAGIC FROM banking_project.silver.disp;
