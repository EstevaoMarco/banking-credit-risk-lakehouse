# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: trans
# MAGIC
# MAGIC Limpa a tabela de transações (bronze.trans, cerca de 1 milhão de linhas) e grava em **banking_project.silver.trans**.
# MAGIC
# MAGIC Transformações:
# MAGIC - **date:** YYMMDD vira data, e **amount** e **balance** viram double
# MAGIC - **type:** PRIJEM = credit, VYDAJ = debit, VYBER = withdrawal
# MAGIC - **operation** e **k_symbol:** códigos em tcheco traduzidos para inglês
# MAGIC - k_symbol vazio vira nulo

# COMMAND ----------

df_trans = spark.table("banking_project.bronze.trans")

display(df_trans.limit(10))
df_trans.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perfil de qualidade
# MAGIC
# MAGIC Contagem de nulos, linhas duplicadas e chaves repetidas antes de transformar.

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_trans.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_trans.columns
    ])
)

# COMMAND ----------

total_rows = df_trans.count()
distinct_rows = df_trans.distinct().count()

print("Total de linhas:", total_rows)
print("Linhas distintas:", distinct_rows)
print("Duplicadas:", total_rows - distinct_rows)

# COMMAND ----------

display(
    df_trans
    .groupBy("trans_id")
    .count()
    .filter(col("count") > 1)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformações

# COMMAND ----------

from pyspark.sql.functions import col, substring, concat, lit, to_date

df_trans_silver = (
    df_trans
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
    .withColumn("balance", col("balance").cast("double"))
)

# COMMAND ----------

display(df_trans_silver.limit(10))
df_trans_silver.printSchema()

# COMMAND ----------

from pyspark.sql.functions import when, col

df_trans_silver = (
    df_trans_silver
    .withColumn(
        "type",
        when(col("type") == "PRIJEM", "credit")
        .when(col("type") == "VYDAJ", "debit")
        .when(col("type") == "VYBER", "withdrawal")
        .otherwise(col("type"))
    )
)

# COMMAND ----------

display(
    df_trans_silver
    .groupBy("type")
    .count()
    .orderBy("type")
)

# COMMAND ----------

df_trans_silver = (
    df_trans_silver
    .withColumn(
        "operation",
        when(col("operation") == "VYBER KARTOU", "card_withdrawal")
        .when(col("operation") == "PREVOD NA UCET", "outgoing_transfer")
        .when(col("operation") == "VYBER", "cash_withdrawal")
        .when(col("operation") == "PREVOD Z UCTU", "incoming_transfer")
        .when(col("operation") == "VKLAD", "cash_deposit")
        .otherwise(col("operation"))
    )
)

# COMMAND ----------

display(
    df_trans_silver
    .groupBy("operation")
    .count()
    .orderBy("operation")
)

# COMMAND ----------

df_trans_silver = (
    df_trans_silver
    .withColumn(
        "k_symbol",
        when(col("k_symbol") == "SIPO", "household_payment")
        .when(col("k_symbol") == "SLUZBY", "services")
        .when(col("k_symbol") == "POJISTNE", "insurance")
        .when(col("k_symbol") == "SANKC. UROK", "penalty_interest")
        .when(col("k_symbol") == "UROK", "interest")
        .when(col("k_symbol") == "DUCHOD", "pension")
        .when(col("k_symbol") == "UVER", "loan_payment")
        .otherwise(col("k_symbol"))
    )
)

# COMMAND ----------

display(
    df_trans_silver
    .groupBy("k_symbol")
    .count()
    .orderBy("k_symbol")
)

# COMMAND ----------

from pyspark.sql.functions import col, length, trim

display(
    df_trans_silver
    .select(
        "k_symbol",
        length(col("k_symbol")).alias("length"),
        length(trim(col("k_symbol"))).alias("trimmed_length")
    )
    .distinct()
    .orderBy("length")
)

# COMMAND ----------

from pyspark.sql.functions import when, trim, col

df_trans_silver = (
    df_trans_silver
    .withColumn(
        "k_symbol",
        when(trim(col("k_symbol")) == "", None)
        .otherwise(col("k_symbol"))
    )
)

# COMMAND ----------

display(
    df_trans_silver
    .groupBy("k_symbol")
    .count()
    .orderBy("k_symbol")
)

# COMMAND ----------

display(
    df_trans_silver.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_trans_silver.columns
    ])
)

# COMMAND ----------

print(df_trans_silver.count())

# COMMAND ----------

display(
    df_trans_silver
    .groupBy("operation")
    .agg(
        sum(when(col("bank").isNull(), 1).otherwise(0)).alias("bank_nulls"),
        sum(when(col("account").isNull(), 1).otherwise(0)).alias("account_nulls")
    )
    .orderBy("operation")
)

# COMMAND ----------

display(
    df_trans_silver
    .filter(
        (col("operation") == "outgoing_transfer") &
        (col("bank").isNull() | col("account").isNull())
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação e conferência
# MAGIC
# MAGIC Grava em Delta e confere o resultado.

# COMMAND ----------

(
    df_trans_silver.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("banking_project.silver.trans")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM banking_project.silver.trans
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*)
# MAGIC FROM banking_project.silver.trans;
