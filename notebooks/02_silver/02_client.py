# Databricks notebook source
df_client = spark.table("banking_project.bronze.client")

display(df_client.limit(10))
df_client.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_client.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_client.columns
    ])
)

# COMMAND ----------

total_rows = df_client.count()
distinct_rows = df_client.distinct().count()

print("Total de linhas:", total_rows)
print("Linhas distintas:", distinct_rows)
print("Duplicadas:", total_rows - distinct_rows)

# COMMAND ----------

display(
    df_client
    .groupBy("client_id")
    .count()
    .filter(col("count") > 1)
)

# COMMAND ----------

from pyspark.sql.functions import length

display(
    df_client
    .groupBy(length(col("birth_number")).alias("birth_number_length"))
    .count()
)

# COMMAND ----------

from pyspark.sql.functions import col, substring, when, concat, lit, to_date

df_client_silver = (
    df_client
    .withColumn("birth_year", substring(col("birth_number"), 1, 2))
    .withColumn("birth_month_raw", substring(col("birth_number"), 3, 2).cast("int"))
    .withColumn("birth_day", substring(col("birth_number"), 5, 2))
    
    .withColumn(
        "gender",
        when(col("birth_month_raw") > 50, "female")
        .otherwise("male")
    )
    
    .withColumn(
        "birth_month",
        when(col("birth_month_raw") > 50, col("birth_month_raw") - 50)
        .otherwise(col("birth_month_raw"))
    )
)

# COMMAND ----------

display(
    df_client_silver.select(
        "client_id",
        "birth_number",
        "birth_year",
        "birth_month_raw",
        "birth_month",
        "birth_day",
        "gender"
    ).limit(20)
)

# COMMAND ----------

from pyspark.sql.functions import concat, lit, lpad, to_date

df_client_silver = (
    df_client_silver
    .withColumn(
        "birth_date",
        to_date(
            concat(
                lit("19"),
                col("birth_year"),
                lpad(col("birth_month"), 2, "0"),
                col("birth_day")
            ),
            "yyyyMMdd"
        )
    )
)

# COMMAND ----------

display(
    df_client_silver.select(
        "client_id",
        "birth_number",
        "birth_date",
        "gender",
        "district_id"
    ).limit(20)
)

# COMMAND ----------

df_client_silver = (
    df_client_silver
    .drop(
        "birth_number",
        "birth_year",
        "birth_month_raw",
        "birth_month",
        "birth_day"
    )
)

# COMMAND ----------

display(df_client_silver.limit(10))
df_client_silver.printSchema()

# COMMAND ----------

display(
    df_client_silver.select([
        sum(
            when(col("birth_date").isNull(), 1).otherwise(0)
        ).alias("birth_date_nulls")
    ])
)

# COMMAND ----------

display(
    df_client_silver
    .groupBy("gender")
    .count()
    .orderBy("gender")
)

# COMMAND ----------

print(df_client_silver.count())

# COMMAND ----------

(
    df_client_silver.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("banking_project.silver.client")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM banking_project.silver.client
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*)
# MAGIC FROM banking_project.silver.client;