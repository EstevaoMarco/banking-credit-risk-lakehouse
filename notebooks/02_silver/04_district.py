# Databricks notebook source
df_district = spark.table("banking_project.bronze.district")

display(df_district.limit(10))
df_district.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_district.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_district.columns
    ])
)

# COMMAND ----------

total_rows = df_district.count()
distinct_rows = df_district.distinct().count()

print("Total de linhas:", total_rows)
print("Linhas distintas:", distinct_rows)
print("Duplicadas:", total_rows - distinct_rows)

# COMMAND ----------

display(
    df_district
    .groupBy("A1")
    .count()
    .filter(col("count") > 1)
)

# COMMAND ----------

df_district_silver = (
    df_district
    .withColumnRenamed("A1", "district_id")
    .withColumnRenamed("A2", "district_name")
    .withColumnRenamed("A3", "region")
    .withColumnRenamed("A4", "population")
    .withColumnRenamed("A5", "municipalities_lt_499")
    .withColumnRenamed("A6", "municipalities_500_1999")
    .withColumnRenamed("A7", "municipalities_2000_9999")
    .withColumnRenamed("A8", "municipalities_gt_10000")
    .withColumnRenamed("A9", "cities")
    .withColumnRenamed("A10", "urban_ratio")
    .withColumnRenamed("A11", "average_salary")
    .withColumnRenamed("A12", "unemployment_rate_1995")
    .withColumnRenamed("A13", "unemployment_rate_1996")
    .withColumnRenamed("A14", "entrepreneurs_per_1000")
    .withColumnRenamed("A15", "crimes_1995")
    .withColumnRenamed("A16", "crimes_1996")
)

# COMMAND ----------

display(df_district_silver.limit(10))
df_district_silver.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col

display(
    df_district_silver
    .select(
        "unemployment_rate_1995",
        "unemployment_rate_1996",
        "crimes_1995",
        "crimes_1996"
    )
    .distinct()
)

# COMMAND ----------

from pyspark.sql.functions import when, col

df_district_silver = (
    df_district_silver
    .withColumn(
        "unemployment_rate_1995",
        when(col("unemployment_rate_1995") == "?", None)
        .otherwise(col("unemployment_rate_1995"))
    )
    .withColumn(
        "crimes_1995",
        when(col("crimes_1995") == "?", None)
        .otherwise(col("crimes_1995"))
    )
)

# COMMAND ----------

df_district_silver = (
    df_district_silver
    .withColumn("population", col("population").cast("int"))
    .withColumn("municipalities_lt_499", col("municipalities_lt_499").cast("int"))
    .withColumn("municipalities_500_1999", col("municipalities_500_1999").cast("int"))
    .withColumn("municipalities_2000_9999", col("municipalities_2000_9999").cast("int"))
    .withColumn("municipalities_gt_10000", col("municipalities_gt_10000").cast("int"))
    .withColumn("cities", col("cities").cast("int"))
    .withColumn("urban_ratio", col("urban_ratio").cast("double"))
    .withColumn("average_salary", col("average_salary").cast("int"))
    .withColumn("unemployment_rate_1995", col("unemployment_rate_1995").cast("double"))
    .withColumn("unemployment_rate_1996", col("unemployment_rate_1996").cast("double"))
    .withColumn("entrepreneurs_per_1000", col("entrepreneurs_per_1000").cast("int"))
    .withColumn("crimes_1995", col("crimes_1995").cast("int"))
    .withColumn("crimes_1996", col("crimes_1996").cast("int"))
)

# COMMAND ----------

df_district_silver.printSchema()

# COMMAND ----------

from pyspark.sql.functions import sum, when, col

display(
    df_district_silver.select([
        sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df_district_silver.columns
    ])
)

# COMMAND ----------

display(
    df_district_silver
    .filter(
        col("unemployment_rate_1995").isNull() |
        col("crimes_1995").isNull()
    )
)

# COMMAND ----------

print(df_district_silver.count())

# COMMAND ----------

(
    df_district_silver.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("banking_project.silver.district")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM banking_project.silver.district
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*)
# MAGIC FROM banking_project.silver.disp;