# Databricks notebook source
display(dbutils.fs.ls("/Volumes/banking_project/bronze/raw_files/"))

# COMMAND ----------

df_account = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load("/Volumes/banking_project/bronze/raw_files/account.csv"))

display(df_account)

# COMMAND ----------

df_account.printSchema()

# COMMAND ----------

display(df_account)
df_account.printSchema()

# COMMAND ----------

files = [
    "account",
    "client",
    "disp",
    "district",
    "loan",
    "trans"]

base_path = "/Volumes/banking_project/bronze/raw_files"

for file_name in files:
    print(f"\n===== {file_name.upper()} =====")

    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("sep", ";")
        .load(f"{base_path}/{file_name}.csv"))

    print("Schema:")
    df.printSchema()

    print("Quantidade de linhas:")
    print(df.count())

    print("Amostra:")
    display(df.limit(5))

# COMMAND ----------

base_path = "/Volumes/banking_project/bronze/raw_files"

df_account = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load(f"{base_path}/account.csv")
)

df_client = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load(f"{base_path}/client.csv")
)

df_disp = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load(f"{base_path}/disp.csv")
)

df_district = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load(f"{base_path}/district.csv")
)

df_loan = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load(f"{base_path}/loan.csv")
)

df_trans = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load(f"{base_path}/trans.csv")
)

# COMMAND ----------

display(
    df_loan
    .groupBy("status")
    .count()
    .orderBy("status")
)

# COMMAND ----------

display(df_trans.select("type").distinct())

display(df_trans.select("operation").distinct())

display(df_trans.select("k_symbol").distinct())

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

display(
    df_trans
    .groupBy("type", "operation")
    .count()
    .orderBy("type", "operation")
)

# COMMAND ----------

display(
    df_trans
    .groupBy("type", "operation", "k_symbol")
    .count()
    .orderBy("type", "operation", "k_symbol")
)

# COMMAND ----------

files = [
    "account",
    "client",
    "disp",
    "district",
    "loan",
    "trans"
]

base_path = "/Volumes/banking_project/bronze/raw_files"

for file_name in files:

    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("sep", ";")
        .load(f"{base_path}/{file_name}.csv")
    )

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(f"banking_project.bronze.{file_name}")
    )

    print(f"{file_name} loaded successfully")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN banking_project.bronze;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM banking_project.bronze.trans;