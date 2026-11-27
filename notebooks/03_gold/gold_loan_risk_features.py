# Databricks notebook source
df_loan = spark.table("banking_project.silver.loan")
df_account = spark.table("banking_project.silver.account")
df_disp = spark.table("banking_project.silver.disp")
df_client = spark.table("banking_project.silver.client")
df_district = spark.table("banking_project.silver.district")

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

df_owner = (
    df_disp
    .filter(col("type") == "owner")
)

# COMMAND ----------

display(
    df_owner
    .groupBy("account_id")
    .count()
    .filter(col("count") != 1)
)

# COMMAND ----------

df_gold_base = (
    df_loan.alias("l")
    .join(
        df_account.alias("a"),
        col("l.account_id") == col("a.account_id"),
        "left"
    )
    .join(
        df_owner.alias("d"),
        col("l.account_id") == col("d.account_id"),
        "left"
    )
    .join(
        df_client.alias("c"),
        col("d.client_id") == col("c.client_id"),
        "left"
    )
)

# COMMAND ----------

print("Loans:", df_loan.count())
print("Gold base:", df_gold_base.count())

# COMMAND ----------

df_gold_base = (
    df_gold_base
    .select(
        col("l.loan_id").alias("loan_id"),
        col("l.account_id").alias("account_id"),
        col("d.client_id").alias("client_id"),

        col("l.date").alias("loan_date"),
        col("l.amount").alias("loan_amount"),
        col("l.duration").alias("loan_duration"),
        col("l.payments").alias("monthly_payment"),
        col("l.status").alias("loan_status"),

        col("c.gender").alias("gender"),
        col("c.birth_date").alias("birth_date"),

        col("a.date").alias("account_open_date"),
        col("a.frequency").alias("frequency"),
        col("a.district_id").alias("district_id")
    )
)

# COMMAND ----------

df_gold_base = (
    df_gold_base
    .join(
        df_district,
        on="district_id",
        how="left"
    )
)

# COMMAND ----------

print(df_gold_base.count())

# COMMAND ----------

from pyspark.sql.functions import months_between, floor, datediff

df_gold_base = (
    df_gold_base
    .withColumn(
        "age_at_loan",
        floor(months_between(col("loan_date"), col("birth_date")) / 12)
    )
    .withColumn(
        "account_age_days",
        datediff(col("loan_date"), col("account_open_date"))
    )
)

# COMMAND ----------

display(
    df_gold_base.select(
        "loan_id",
        "loan_date",
        "birth_date",
        "age_at_loan",
        "account_open_date",
        "account_age_days"
    ).limit(20)
)

# COMMAND ----------

display(
    df_gold_base.select(
        "age_at_loan",
        "account_age_days"
    ).summary()
)

# COMMAND ----------

display(
    df_gold_base
    .filter(col("age_at_loan") < 18)
    .select(
        "loan_id",
        "account_id",
        "client_id",
        "birth_date",
        "loan_date",
        "age_at_loan"
    )
    .orderBy("age_at_loan")
)

# COMMAND ----------

display(
    df_gold_base
    .filter(col("age_at_loan") < 18)
    .select(
        "loan_id",
        "account_id",
        "client_id",
        "birth_date",
        "loan_date",
        "age_at_loan",
        "loan_status"
    )
    .orderBy("age_at_loan")
)

# COMMAND ----------

display(
    df_gold_base
    .groupBy("age_at_loan")
    .count()
    .orderBy("age_at_loan")
)

# COMMAND ----------

from pyspark.sql.functions import (
    col, date_sub, count, avg, sum, min,
    stddev_samp, when
)

df_trans = spark.table("banking_project.silver.trans")


def create_transaction_features(days):

    df_loan_trans = (
        df_gold_base.alias("g")
        .join(
            df_trans.alias("t"),
            (
                (col("g.account_id") == col("t.account_id")) &
                (col("t.date") >= date_sub(col("g.loan_date"), days)) &
                (col("t.date") < col("g.loan_date"))
            ),
            "left"
        )
    )

    df_features = (
        df_loan_trans
        .groupBy(col("g.loan_id").alias("loan_id"))
        .agg(
            count(col("t.trans_id"))
            .alias(f"transaction_count_{days}d"),

            avg(col("t.amount"))
            .alias(f"avg_transaction_amount_{days}d"),

            sum(
                when(col("t.type") == "credit", col("t.amount"))
                .otherwise(0)
            ).alias(f"total_credit_{days}d"),

            sum(
                when(col("t.type") == "debit", col("t.amount"))
                .otherwise(0)
            ).alias(f"total_debit_{days}d"),

            avg(col("t.balance"))
            .alias(f"avg_balance_{days}d"),

            min(col("t.balance"))
            .alias(f"min_balance_{days}d"),

            stddev_samp(col("t.balance"))
            .alias(f"balance_stddev_{days}d")
        )
    )

    return df_features

# COMMAND ----------

from pyspark.sql.functions import (
    col, date_sub, count, avg, sum, min,
    stddev_samp, when
)

df_trans = spark.table("banking_project.silver.trans")


def create_transaction_features(days):

    df_loan_trans = (
        df_gold_base.alias("g")
        .join(
            df_trans.alias("t"),
            (
                (col("g.account_id") == col("t.account_id")) &
                (col("t.date") >= date_sub(col("g.loan_date"), days)) &
                (col("t.date") < col("g.loan_date"))
            ),
            "left"
        )
    )

    df_features = (
        df_loan_trans
        .groupBy(col("g.loan_id").alias("loan_id"))
        .agg(
            count(col("t.trans_id"))
            .alias(f"transaction_count_{days}d"),

            avg(col("t.amount"))
            .alias(f"avg_transaction_amount_{days}d"),

            sum(
                when(col("t.type") == "credit", col("t.amount"))
                .otherwise(0)
            ).alias(f"total_credit_{days}d"),

            sum(
                when(col("t.type") == "debit", col("t.amount"))
                .otherwise(0)
            ).alias(f"total_debit_{days}d"),

            avg(col("t.balance"))
            .alias(f"avg_balance_{days}d"),

            min(col("t.balance"))
            .alias(f"min_balance_{days}d"),

            sum(
                when(col("t.balance") < 0, 1)
                .otherwise(0)
            ).alias(f"negative_balance_transactions_{days}d"),

            stddev_samp(col("t.balance"))
            .alias(f"balance_stddev_{days}d")
        )
    )

    return df_features

# COMMAND ----------

df_trans_features_30d = create_transaction_features(30)
df_trans_features_90d = create_transaction_features(90)
df_trans_features_180d = create_transaction_features(180)

# COMMAND ----------

df_gold = (
    df_gold_base
    .join(
        df_trans_features_30d,
        on="loan_id",
        how="left"
    )
    .join(
        df_trans_features_90d,
        on="loan_id",
        how="left"
    )
    .join(
        df_trans_features_180d,
        on="loan_id",
        how="left"
    )
)

# COMMAND ----------

print(df_gold.count())

# COMMAND ----------

df_gold = (
    df_gold

    .withColumn(
        "credit_debit_ratio_90d",
        when(
            col("total_debit_90d") != 0,
            col("total_credit_90d") / col("total_debit_90d")
        )
    )

    .withColumn(
        "negative_balance_flag_90d",
        when(col("min_balance_90d") < 0, 1).otherwise(0)
    )
)

# COMMAND ----------

display(
    df_gold.select(
        "loan_id",
        "transaction_count_30d",
        "transaction_count_90d",
        "transaction_count_180d",
        "negative_balance_transactions_30d",
        "negative_balance_transactions_90d",
        "negative_balance_transactions_180d",
        "credit_debit_ratio_90d",
        "negative_balance_flag_90d"
    )
    .limit(10)
)

# COMMAND ----------

display(
    df_gold.select(
        "transaction_count_30d",
        "transaction_count_90d",
        "transaction_count_180d",
        "negative_balance_transactions_30d",
        "negative_balance_transactions_90d",
        "negative_balance_transactions_180d",
        "credit_debit_ratio_90d"
    ).describe()
)

# COMMAND ----------

display(
    df_gold
    .filter(col("credit_debit_ratio_90d").isNull())
    .select(
        "loan_id",
        "total_credit_90d",
        "total_debit_90d",
        "credit_debit_ratio_90d"
    )
)

# COMMAND ----------

display(
    df_gold
    .select(
        "loan_id",
        "total_credit_90d",
        "total_debit_90d",
        "credit_debit_ratio_90d"
    )
    .orderBy(col("credit_debit_ratio_90d").desc())
    .limit(10)
)

# COMMAND ----------

df_gold = (
    df_gold

    .withColumn(
        "credit_debit_ratio_90d",
        when(
            col("total_debit_90d") != 0,
            col("total_credit_90d") / col("total_debit_90d")
        )
    )

    .withColumn(
        "negative_balance_flag_90d",
        when(col("min_balance_90d") < 0, 1).otherwise(0)
    )
)

# COMMAND ----------

from pyspark.sql.functions import when, col

df_gold = (
    df_gold
    .withColumn(
        "default",
        when(col("loan_status") == "A", 0)
        .when(col("loan_status") == "B", 1)
        .otherwise(None)
    )
)

# COMMAND ----------

display(
    df_gold
    .groupBy("loan_status", "default")
    .count()
    .orderBy("loan_status")
)

# COMMAND ----------

df_ml = (
    df_gold
    .filter(col("default").isNotNull())
)

# COMMAND ----------

print(df_ml.count())

# COMMAND ----------

display(
    df_ml
    .groupBy("default")
    .count()
    .orderBy("default")
)

# COMMAND ----------

from pyspark.sql.functions import coalesce, lit, col

df_gold = (
    df_gold
    .withColumn(
        "balance_stddev_90d",
        coalesce(col("balance_stddev_90d"), lit(0.0))
    )
)

# COMMAND ----------

from pyspark.sql.functions import log1p

df_gold = (
    df_gold
    .withColumn(
        "credit_debit_ratio_90d_log",
        when(
            col("credit_debit_ratio_90d").isNotNull(),
            log1p(col("credit_debit_ratio_90d"))
        )
    )
)

# COMMAND ----------

df_gold = (
    df_gold
    .withColumn(
        "cash_flow_30d",
        col("total_credit_30d") - col("total_debit_30d")
    )
    .withColumn(
        "cash_flow_90d",
        col("total_credit_90d") - col("total_debit_90d")
    )
    .withColumn(
        "cash_flow_180d",
        col("total_credit_180d") - col("total_debit_180d")
    )
)

# COMMAND ----------

df_gold_final = df_gold.select(
    "loan_id",
    "account_id",
    "client_id",
    "loan_date",
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

    # 30 dias
    "transaction_count_30d",
    "avg_transaction_amount_30d",
    "total_credit_30d",
    "total_debit_30d",
    "cash_flow_30d",
    "avg_balance_30d",
    "min_balance_30d",
    "negative_balance_transactions_30d",
    "balance_stddev_30d",

    # 90 dias
    "transaction_count_90d",
    "avg_transaction_amount_90d",
    "total_credit_90d",
    "total_debit_90d",
    "cash_flow_90d",
    "avg_balance_90d",
    "min_balance_90d",
    "negative_balance_transactions_90d",
    "balance_stddev_90d",

    # Features derivadas 90 dias
    "credit_debit_ratio_90d",
    "credit_debit_ratio_90d_log",
    "negative_balance_flag_90d",

    # 180 dias
    "transaction_count_180d",
    "avg_transaction_amount_180d",
    "total_credit_180d",
    "total_debit_180d",
    "cash_flow_180d",
    "avg_balance_180d",
    "min_balance_180d",
    "negative_balance_transactions_180d",
    "balance_stddev_180d",

    "loan_status",
    "default"
)

# COMMAND ----------

print(df_gold_final.count())
df_gold_final.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_gold_final.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_gold_final.columns
    ])
)

# COMMAND ----------

print(df_gold_final.count())
print(len(df_gold_final.columns))

# COMMAND ----------

(
    df_gold_final.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("banking_project.gold.loan_risk_features")
)

# COMMAND ----------

spark.table("banking_project.gold.loan_risk_features").count()

# COMMAND ----------

len(
    spark.table("banking_project.gold.loan_risk_features").columns
)