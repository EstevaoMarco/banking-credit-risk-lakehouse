# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS banking_project;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS banking_project.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS banking_project.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS banking_project.gold;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW SCHEMAS IN banking_project;