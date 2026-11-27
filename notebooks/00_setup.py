# Databricks notebook source
# MAGIC %md
# MAGIC # Setup do ambiente
# MAGIC
# MAGIC Cria o catálogo **banking_project** no Unity Catalog e os schemas da arquitetura medallion:
# MAGIC - **bronze:** dados brutos, como vieram dos CSVs
# MAGIC - **silver:** dados limpos, tipados e traduzidos
# MAGIC - **gold:** tabela de features pronta para o modelo
# MAGIC
# MAGIC Os arquivos CSV devem ser carregados antes no Volume **banking_project.bronze.raw_files**.

# COMMAND ----------

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
