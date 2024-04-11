# Databricks notebook source
import sys
import os

# Get the current notebook path and subtract the string from it to get the project path
path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
PROJECT_PATH = path[:path.find('Follow-The-Leaders')] + 'Follow-The-Leaders/'

# If it's not in sys.path then add it
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# and now we can import from the utils directory inside the repo:
from utils.constants import *

import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = pyspark.sql.SparkSession.builder.getOrCreate()
display(spark)

# COMMAND ----------

good_example = spark.read.parquet(PROJECT_DATA_DIR + "/retrieval_concatenated_position_about_educations_details")
bad_example = spark.read.parquet(PROJECT_DATA_DIR + "/bad_retrieval_concatenated_position_about_educations_details")

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizations of The Good Retrieval Example

# COMMAND ----------

display(good_example)

# COMMAND ----------

display(good_example.select(F.explode("processed_doc").alias("words")).groupBy("words").count().orderBy(F.col("count").desc()).limit(10))

# COMMAND ----------

display(good_example.select(F.explode("processed_doc").alias("words")).groupBy("words").count().orderBy(F.col("count").desc()).limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizations of The Inconsistent Retrieval Example

# COMMAND ----------

display(bad_example)

# COMMAND ----------

display(bad_example.select(F.explode("processed_doc").alias("words")).groupBy("words").count().orderBy(F.col("count").desc()).limit(10))

# COMMAND ----------

display(bad_example.select(F.explode("processed_doc").alias("words")).groupBy("words").count().orderBy(F.col("count").desc()).limit(100))
