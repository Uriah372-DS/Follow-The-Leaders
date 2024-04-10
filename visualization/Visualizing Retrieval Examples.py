# Databricks notebook source
import sys
import os

PROJECT_PATH = '/Workspace/Repos/uriah372@campus.technion.ac.il/Follow-The-Leaders'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from utils.constants import *
from utils.functions import *
import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()
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
