# Databricks notebook source
import sys
import os

PROJECT_PATH = '/Workspace/Repos/uriah372@campus.technion.ac.il/Follow-The-Leaders'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from utils.constants import *
from utils.functions import *
import pyspark
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec, CountVectorizer, IDF, SQLTransformer
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.window import Window
import string

spark = SparkSession.builder.appName("ClusteringBasedEvaluation").getOrCreate()
display(spark)

# COMMAND ----------

profiles = spark.read.parquet("/linkedin/people").withColumnRenamed("сourses", "courses")
search_results = spark.read.parquet(PROJECT_DATA_DIR + "/retrieval_example")

# COMMAND ----------

search_results.count()

# COMMAND ----------

target_ids = [r["id"] for r in search_results.select("id").collect()]
print(target_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC # Building Labels

# COMMAND ----------

clustered_profiles = profiles.withColumn("cluster", F.when(F.col("id").isin(target_ids), F.lit(1)).otherwise(F.lit(0)))

# COMMAND ----------

# MAGIC %md
# MAGIC # Building Features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Word2Vec

# COMMAND ----------

def build_features_pipeline(df: pyspark.sql.DataFrame, 
                            col: str, 
                            vec_size: int=100, 
                            min_count: int=5, 
                            windowSize: int=5) -> Pipeline:
    regexTokenizer = RegexTokenizer(inputCol="about", toLowercase=True)
    stopwordsRemover = StopWordsRemover(inputCol=regexTokenizer.getOutputCol(), stopWords=STOPWORDS)
    model = Word2Vec(vectorSize=vec_size,
                    inputCol=stopwordsRemover.getOutputCol(),
                    outputCol=col + "_features",
                    minCount=min_count,
                    numPartitions=df.rdd.getNumPartitions(),
                    maxIter=(df.rdd.getNumPartitions() // 5) + 1,
                    windowSize=windowSize)
    sqlTrans = SQLTransformer(statement=f"""SELECT {', '.join("'" + c + "'" for c in df.columns)}, {col}_features FROM __THIS__""")
    pipeline = Pipeline(stages=[
        regexTokenizer,
        stopwordsRemover,
        model,
        sqlTrans
    ])
    return pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using TF-IDF

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluating Clustering

# COMMAND ----------

clustered_profiles = clustered_profiles.na.fill(value="", subset="about")
features_pipeline = build_features_pipeline(df=clustered_profiles, col="about").fit(clustered_profiles)
clustered_profiles = features_pipeline.transform(clustered_profiles)

# COMMAND ----------

temp = features_pipeline.stages[2].transform(features_pipeline.stages[1].transform(features_pipeline.stages[0].transform(clustered_profiles)))
temp.display()

# COMMAND ----------

# There was a problem with some of the column names inside the statement
sqlt = features_pipeline.stages[3].setStatement(value="SELECT 'about', 'avatar', 'certifications', 'city', 'country_code', 'current_company', 'current_company:company_id', 'current_company:name', 'education', 'educations_details', 'experience', 'followers', 'following', 'groups', 'id', 'languages', 'name', 'people_also_viewed', 'position', 'posts', 'recommendations', 'recommendations_count', 'timestamp', 'url', 'volunteer_experience', 'courses', 'cluster', 'about_features' FROM __THIS__")
sqlt.transform(clustered_profiles)

# COMMAND ----------

clustered_profiles = features_pipeline.transform(clustered_profiles)

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator
clusteringEvaluator = ClusteringEvaluator(featuresCol="about_features", predictionCol="cluster").setDistanceMeasure("cosine")
silhouette = clusteringEvaluator.evaluate(clustered_profiles)
print("Silhouette Score of The Retrieval:", silhouette)

# COMMAND ----------

# MAGIC %md
# MAGIC Not the best, but not the worst either! Can be attributed to the curse of dimensionality, and IR clustering unually achieves worse results then regular classification tasks in general because the corpus itself might not be well structured for a clustering tast, but that doesn't mean that it isn't useful!

# COMMAND ----------

from pyspark.sql.window import Window
def set_random_clusters(df):
    # Add a random number column
    df_with_rand = df.withColumn("rand", F.rand())

    # Define a window spec without partitioning but ordered by the random number descendingly
    windowSpec = Window.orderBy(F.col("rand").desc())

    # Add a row number based on the random number
    df_with_row_number = df_with_rand.withColumn("row_num", F.row_number().over(windowSpec))

    # Assign labels: 1 to the top 50 rows, 0 to the others
    df_with_labels = df_with_row_number.withColumn("cluster", (F.col("row_num") <= 50).cast("int")).drop("rand", "row_num")

    # Show the result
    return df_with_labels

# COMMAND ----------

true_silhouette = silhouette
random_silhouette = []
for _ in range(10):
    random_silhouette.append(clusteringEvaluator.evaluate(set_random_clusters(clustered_profiles)))
print("True Silhouette:", true_silhouette)
print("Average Random Silhouette:", sum(random_silhouette) / len(random_silhouette))

# COMMAND ----------

print("Silhouette Improvement Ratio:", true_silhouette / (sum(random_silhouette) / len(random_silhouette)))

# COMMAND ----------

model = features_pipeline.stages[2]

# COMMAND ----------

model.findSynonymsArray(word="data", num=5)
