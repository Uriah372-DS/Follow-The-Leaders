# Databricks notebook source
# MAGIC %md
# MAGIC install pyLDAvis directory for visualiztion purposes of the lda (Not for the user, for us)

# COMMAND ----------

!pip install --upgrade pip
!pip uninstall scipy
!pip install scipy==1.10.1
!pip install --upgrade gensim pyLDAvis
dbutils.library.restartPython()

# COMMAND ----------

import sys
import os

PROJECT_PATH = '/Workspace/Repos/uriah372@campus.technion.ac.il/Follow-The-Leaders'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from utils.constants import *
from utils.functions import *
import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = pyspark.sql.SparkSession.builder.getOrCreate()
display(spark)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from gensim import corpora, models
from pprint import pprint
import pandas as pd
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# COMMAND ----------

# MAGIC %md
# MAGIC read the leaders profiles file that we created in the notebook "Analayzing Scraped Leaders". It is the fusion of the leaders profiles from our search engine.
# MAGIC We concatenate about, position and education_details columns to one column (that will later on be processed to be the input of the LDA model)

# COMMAND ----------

leaders = spark.read.parquet(PROJECT_DATA_DIR + "/retrieval_example")
leaders = leaders.na.fill('', subset=['about', 'position', 'educations_details'])
leaders = leaders.withColumn('complete_document', F.concat(F.col('position'), F.lit('\n'), F.col('about'), F.lit('\n'), F.col('educations_details')))

# COMMAND ----------

display(leaders.select('complete_document'))

# COMMAND ----------

 # process text:
regexp_punct = '[_()\'\"\[\]\{\}•*^%$#@:;,.!?></|\\-&]'
leaders = leaders.withColumn(f"processed_doc0", F.regexp_replace(leaders['complete_document'], regexp_punct, ' '))
tokenizer = RegexTokenizer(inputCol=f"processed_doc0", outputCol=f"processed_doc1", toLowercase=True)
stopwordsRemover = StopWordsRemover(inputCol=f"processed_doc1",  outputCol=f"processed_doc", stopWords=STOPWORDS)

p = Pipeline(stages=[tokenizer, stopwordsRemover]).fit(leaders)

leaders = p.transform(leaders)

# COMMAND ----------

display(leaders)

# COMMAND ----------

leaders_doc_list = leaders.toPandas()['processed_doc'].tolist()

# COMMAND ----------

# Create a dictionary from  leaders_doc_list
dictionary = corpora.Dictionary(leaders_doc_list)
# Create a corpus (bag of words representation)
bag_of_words = [dictionary.doc2bow(text) for text in leaders_doc_list]
# Run LDA with 3 topics
lda_model = models.LdaModel(bag_of_words, num_topics=3, id2word=dictionary, passes=10)

# COMMAND ----------

# Print the main idea (topic) extracted by LDA
main_idea = lda_model.print_topics(num_words=25)
#clean the probaliltys of the words and keep onltly the words
topics_words = [topic[1] for topic in main_idea]
cleaned_topics = [
    [word.split('*')[1].strip().replace('"', '') for word in topic.split('+')]
    for topic in topics_words
]
pprint(cleaned_topics)

# COMMAND ----------

# MAGIC %md
# MAGIC Save the lda results for the notebook LLM Integration

# COMMAND ----------

leaders.write.mode("overwrite").parquet(PROJECT_DATA_DIR + "/retrieval_concatenated_position_about_educations_details")

directory = PROJECT_DATA_DIR + "/lda/"
file_name = "data_science_query.txt"
file_path = os.path.join(directory, file_name)

# Create the directory if it doesn't exist already
if not os.path.exists(directory):
    os.makedirs(directory)

with open(file_path, mode='w') as file:
    for sublist in cleaned_topics:
        file.write(' '.join(sublist) + '\n')

# COMMAND ----------

# MAGIC %md
# MAGIC For us - Interactive visualiztion to explore the lda topic assigment:

# COMMAND ----------

# Create the visualization
pyLDAvis.display(gensimvis.prepare(lda_model, bag_of_words, dictionary))

# COMMAND ----------

# MAGIC %md
# MAGIC # LDA on The Inconsistent Example:

# COMMAND ----------

bad_leaders = spark.read.parquet(PROJECT_DATA_DIR + "/bad_retrieval_example")
bad_leaders = bad_leaders.na.fill('', subset=['about', 'position', 'educations_details'])
bad_leaders = bad_leaders.withColumn('complete_document', F.concat(F.col('position'), F.lit('\n'), F.col('about'), F.lit('\n'), F.col('educations_details')))

# COMMAND ----------

bad_leaders = bad_leaders.withColumn(f"processed_doc0", F.regexp_replace(bad_leaders['complete_document'], regexp_punct, ' '))
tokenizer = RegexTokenizer(inputCol=f"processed_doc0", outputCol=f"processed_doc1", toLowercase=True)
stopwordsRemover = StopWordsRemover(inputCol=f"processed_doc1",  outputCol=f"processed_doc", stopWords=STOPWORDS)

p = Pipeline(stages=[tokenizer, stopwordsRemover]).fit(bad_leaders)

bad_leaders = p.transform(bad_leaders)
bad_leaders_doc_list = bad_leaders.toPandas()['processed_doc'].tolist()

# COMMAND ----------

# Create a dictionary from  bad_leaders_doc_list
dictionary = corpora.Dictionary(bad_leaders_doc_list)
# Create a corpus (bag of words representation)
bag_of_words = [dictionary.doc2bow(text) for text in bad_leaders_doc_list]
# Run LDA with 3 topics
lda_model = models.LdaModel(bag_of_words, num_topics=3, id2word=dictionary, passes=10)

# COMMAND ----------

# Print the main idea (topic) extracted by LDA
main_idea = lda_model.print_topics(num_words=25)
#clean the probaliltys of the words and keep onltly the words
topics_words = [topic[1] for topic in main_idea]
cleaned_topics = [
    [word.split('*')[1].strip().replace('"', '') for word in topic.split('+')]
    for topic in topics_words
]
pprint(cleaned_topics)

# COMMAND ----------

# Create the visualization
pyLDAvis.display(gensimvis.prepare(lda_model, bag_of_words, dictionary))

# COMMAND ----------

bad_leaders.write.mode("overwrite").parquet(PROJECT_DATA_DIR + "/bad_retrieval_concatenated_position_about_educations_details")

directory = PROJECT_DATA_DIR + "/lda/"
file_name = "bad_query.txt"
file_path = os.path.join(directory, file_name)

# Create the directory if it doesn't exist already
if not os.path.exists(directory):
    os.makedirs(directory)

with open(file_path, mode='w') as file:
    for sublist in cleaned_topics:
        file.write(' '.join(sublist) + '\n')
