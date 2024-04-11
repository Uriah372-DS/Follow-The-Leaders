# Databricks notebook source
# MAGIC %md
# MAGIC #LLM Integration - Using Gemini to build career path and timeline for users
# MAGIC

# COMMAND ----------

!pip install --upgrade pip
!pip install --upgrade --quiet langchain-google-genai pillow
!pip install langchain

# COMMAND ----------

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

# MAGIC %md
# MAGIC Loading Gemini from Langchain

# COMMAND ----------

from langchain_google_genai import ChatGoogleGenerativeAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Good Example Query

# COMMAND ----------

q_field='data'
q_position='machine learning engineer'
q_companies='google amazon meta'
q_industry ='high tech'

# COMMAND ----------

corpus = spark.read.parquet(PROJECT_DATA_DIR + "/retrieval_concatenated_position_about_educations_details").limit(50)

# COMMAND ----------

# MAGIC %md
# MAGIC Getting context from retrieved information

# COMMAND ----------

query_context_list = corpus.select(F.collect_list('complete_document').alias('complete_document')).collect()[0]['complete_document']

# COMMAND ----------

# MAGIC %md
# MAGIC Getting LDA analysis results

# COMMAND ----------

read_cleaned_topics = []
with open(os.path.join(PROJECT_DATA_DIR+"/lda/", "data_science_query.txt"), mode='r') as file:
    for line in file:
        # Split each line into a list of words
        words = line.strip().split()
        read_cleaned_topics.append(words)

relevant_topics = read_cleaned_topics[0]
relevant_string = ', '.join(relevant_topics)
print(relevant_string)


# COMMAND ----------

# MAGIC %md
# MAGIC Setting up Gemini

# COMMAND ----------

LLM = 'gemini-pro'
TEMP = 0.1
API_KEY = 'AIzaSyCILJLSwNywi3qY7njwvE04whGyH6zwstY'

lang_model = ChatGoogleGenerativeAI(model=LLM, temperature=TEMP, google_api_key=API_KEY, convert_system_message_to_human=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Raw query attempt

# COMMAND ----------

import re
def remove_special_chars(text):
    # Remove \n
    text = re.sub(r'\\n', '', text)
    # Remove emojis
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
    return text

# COMMAND ----------

n_steps = 7
context = remove_special_chars(' '.join(query_context_list))
query = f'Field of Interest - {q_field}, Desired Position - {q_position}, Desired Companies - {q_companies}, Desired Industry - {q_industry}'
v2_prompt = f" are tasked with building a career path for an entry-level linkedin user who is interested in the field of {q_field} and wants to be a {q_position}. We add self-descriptions of linkedin profiles which are relevant to the query-submitter's goals here, use infromation from this in your output - {context}, Output should be a list of up to {n_steps} steps for the user to take, each is explained fully and contains concrete examples."
res = remove_special_chars(lang_model.invoke(v2_prompt).content)
print(res)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain of Thoughts (CoT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summarization Task
# MAGIC

# COMMAND ----------

summ_prompt = f'You are given self-descriptions of linkedin users, including their job titles and where they got their education. Summarize these documents, identify recurrent patterns and elaborate on them. Explain commonality between education details and commonality between jobs. Try to focus about topics from here: {relevant_string} The documents: {context}'
summ_context = lang_model.invoke(summ_prompt)
print(summ_context.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Prompt

# COMMAND ----------

# The Query:
q_field='data'
q_position='machine learning engineer'
q_companies='google amazon meta'
q_industry ='high tech'

# Read relevant topics from LDA:
read_cleaned_topics = []
with open(os.path.join(PROJECT_DATA_DIR+"/lda/", "data_science_query.txt"), mode='r') as file:
    for line in file:
        # Split each line into a list of words
        words = line.strip().split()
        read_cleaned_topics.append(words)

relevant_topics = read_cleaned_topics[0]
relevant_string = ', '.join(relevant_topics)

# Read context from search results:
leaders = spark.read.parquet(PROJECT_DATA_DIR + "/retrieval_concatenated_position_about_educations_details").limit(50)
query_context_list = leaders.select(F.collect_list('complete_document').alias('complete_document')).collect()[0]['complete_document']
context = remove_special_chars(' '.join(query_context_list))

# Use LLM to Summarize the context:
summ_prompt = f'You are given self-descriptions of linkedin users, including their job titles and where they got their education. Summarize these documents, identify recurrent patterns and elaborate on them. Explain commonality between education details and commonality between jobs. Try to focus about topics from here: {relevant_string} The documents: {context}'
summ_context = lang_model.invoke(summ_prompt)

# Call LLM to write a career path based on the summary:
prompt = f"You are tasked with building a career path for a linkedin user in the field of {q_field} who wants to be a {q_position} at a company named or focused around {q_companies} and at the industry of {q_industry}. You are given information about linkedin profiles that match the user's goal. Use the information to build the career path, recommend concrete and practical actions that appear in the information and elaborate as much as you can. Give fluent explanetions. Output should conatin {n_steps} steps and under each step give 3 additional recommendations or examples based on information about linkedin profiles. information: {summ_context.content}"
res = lang_model.invoke(prompt)
print(res.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inconsistent Example Query:

# COMMAND ----------

# The Query:
q_field = 'healthcare'
q_position = 'lumberjack'
q_companies = 'construction'
q_industry = 'software development'

# Read relevant topics from LDA:
read_cleaned_topics = []
with open(os.path.join(PROJECT_DATA_DIR+"/lda/", "bad_query.txt"), mode='r') as file:
    for line in file:
        # Split each line into a list of words
        words = line.strip().split()
        read_cleaned_topics.append(words)

relevant_topics = read_cleaned_topics[0]
relevant_string = ', '.join(relevant_topics)

# Read context from search results:
leaders = spark.read.parquet(PROJECT_DATA_DIR + "/bad_retrieval_concatenated_position_about_educations_details").limit(50)
query_context_list = leaders.select(F.collect_list('complete_document').alias('complete_document')).collect()[0]['complete_document']
context = remove_special_chars(' '.join(query_context_list))

# Use LLM to Summarize the context:
summ_prompt = f'You are given self-descriptions of linkedin users, including their job titles and where they got their education. Summarize these documents, identify recurrent patterns and elaborate on them. Explain commonality between education details and commonality between jobs. Try to focus about topics from here: {relevant_string} The documents: {context}'
summ_context = lang_model.invoke(summ_prompt)

# Call LLM to write a career path based on the summary:
prompt = f"You are tasked with building a career path for a linkedin user in the field of {q_field} who wants to be a {q_position} at a company named or focused around {q_companies} and at the industry of {q_industry}. You are given information about linkedin profiles that match the user's goal. Use the information to build the career path, recommend concrete and practical actions that appear in the information and elaborate as much as you can. Give fluent explanetions. Output should conatin {n_steps} steps and under each step give 3 additional recommendations or examples based on information about linkedin profiles. information: {summ_context.content}"
res = lang_model.invoke(prompt)
print(res.content)
