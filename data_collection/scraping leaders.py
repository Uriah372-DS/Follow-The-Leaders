# Databricks notebook source
!pip install bs4


# COMMAND ----------

import sys
import os
import bs4 

PROJECT_PATH = '/Workspace/Repos/uriah372@campus.technion.ac.il/Follow-The-Leaders'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from utils.constants import *
from utils.functions import *

import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import pandas as pd
import numpy as np
import requests
import html
import re
import time
import pickle
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

spark = pyspark.sql.SparkSession.builder.getOrCreate()
display(spark)

# COMMAND ----------

!pip install --upgrade pip
!pip install bs4 chardet cchardet html5lib lxml

# COMMAND ----------

leaders_pd = pd.read_parquet(LOCAL_DATA + '/scraped_data.parquet')

# COMMAND ----------

leaders_pd

# COMMAND ----------

dict_list = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
for i in range(len(dict_list)):
    with open(LOCAL_DATA + f'/saved_responses_{i}.pkl', 'rb') as handle:
        dict_list[i] = pickle.load(handle)
responses = {k: dict_list[i][k] for i in range(len(dict_list)) for k in dict_list[i].keys()}

# COMMAND ----------

def get_response(url):
    response = responses[url.split('/in/')[1]]
    return response

def scrape_linkedin_profile(url) -> BeautifulSoup:
    response = get_response(url=url)
    return BeautifulSoup(response, features='html.parser', store_line_numbers=False)

# COMMAND ----------

some_url = leaders_pd.at[0, 'url']
test = scrape_linkedin_profile(url=some_url)
print('original_encoding:', test.original_encoding)
print('contains_replacement_characters (if true then data was lost in encoding):', test.contains_replacement_characters)
print(test)

# COMMAND ----------

def extract_about_section(soup: BeautifulSoup, df: pd.DataFrame, row_index: int):
    # Find the meta tag that contains the description of the profile
    meta_tag = soup.find('meta', attrs={'name': 'description'})

    if meta_tag and meta_tag.has_attr('content'):
        about_section = meta_tag['content']

        # Since the content contains HTML entities like "<br>", we want to remove these
        about_section = about_section.replace('<br>', '\n')
        if about_section.find(' | Learn more about') != -1:
            about_section = about_section.replace(about_section[about_section.find(' | Learn more about'):], '')
        about_section = re.sub(r'<[^>]+>', '', about_section)

        # The extracted About section text
        scraped_about = about_section
    else:
        scraped_about = None
    
    if scraped_about is not None and len(scraped_about) > len(df.at[row_index, 'about']):
        return scraped_about, True
    return df.at[row_index, 'about'], False

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the first profile, we can see that the about section that we retrieved contains much more text than the corresponding row in our data!
# MAGIC
# MAGIC We can focus on scraping these kinds of rows to enrich our data!

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming `leaders_pd` is your DataFrame and it has a column 'url' containing the URLs to scrape
# Also assuming the functions `scrape_linkedin_profile` and `extract_about_section` are defined elsewhere

def scrape_url(df, row_index, url, extractions: dict):
    soup = scrape_linkedin_profile(url=url)
    scraped = {}
    is_edited = []
    for col, extraction_fn in extractions.items():
        scraped[col], is_new_value = extraction_fn(soup=soup, df=df, row_index=row_index)
        is_edited.append(is_new_value)
    return row_index, scraped, any(is_edited)

def update_df(df, results):
    df['modified'] = [False] * len(df)
    for row_index, scraped, is_modified in results:
        # Mark whether the row was modified during scraping:
        df.at[row_index, 'modified'] = is_modified
        # Set values of the row in each scraped column:
        for col, res in scraped.items():
            df.at[row_index, col] = res
    return df

def parallel_scrape(df: pd.DataFrame, extractions: dict):
    with ThreadPoolExecutor() as executor:
        # Submit all scraping tasks and add them to the list of futures
        futures = {executor.submit(scrape_url, df, row_index, row['url'], extractions): row_index for row_index, row in df.iterrows()}
        results = []
        
        # Wrap as_completed with tqdm for a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Profiles"):
            row_index = futures[future]
            result = future.result()
            results.append(result)
    
    # Update the DataFrame with the results
    return update_df(df, results)

# Run the parallel scraping process. If we want to scrape other parts of the profiles, we just need to create a function that receives the same inputs as 'extract_about_section' and returns the value that will be replacing the original value in the relevant column
leaders_pd = pd.read_parquet(LOCAL_DATA + '/scraped_data.parquet')
leaders_pd = parallel_scrape(leaders_pd, extractions={'about': extract_about_section,})

# COMMAND ----------

leaders_pd.at[0, 'about']

# COMMAND ----------

leaders_pd['modified'].value_counts()

# COMMAND ----------

import json
with open(LOCAL_DATA + '/scraped_data.parquet', 'wb') as file:
    for c in ['experience']:
        leaders_pd[c] = leaders_pd[c].apply(json.dumps)
    leaders_pd.to_parquet(path=file)

# COMMAND ----------

leaders_pd = pd.read_parquet(LOCAL_DATA + '/scraped_data.parquet')
for c in ['experience']:
    leaders_pd[c] = leaders_pd[c].apply(json.loads)
leaders_pd
