# Databricks notebook source
# MAGIC %md
# MAGIC # Read LinkedIn Data

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

profiles = spark.read.parquet('/linkedin/people')
profiles = profiles.filter(profiles['current_company.company_id'].isNull() | profiles['current_company.link'].contains(profiles['current_company.company_id']))  # removes 6 bugged rows

# COMMAND ----------

companies = spark.read.parquet('/linkedin/companies')
companies = companies.withColumn('meta_industry', 
                                 F.udf(lambda x: None if x is None else META_INDUSTRIES_DICT[x], 
                                       returnType=StringType())(companies['industries']))

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieving Top Companies

# COMMAND ----------

# MAGIC %md
# MAGIC First we clean the data while doing some EDA to prepare for the large scale scraping that we're about to do.

# COMMAND ----------

# We want to sort by 'employees_in_linkedin' so we first need to check it's validity and origin
print(companies.filter(F.size(companies['employees']) != companies['employees_in_linkedin']).count())
print(companies.filter(companies['employees'].isNotNull()).count())
print(companies.filter(companies['employees_in_linkedin'].isNotNull()).count())
print(companies.filter(companies['employees_in_linkedin'].isNull() & companies['employees'].isNotNull()).count())
print(companies.filter(companies['employees_in_linkedin'].isNotNull() & companies['employees'].isNull()).count())

# COMMAND ----------

# Fix empty rows by filling with the length of the array of employees
companies = companies.withColumn('employees_in_linkedin', 
                                 F.when(companies['employees_in_linkedin'].isNull() & companies['employees'].isNotNull(), 
                                 F.size(companies['employees'])).otherwise(companies['employees_in_linkedin']))

# Check that it worked:
print(companies.filter(F.size(companies['employees']) != companies['employees_in_linkedin']).count())
print(companies.filter(companies['employees'].isNotNull()).count())
print(companies.filter(companies['employees_in_linkedin'].isNotNull()).count())
print(companies.filter(companies['employees_in_linkedin'].isNull() & companies['employees'].isNotNull()).count())
print(companies.filter(companies['employees_in_linkedin'].isNotNull() & companies['employees'].isNull()).count())

# COMMAND ----------

# MAGIC %md
# MAGIC We need the urls and ids of companies to look for the relevant employees in them, but we can see a connection between them as well:

# COMMAND ----------

companies.filter(~companies['url'].contains(companies['id'])).count()

# COMMAND ----------

# MAGIC %md
# MAGIC The company id is taken directly from its page url. This means that we can simply save the id and disregard the urls because they are all built the same.

# COMMAND ----------

# Also notice that the 'url' is a column without nulls, and so is 'id'
print(companies.filter(companies['url'].isNull()).count() == 0)
print(companies.filter(companies['id'].isNull()).count() == 0)

# COMMAND ----------

# MAGIC %md
# MAGIC We also noticed that there are 2 types of urls in the data: company and school urls

# COMMAND ----------

company_url_preffix = 'https://www.linkedin.com/company/'
school_url_preffix = 'https://www.linkedin.com/school/'

companies.filter(~(companies['url'].contains(company_url_preffix) | companies['url'].contains(school_url_preffix))).count()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can retrieve the top n companies with the most employees in each meta industry:

# COMMAND ----------

def get_top_companies(topn: int, recursive=True):
    window = Window.partitionBy(F.col('meta_industry')).orderBy(F.col('employees_in_linkedin').desc())
    ret = companies.select(F.rank().over(window).alias('rank'), '*').filter(F.col('rank') <= topn).orderBy('rank', F.col('employees_in_linkedin').desc()).collect()
    return [r.asDict(recursive) for r in ret]

# COMMAND ----------

# MAGIC %md
# MAGIC # Searching For Leaders

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the top companies, we need to find the leaders in each company

# COMMAND ----------

profiles.filter(profiles['current_company:company_id'].isNotNull()).count() / profiles.count()

# COMMAND ----------

# Notice that 'current_company.title' is identical to 'position'
profiles.filter(profiles['current_company.title'] == profiles['position']).count() == profiles.filter(profiles['current_company.title'].isNotNull()).count()

# COMMAND ----------

companies_to_scrape[0]

# COMMAND ----------

target_terms = ['chairman', 
                'ceo', 'chief executive officer', 
                'coo', 'chief operating officer', 'cto', 'chief technology officer', 'cfo', 'chief financial officer', 'chro', 'chief human resources officer', 
                'president', 
                'vp', 'vice president', 
                'director', 
                'manager']

companies_to_scrape = get_top_companies(topn=5)

leaders_to_scrape = profiles.filter(profiles['current_company.company_id'].isNotNull() & 
                                    (profiles['current_company:company_id'].isin([company['id'] for company in companies_to_scrape])) & 
                                    F.lower(profiles['position']).rlike("|".join(["(" + pat + ")" for pat in target_terms])))

print('Total Profiles:', leaders_to_scrape.count())
leaders_to_scrape.groupBy('current_company.name').count().orderBy(F.col('count').desc()).display()

# COMMAND ----------

leaders_to_scrape.write.mode('overwrite').parquet(TARGET_USERS)
