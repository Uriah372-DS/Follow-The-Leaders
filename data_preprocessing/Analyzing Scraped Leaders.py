# Databricks notebook source
import sys
import os

PROJECT_PATH = '/Workspace/Repos/uriah372@campus.technion.ac.il/Follow-The-Leaders'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import numpy as np
import pyspark
from typing import Union, Callable
from utils.constants import *
from utils.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = pyspark.sql.SparkSession.builder.getOrCreate()
display(spark)

# COMMAND ----------

profiles = spark.read.parquet('/linkedin/people')
profiles = profiles.select(*sorted([c for c in profiles.columns]))
profiles.display()

scraped = spark.read.parquet(LOCAL_DATA + "/scraped_leaders")
scraped = scraped.select(*sorted([c for c in scraped.columns]))
scraped.display()

# COMMAND ----------

def get_new_columns(original_df, new_df):
    new_columns = new_df.columns
    for c in original_df.columns:
        if c in new_columns:
            new_columns.remove(c)
    return new_columns
new_columns = get_new_columns(profiles, scraped)
unscraped_columns = get_new_columns(scraped, profiles)
print(new_columns)
print(unscraped_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC There are additional columns in the scraped data, but also ones that are not named correctly and other that can be parsed into corresponding columns in the original data.

# COMMAND ----------

# Not named correctly:
# scraped = scraped.withColumnRenamed("current_company_company_id", "current_company:company_id")
# scraped = scraped.withColumnRenamed("current_company_name", "current_company:name")

# Can be parsed into corresponding columns:
# scraped = scraped.withColumn("url", F.col("input.url")).drop("input")

new_columns = get_new_columns(profiles, scraped)
unscraped_columns = get_new_columns(scraped, profiles)
print(new_columns)
print(unscraped_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also see that some of the schema wasn't inferred correctly. we can fix that because the data is the same data, It's just a format issue.

# COMMAND ----------

common_columns = []
for c in [*scraped.columns, *profiles.columns]:
    if c in scraped.columns and c in profiles.columns and c not in common_columns:
        common_columns.append(c)
common_columns

# COMMAND ----------

true_schema = StructType([
    StructField('about', StringType(), True), 
    StructField('avatar', StringType(), True), 
    StructField('canonical_url', StringType(), True), 
    StructField("certifications", ArrayType(
                StructType([
                        StructField("meta", StringType(), True),
                        StructField("subtitle", StringType(), True),
                        StructField("title", StringType(), True),
                    ]), True,), True,), 
    StructField('city', StringType(), True), 
    StructField('connections', IntegerType(), True), 
    StructField('country_code', StringType(), True), 
    StructField("current_company",
            StructType([
                    StructField("company_id", StringType(), True),
                    StructField("industry", StringType(), True),
                    StructField("link", StringType(), True),
                    StructField("name", StringType(), True),
                    StructField("title", StringType(), True),
                ]), True,), 
    StructField('current_company:company_id', StringType(), True), 
    StructField('current_company:name', StringType(), True), 
    StructField("education", ArrayType(
                StructType([
                        StructField("degree", StringType(), True),
                        StructField("end_year", StringType(), True),
                        StructField("field", StringType(), True),
                        StructField("meta", StringType(), True),
                        StructField("start_year", StringType(), True),
                        StructField("title", StringType(), True),
                        StructField("url", StringType(), True),
                    ]), True,), True,), 
    StructField('educations_details', StringType(), True), 
    StructField('error', StringType(), True), 
    StructField("experience", ArrayType(
                StructType([
                        StructField("company", StringType(), True),
                        StructField("company_id", StringType(), True),
                        StructField("description", StringType(), True),
                        StructField("duration", StringType(), True),
                        StructField("duration_short", StringType(), True),
                        StructField("end_date", StringType(), True),
                        StructField("location", StringType(), True),
                        StructField("positions", ArrayType(
                                StructType([
                                        StructField("description", StringType(), True),
                                        StructField("duration", StringType(), True),
                                        StructField("duration_short", StringType(), True),
                                        StructField("end_date", StringType(), True),
                                        StructField("meta", StringType(), True),
                                        StructField("start_date", StringType(), True),
                                        StructField("subtitle", StringType(), True),
                                        StructField("title", StringType(), True),
                                    ]), True,), True,),
                        StructField("start_date", StringType(), True),
                        StructField("subtitle", StringType(), True),
                        StructField("subtitleURL", StringType(), True),
                        StructField("title", StringType(), True),
                        StructField("url", StringType(), True),
                    ]), True,), True,), 
    StructField('followers', IntegerType(), True), 
    StructField('id', StringType(), True), 
    StructField('locale', StringType(), True), 
    StructField('locations', ArrayType(StringType(), True), True), 
    StructField('name', StringType(), True), 
    StructField("people_also_viewed", ArrayType(
                StructType([
                    StructField("profile_link", StringType(), True)
                    ]), True), True,),
    StructField('position', StringType(), True), 
    StructField('region', StringType(), True), 
    StructField('url', StringType(), True)
])

# COMMAND ----------

# scraped = spark.read.json(path="file:" + LOCAL_DATA + "/scraped_leaders.json", multiLine=True, schema=true_schema)
# scraped = spark.write.mode("overwrite").parquet(LOCAL_DATA + "/scraped_leaders")
scraped = spark.read.parquet(LOCAL_DATA + "/scraped_leaders")
scraped.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Some of the text is not fully html-parsed, we need to do that to get the same format as our original data.

# COMMAND ----------

!pip install --upgrade pip
!pip install html2text bs4

# COMMAND ----------

from bs4 import BeautifulSoup
import html2text

# Create an html2text handler
handler = html2text.HTML2Text()
handler.ul_item_mark = '•'

@F.udf(returnType=BooleanType())
def contains_html(column_value):
    if column_value is None:
        return column_value
    # Parse the string using BeautifulSoup
    soup = BeautifulSoup(column_value, 'html.parser')
    
    # Find all tags in the parsed string
    tags = [tag.name for tag in soup.find_all()]
    
    # BeautifulSoup might add <html>, <body>, and <p> tags automatically if it thinks
    # the string is HTML but lacks these tags. So, we check for the presence of other tags.
    # You might adjust the list of ignored tags based on your requirements.
    ignored_tags = {'html', 'body', 'p'}
    
    # Check if there are any tags other than the ignored ones
    has_tags = any(tag not in ignored_tags for tag in tags)
    return has_tags

@F.udf(returnType=StringType())
def format_text(text):
    if text is not None:
        return handler.handle(text).rstrip('\n')
    return text

# COMMAND ----------

# MAGIC %md
# MAGIC Now we need to parse all columns that might have html tags in them.

# COMMAND ----------

from pyspark.sql.types import _parse_datatype_string

BRACKETS = {'(': ')', '[': ']', '{': '}', '<': '>'}
def ignore_brackets_split(s, separator=','):
    """
        Copied from latest version of pyspark.sql.types
        Splits the given string by given separator, but ignore separators inside brackets pairs, e.g.
        given "a,b" and separator ",", it will return ["a", "b"], but given "a<b,c>, d", it will return
        ["a<b,c>", "d"].
    """
    parts = []
    buf = ""
    level = 0
    for c in s:
        if c in BRACKETS.keys():
            level += 1
            buf += c
        elif c in BRACKETS.values():
            if level == 0:
                raise ValueError("Brackets are not correctly paired: %s" % s)
            level -= 1
            buf += c
        elif c == separator and level > 0:
            buf += c
        elif c == separator:
            parts.append(buf)
            buf = ""
        else:
            buf += c

    if len(buf) == 0:
        raise ValueError("The %s cannot be the last char: %s" % (separator, s))
    parts.append(buf)
    return parts

@F.udf(returnType=StringType())
def parse_text(text):
    if not contains_html.func(text):
        return text
    else:
        return format_text.func(text)

def udf_supporting_transform(df: pyspark.sql.DataFrame,
                             col: Union[str, pyspark.sql.Column], 
                             f: Union[Callable[[pyspark.sql.Column], pyspark.sql.Column], 
                                      Callable[[pyspark.sql.Column, pyspark.sql.Column], pyspark.sql.Column]],
                             element_type: str) -> pyspark.sql.Column:
    """
     In a typically annoying fashion, pyspark doesn't support activating a UDF on elements of an array column inside the transform function, 
     So for our use-case we created this function to bypass this restriction. 
     It almost bypasses it completely, but it doesn't preserve null values in the array. It just removes them together with the padding.
    """
    temp_column = str((hash(str(col)) + int(np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max))))
    max_length = df.select(F.max(F.size(col))).collect()[0][0]

    element_type = _parse_datatype_string(element_type)
    @F.udf(returnType=ArrayType(elementType=element_type))
    def pad_array(arr, length, pad_value):
        if arr is None:
            return [pad_value] * length
        return arr + [pad_value] * (length - len(arr))

    return F.array_except(F.array(*[f(F.element_at(pad_array(col, F.lit(max_length), F.lit(None)), i)) for i in range(1, max_length + 1)]), 
                          F.array(F.lit(None)))

def parse_string_column(column_name: Union[str, pyspark.sql.Column]) -> pyspark.sql.Column:
    if isinstance(column_name, str):
        column_expr = F.col(column_name)
    else:
        column_expr = column_name
    return F.when(contains_html(column_expr) == False, column_expr).otherwise(format_text(column_expr))

def parse_array_column(df: pyspark.sql.DataFrame, column_name: Union[str, pyspark.sql.Column], column_type: str) -> pyspark.sql.DataFrame:
    if isinstance(column_name, str):
        column_expr = F.col(column_name)
    else:
        column_expr = column_name
    
    element_type = column_type[6:-1]  # [6:-1] strips ("array<") and ">"
    
    if element_type.startswith('string'):
        return udf_supporting_transform(df=df, col=column_expr, f=lambda x: parse_string_column(column_name=x), element_type=element_type)
    
    if element_type.startswith('struct'):
        return udf_supporting_transform(df=df, col=column_expr, f=lambda x: parse_struct_column(df=df, column_name=x, column_type=element_type), element_type=element_type)
    
    if element_type.startswith('array'):
        return udf_supporting_transform(df=df, col=column_expr, f=lambda x: parse_array_column(df=df, column_name=x, column_type=element_type), element_type=element_type)
    
    return column_expr

def parse_struct_column(df: pyspark.sql.DataFrame, column_name: Union[str, pyspark.sql.Column], column_type: str) -> pyspark.sql.DataFrame:
    if isinstance(column_name, str):
        column_expr = F.col(column_name)
    else:
        column_expr = column_name
    
    fields = {}
    for part in ignore_brackets_split(column_type[7:-1], ","):  # [7:-1] strips ("struct<") and ">"
        field_name, field_type = ignore_brackets_split(part, ":")
        fields[field_name] = field_type
    
    for field_name, field_type in fields.items():
        if field_type.startswith('string'):
            fields[field_name] = parse_string_column(column_name=column_expr[field_name])
        elif field_type.startswith('struct'):
            fields[field_name] = parse_struct_column(df=df, column_name=column_expr[field_name], column_type=field_type)
        elif field_type.startswith('array'):
            fields[field_name] = parse_array_column(df=df, column_name=column_expr[field_name], column_type=field_type)
        else:
            fields[field_name] = F.col(field_name)
    
    for field_name in fields.keys():
        column_expr = column_expr.withField(fieldName=field_name, col=fields[field_name])
    
    return column_expr

def parse_columns(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
     This function parses ALL of the string columns in the dataframe (yes, even the nested fields) and converts them into pythonic strings without any tags.
     It does this in a mutually recursive fashion and was an absolute headache to debug, but it is also very clever and extendable, and can work with any arbitrary UDF that you wish to activate on every string column in the dataframe.
     I took inspiration from the pyspark.sql.functions.fromJson function When building this function, which if you look inside the pyspark source code you'll find that is also a mutually recursive function, and it does so by having a function for each data-type and having them call each-other recursively.
    """
    ddf = df.select("*")  # copy df
    for column_name, column_type in df.dtypes:
        if column_type.startswith('string'):
            ddf: pyspark.sql.DataFrame = ddf.withColumn(column_name, parse_string_column(column_name=column_name))
        elif column_type.startswith('array'):
            ddf: pyspark.sql.DataFrame = ddf.withColumn(column_name, parse_array_column(df=df, column_name=column_name, column_type=column_type))
        elif column_type.startswith('struct'):
            ddf: pyspark.sql.DataFrame = ddf.withColumn(column_name, parse_struct_column(df=df, column_name=column_name, column_type=column_type))
    return ddf

# COMMAND ----------

# MAGIC %md
# MAGIC That was a difficult function to build... Let's move on to evaluating the scraping quality

# COMMAND ----------

scraped = parse_columns(df=scraped)
scraped.display()

# COMMAND ----------

leaders_ids = [r.id for r in scraped.select("id").collect()]
original_leaders = profiles.filter(profiles["id"].isin(leaders_ids))

# COMMAND ----------

comparisons = original_leaders.join(scraped.select("id", *[F.col(c).alias(c + "_scraped") for c in scraped.columns if c != "id"]), on="id")
display(comparisons)

# COMMAND ----------

# compare useful string columns
condition = F.col("position") != F.col("position_scraped")
for c in ["current_company:company_id", "current_company:name", "about"]:
    if c in ["position"]:
        continue
    condition = condition | (F.col(c).isNotNull() & (F.col(c) != F.col(c + "_scraped")))
perc_modified = round(100 * comparisons.where(condition).count() / comparisons.count(), 2)
print("Percentage of Modified String Columns:", str(perc_modified) + '%')

# COMMAND ----------

# compare useful array columns
condition = F.col("certifications.meta") != F.col("certifications_scraped.meta")
for c, c2 in [
              ("certifications.subtitle", "certifications_scraped.subtitle"), 
              ("certifications.title", "certifications_scraped.title"), 
              ("education.degree", "education_scraped.degree"), 
              ("education.end_year", "education_scraped.end_year"), 
              ("education.field", "education_scraped.field"), 
              ("education.meta", "education_scraped.meta"), 
              ("education.start_year", "education_scraped.start_year"), 
              ("education.url", "education_scraped.url"), 
              ("experience.company", "experience_scraped.company"), 
              ("experience.company_id", "experience_scraped.company_id"), 
              ("experience.start_date", "experience_scraped.start_date"), 
              ("experience.end_date", "experience_scraped.end_date"), 
              ("experience.duration", "experience_scraped.duration"), 
              ("experience.duration_short", "experience_scraped.duration_short")]:
    if c in ["certifications.meta"]:
        continue
    condition = condition | ((F.size(F.col(c)) > 0) & (F.size(F.array_except(F.col(c2), F.col(c))) > 0))
perc_modified = round(100 * comparisons.where(condition).count() / comparisons.count(), 2)
print("Percentage of Modified Array Columns:", str(perc_modified) + '%')
