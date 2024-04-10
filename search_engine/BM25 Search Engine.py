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

# MAGIC %md
# MAGIC Brainstorm:
# MAGIC 1. As input we get a meta industry - filter data based on this to kick out evey profile not with this meta industry.
# MAGIC 2. Tokenize + Word embedding for degree in education + features about rank (bachelor's, masters...)
# MAGIC 3. TF IDF to about section (Assuming it will be filled), then in the input we feed keywords about 
# MAGIC 4. word embedding and analyzing position column.

# COMMAND ----------

# MAGIC %md
# MAGIC Field Features:
# MAGIC * What fields do you want to work in (example: Data Sceicnce, Algorithms, Electrical Engineering,...)
# MAGIC * What academic aspirations you have (Bachelor, Master, PHD, what field, universty,...)
# MAGIC * What positions do you want to hold (Software Engineer, Data Sceinetist,...)
# MAGIC * What companies do you wish to work for (Google, Amazon, Meta,...)
# MAGIC
# MAGIC Leader Features:
# MAGIC *  Number of followers
# MAGIC * Length of 'experience' column
# MAGIC * Current Company (If the company is 'good')
# MAGIC * Level of education (Has an advenced degree)

# COMMAND ----------

# MAGIC %md
# MAGIC *Important - MAKE INPUT SIMPLE AND EXPRESSIVE*

# COMMAND ----------

# MAGIC %md
# MAGIC Pipeline:
# MAGIC 1. Filter data based on meta-industry input
# MAGIC 2. Use TF-IDF to score users based on their relevance (input from user - term, data - document)
# MAGIC   alternative: Use embeddings to analyize similarity between textual data.
# MAGIC 3. filter based on leader features (take those with ) 

# COMMAND ----------

# MAGIC %md
# MAGIC # BM25 Retrieval

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, Normalizer
import pyspark.sql.functions as F
import pyspark.ml.functions as mlF
from pyspark.ml.feature import CountVectorizer

# COMMAND ----------

profiles = spark.read.parquet('/linkedin/people')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Indexing:

# COMMAND ----------

from typing import Union, Tuple
from pyspark import keyword_only
from pyspark.ml import Pipeline
from sparkml_base_classes import TransformerBaseClass
from pyspark.ml.feature import StopWordsRemover
import re

class IndexingPipelineModel(TransformerBaseClass):
    @keyword_only
    def __init__(self, pipeline=None):
        super().__init__()
    
    def getPipeline(self):
        return self._pipeline
    
    def transformQuery(self, query: str):
        """ Process the query for retrieval on the corresponding indexed corpus. """
        schema = StructType([StructField(documents_column, StringType(), False)])
        queryDF = spark.createDataFrame([(query,)], schema=schema)
        
        return self.transform(queryDF).head()

    def _transform(self, ddf: pyspark.sql.DataFrame):
        self._logger.info("IndexingPipeline transform")
        ddf = ddf.fillna(value="", subset=documents_column)

        ddf = ddf.withColumn(f"processed_{documents_column}_0", F.regexp_replace(ddf[documents_column], regexp_punct, ' '))

        ddf = self._pipeline.transform(ddf)

        ddf = ddf.withColumn(f"{documents_column}_len", F.size(f"processed_{documents_column}"))
        ddf = ddf.withMetadata(f"{documents_column}_len", {"avg_doc_len": ddf.agg({f"{documents_column}_len": "mean"}).collect()[0][f"avg({documents_column}_len)"]})
        
        ddf = ddf.drop(f"processed_{documents_column}_0", 
                        f"processed_{documents_column}_1")
        return ddf

def index_corpus(raw_corpus: pyspark.sql.DataFrame, documents_column: Union[str, pyspark.sql.Column]) -> Tuple[pyspark.sql.DataFrame, IndexingPipelineModel]:
    """
        Adds the following columns to the dataframe:

        text: Original text of the document.
        text_len: Length of the text document (also stores the average document length in its metadata).
        processed_text: Tokenized, lower-cased, punctuation and stopwords removed document text.
        text_tf_vector: vector-type column storing the document tf array.
        text_tfidf_vector: vector-type column storing the document tfidf array.

        replacing the word 'text' in each column name with the name of documents_column.

        Also returns a Pipeline object for parsing a query to get the same tfidf representation as the document.
    """
    if not isinstance(documents_column, str):
        corpus = raw_corpus.withColumn(str(documents_column), documents_column)
        documents_column = str(documents_column)
        corpus = raw_corpus.fillna(value="", subset=documents_column)
    else:
        corpus = raw_corpus.fillna(value="", subset=documents_column)
    
    # process text:
    regexp_punct = '[_()\'\"\[\]\{\}•*^%$#@:;,.!?></|\\-]'
    corpus = corpus.withColumn(f"processed_{documents_column}_0", F.regexp_replace(corpus[documents_column], regexp_punct, ' '))
    tokenizer = RegexTokenizer(inputCol=f"processed_{documents_column}_0", outputCol=f"processed_{documents_column}_1", toLowercase=True)
    stopwordsRemover = StopWordsRemover(inputCol=f"processed_{documents_column}_1",  outputCol=f"processed_{documents_column}", stopWords=STOPWORDS)

    # create tf vector:
    distinct_tokens = stopwordsRemover.transform(tokenizer.transform(corpus)).select(F.explode(F.col(f"processed_{documents_column}"))).distinct().count()
    tf = CountVectorizer(inputCol=f"processed_{documents_column}", 
                         outputCol=f"{documents_column}_tf_vector", 
                         vocabSize=distinct_tokens
                         )

    # create tfidf vector:
    idf = IDF(inputCol=f"{documents_column}_tf_vector", outputCol=f"{documents_column}_tfidf_vector")

    p = Pipeline(stages=[tokenizer, stopwordsRemover, tf, idf]).fit(corpus)

    # create text_len:
    corpus = p.transform(corpus)
    corpus = corpus.withColumn(f"{documents_column}_len", F.size(f"processed_{documents_column}"))
    corpus = corpus.withMetadata(f"{documents_column}_len", {"avg_doc_len": corpus.agg({f"{documents_column}_len": "mean"}).collect()[0][f"avg({documents_column}_len)"]})

    corpus = corpus.drop(f"processed_{documents_column}_0", 
                         f"processed_{documents_column}_1")

    indexing_pipeline = IndexingPipelineModel(pipeline=p)

    return corpus, indexing_pipeline

# COMMAND ----------

indexed_corpus, indexing_pipeline = index_corpus(raw_corpus=profiles, documents_column="about")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1 - Without UDF

# COMMAND ----------

from pyspark.ml.feature import CountVectorizerModel

def retrieve1(query: str, 
             indexed_corpus: pyspark.sql.DataFrame, 
             indexing_pipeline: Pipeline, 
             documents_column: str, 
             topn: int=50,
             k1: float=0.7, 
             b: float=0.25) -> pyspark.sql.DataFrame:
    """
        Returns the corpus with additional columns:
        text_bm25_rsv: the Retrieval Status Value of the Okapi-BM25 retrieval method
        text_bm25_rank: the ranking induced by the rsv of documents in the corpus
    """

    # Parse the query of relevant information:
    TFModel = None
    for stage in indexing_pipeline.getPipeline().stages:
        if isinstance(stage, CountVectorizerModel):
            TFModel = stage
            break
    query = query.lower()
    query_index_list = [TFModel.vocabulary.index(term) for term in query.split(' ')]
    query_col = F.array([F.lit(term) for term in query.split(' ')])

    avg_doc_len = indexed_corpus.schema[f"{documents_column}_len"].metadata["avg_doc_len"]

    # filter to only rows that contain at least one of the query terms:
    tfidf = indexed_corpus.filter(F.size(F.array_intersect(f"processed_{documents_column}", query_col)) > 0)

    # filter arrays to only include query entries:
    tfidf = tfidf.withColumn(f"{documents_column}_tf_query", 
                             F.array([F.element_at(mlF.vector_to_array(F.col(f"{documents_column}_tf_vector")), i + 1) 
                                      for i in query_index_list]))
    tfidf = tfidf.withColumn(f"{documents_column}_tfidf_query", 
                             F.array([F.element_at(mlF.vector_to_array(F.col(f"{documents_column}_tfidf_vector")), i + 1) 
                                      for i in query_index_list]))
    tfidf = tfidf.withColumn(f"{documents_column}_len_query", 
                             F.array([F.element_at(F.expr(f"transform(sequence(1, {TFModel.getVocabSize()}), x -> {documents_column}_len)"), i + 1) 
                                      for i in query_index_list]))

    # Calculate BM25 score for each document in the corpus:
    tfidf = tfidf.withColumn(f"{documents_column}_bm25_rsv", F.arrays_zip(F.col(f"{documents_column}_tf_query").alias("tf_query"), F.col(f"{documents_column}_tfidf_query").alias("tfidf_query"), F.col(f"{documents_column}_len_query").alias("doc_len_query")))

    tfidf = tfidf.withColumn(f"{documents_column}_bm25_rsv", F.transform(f"{documents_column}_bm25_rsv", 
                                                lambda c: (c.tfidf_query * (k1 + 1)) / ((k1 * ((1 - b) + b * (c.doc_len_query / avg_doc_len))) + c.tf_query)))
    tfidf = tfidf.withColumn(f"{documents_column}_bm25_rsv", F.aggregate(f"{documents_column}_bm25_rsv", F.lit(0.0), lambda acc, x: acc + x))

    tfidf = tfidf.drop(f"{documents_column}_tf_query", f"{documents_column}_tfidf_query", f"{documents_column}_len_query")
    
    tfidf = tfidf.withColumn(f"{documents_column}_bm25_rank", F.dense_rank().over(Window.orderBy(F.col(f"{documents_column}_bm25_rsv").desc())))

    tfidf = tfidf.where(F.col(f"{documents_column}_bm25_rank") < topn)

    return tfidf


# COMMAND ----------

result1 = retrieve1(query="data science", 
                indexed_corpus=indexed_corpus, 
                indexing_pipeline=indexing_pipeline,
                documents_column="about",
                topn=50,
                k1=0.7,
                b=0.25).select("id", "followers", "about", "about_bm25_rsv", "about_bm25_rank").orderBy(F.col("about_bm25_rank").desc())
result1.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2 - With UDF

# COMMAND ----------

pyspark.ml.linalg.Vectors.dense([1,2,3]) / pyspark.ml.linalg.Vectors.dense([1,2,3])

# COMMAND ----------

from pyspark.ml.feature import CountVectorizerModel, VectorSlicer
import numpy as np
def retrieve2(query: str, 
             indexed_corpus: pyspark.sql.DataFrame, 
             indexing_pipeline: Pipeline, 
             documents_column: str, 
             topn: int=50,
             k1: float=0.7, 
             b: float=0.25) -> pyspark.sql.DataFrame:
    """
        Returns the corpus with additional columns:
        text_bm25_rsv: the Retrieval Status Value of the Okapi-BM25 retrieval method
        text_bm25_rank: the ranking induced by the rsv of documents in the corpus
    """
    rsv_col_name = f"{query}_{documents_column}_bm25_rsv"
    rank_col_name = f"{query}_{documents_column}_bm25_rank"

    # Parse the query for relevant information:
    TFModel = None
    for stage in indexing_pipeline.getPipeline().stages:
        if isinstance(stage, CountVectorizerModel):
            TFModel = stage
            break
    query = query.lower()
    query_index_list = []
    for term in query.split(' '):
        if term in TFModel.vocabulary:
            query_index_list.append(TFModel.vocabulary.index(term))
    query_index_list = spark.sparkContext.broadcast(query_index_list)
    query_col = F.array([F.lit(term) for term in query.split(' ')])

    avg_doc_len = spark.sparkContext.broadcast(indexed_corpus.schema[f"{documents_column}_len"].metadata["avg_doc_len"])

    # filter to only rows that contain at least one of the query terms:
    tfidf = indexed_corpus.filter(F.size(F.array_intersect(f"processed_{documents_column}", query_col)) > 0)

    @F.udf(returnType=DoubleType())
    def bm25(tf_vector: pyspark.ml.linalg.Vector, 
             tfidf_vector: pyspark.ml.linalg.Vector, 
             doc_len: int) -> float:
        if isinstance(tf_vector, pyspark.ml.linalg.SparseVector):
            # Initialize score
            score = 0.0
            # Iterate through the non-zero entries of the tf_vector
            for i, tf in zip(tf_vector.indices, tf_vector.values):
                # Ensure i is in the tfidf_vector.indices to avoid KeyError
                if i in tfidf_vector.indices:
                    # Find the corresponding tf-idf value
                    tfidf = tfidf_vector.values[list(tfidf_vector.indices).index(i)]
                    # Calculate the denominator for the BM25 formula component
                    denom = tf + (k1 * (1 - b + (b * (doc_len / avg_doc_len.value))))
                    # Update the score
                    score += tfidf * (k1 + 1) / denom
            return float(score)
        else:
            # Convert Spark DenseVectors to NumPy arrays
            tf_array = np.array(tf_vector)
            tfidf_array = np.array(tfidf_vector)
            
            # Calculate the BM25 term-by-term and sum them up
            denom = tf_array + (k1 * (1 - b + (b * (doc_len / avg_doc_len.value))))
            score = tfidf_array * (k1 + 1) / denom
            score = np.sum(score)
            return float(score)
    
    # filter the vectors before the UDF to save memory:
    query_terms_filter = VectorSlicer(inputCol=f"{documents_column}_tf_vector", 
                                      outputCol=f"{documents_column}_tf_vector_filtered", 
                                      indices=query_index_list.value)
    
    tfidf = query_terms_filter.transform(tfidf)

    query_terms_filter = query_terms_filter.setInputCol(f"{documents_column}_tfidf_vector").setOutputCol(f"{documents_column}_tfidf_vector_filtered")
    tfidf = query_terms_filter.transform(tfidf)

    # calculate scores:
    tfidf = tfidf.withColumn(rsv_col_name, 
                             bm25(f"{documents_column}_tf_vector_filtered", 
                                  f"{documents_column}_tfidf_vector_filtered", 
                                  f"{documents_column}_len"))

    # add ranks to the results and return top n results only:
    tfidf = tfidf.withColumn(rank_col_name, F.row_number().over(Window.orderBy(F.col(rsv_col_name).desc(), F.col(f"{documents_column}_len").desc())))
    tfidf = tfidf.where(F.col(rank_col_name) <= topn)

    return tfidf

# COMMAND ----------

result2 = retrieve2(query="data science",
                    indexed_corpus=indexed_corpus,
                    indexing_pipeline=indexing_pipeline,
                    documents_column="about",
                    topn=50,
                    k1=0.7,
                    b=0.25)
result2 = result2.select("id", "about", "about_tf_vector", "about_tfidf_vector", "data science_about_bm25_rsv", "data science_about_bm25_rank")
result2 = result2.orderBy(F.col("data science_about_bm25_rank"))
display(result2)

# COMMAND ----------

# MAGIC %md
# MAGIC Looks Pretty Good!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Corpus Visualizations:

# COMMAND ----------

from pyspark.ml.clustering import BisectingKMeans, LDA

# COMMAND ----------

latantDir = LDA(featuresCol="position_tf_vector").fit(indexed_corpus)

# COMMAND ----------

TFModel = None
for stage in indexing_pipeline.getPipeline().stages:
    if isinstance(stage, CountVectorizerModel):
        TFModel = stage
        break
index2term = TFModel.vocabulary

# COMMAND ----------

topn_terms = 20
topics_df = latantDir.describeTopics(topn_terms).toPandas()
topics_df

# COMMAND ----------

topics_df["terms"] = topics_df.apply(func=lambda x: [index2term[x['termIndices'][i]] for i in range(len(x['termIndices']))], axis=1)
by_topic = pd.DataFrame(index=range(topn_terms), columns=topics_df[['topic', 'terms']].T.columns)

for c in by_topic.columns:
    by_topic[c] = topics_df[['topic', 'terms']].T.at['terms', c]
by_topic

# COMMAND ----------

bisectKM = BisectingKMeans(k=10, featuresCol="position_tfidf_vector", predictionCol="bisectKM_predictions").fit(indexed_corpus)

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator
# Sum of squared distances between the input points and their corresponding cluster centers:
ClusteringEvaluator(featuresCol="position_tfidf_vector", predictionCol="bisectKM_predictions").evaluate(dataset=bisectKM.transform(indexed_corpus))

# COMMAND ----------

# MAGIC %md
# MAGIC Pretty bad clustering...

# COMMAND ----------

# MAGIC %md
# MAGIC #General Retrieval Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query Format

# COMMAND ----------

from collections import OrderedDict
query_format = OrderedDict([
    ('field_of_interest', 'data'),
    ('desired_position', 'machine learning engineer'),
    ('desired_companies', 'google amazon meta'),
    ('desired_industry', 'high tech')
])

query_field_map = OrderedDict([
    ('field_of_interest', ['about', 'position']),
    ('desired_position', ['position']),
    ('desired_companies', ['about', 'current_company:name']),
    ('desired_industry', ['about'])
])
query_fields = list(set([val for vals in query_field_map.values() for val in vals]))
query_fields

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complete Indexing

# COMMAND ----------

from tqdm.auto import tqdm
def index_complete_query(raw_corpus: pyspark.sql.DataFrame, query_fields: list[str]):
    indexed_corpus = raw_corpus.select("*")  # copies the dataframe to avoid changing the source
    pipe_stages = []
    t_bar = tqdm(query_fields, total=len(query_fields))
    for field in t_bar:
        t_bar.set_description(f"Indexing - {field}")
        indexed_corpus, indexing_pipeline = index_corpus(raw_corpus=indexed_corpus, documents_column=field)
        pipe_stages.append(indexing_pipeline)
    complete_indexing_pipeline = Pipeline(stages=pipe_stages)
    index_columns = ["id", 
    *[field for field in query_fields], 
    *[field + "_len" for field in query_fields], 
    *["processed_" + field for field in query_fields], 
    *[field + "_tf_vector" for field in query_fields], 
    *[field + "_tfidf_vector" for field in query_fields]]
    indexed_corpus = indexed_corpus.select(*index_columns)
    return indexed_corpus, complete_indexing_pipeline

# COMMAND ----------

indexed_corpus, complete_indexing_pipeline = index_complete_query(raw_corpus=profiles, query_fields=query_fields)

# COMMAND ----------

indexed_corpus.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complete Retrieval

# COMMAND ----------

def retrieve_complete_query(query_format: dict, 
                            query_fields: list[str], 
                            query_field_map: dict, 
                            indexed_corpus: pyspark.sql.DataFrame, 
                            indexing_pipeline: Union[pyspark.ml.Pipeline, pyspark.ml.PipelineModel], 
                            topn: int):
    iter_args = [(query_key, query_value, field) for query_key, query_value in query_format.items() for field in query_field_map[query_key]]
    retrieved_profiles: pyspark.sql.DataFrame = None
    for query_key, query_value, field in iter_args:
        retrieved: pyspark.sql.DataFrame = retrieve2(query=query_value, 
                                                    indexed_corpus=indexed_corpus.select("id", f"processed_{field}", f"{field}_len", f"{field}_tf_vector", f"{field}_tfidf_vector"), 
                                                    indexing_pipeline=indexing_pipeline.getStages()[query_fields.index(field)],
                                                    documents_column=field,
                                                    topn=topn,
                                                    k1=0.7,
                                                    b=0.25).select(F.col("id"), 
                                                                    F.col(f"{query_value}_{field}_bm25_rsv"), 
                                                                    F.col(f"{query_value}_{field}_bm25_rank"))
        # retrieved = retrieved.coalesce(numPartitions=1)
        if retrieved_profiles is None:
            retrieved_profiles = retrieved
        else:
            retrieved = F.broadcast(retrieved)
            retrieved = retrieved.withColumnRenamed("id", "other_id")
            retrieved_profiles = retrieved_profiles.join(retrieved, 
                                                         on=retrieved_profiles["id"] == retrieved["other_id"], 
                                                         how="outer")
            retrieved_profiles = retrieved_profiles.select(
                F.when(retrieved_profiles["id"].isNotNull(), retrieved_profiles["id"]).otherwise(retrieved["other_id"]).alias("new_id"), 
                "*")
            retrieved_profiles = retrieved_profiles.drop("id", "other_id")
            retrieved_profiles = retrieved_profiles.withColumnRenamed("new_id", "id")
    return retrieved_profiles

# COMMAND ----------

results = retrieve_complete_query(query_format=query_format, 
                                  query_fields=query_fields, 
                                  query_field_map=query_field_map, 
                                  indexed_corpus=indexed_corpus, 
                                  indexing_pipeline=complete_indexing_pipeline, 
                                  topn=50)
results.display()

# COMMAND ----------

results.replace(float('nan'), None).summary("count").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fusion

# COMMAND ----------

def reciprocal_rank(retrieved_profiles: pyspark.sql.DataFrame, v: float):
    return retrieved_profiles.withColumn("fused_rank", sum([F.when(F.col(c).isNotNull(), 1/(F.col(c) + v)).otherwise(0) for c in retrieved_profiles.columns if c.endswith("rank")]))

# COMMAND ----------

fused_results = reciprocal_rank(retrieved_profiles=results, v=1)
retrieval_example = fused_results.join(profiles, 'id', 'inner').orderBy(F.col("fused_rank")).select(*profiles.columns)
display(retrieval_example)

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieval stage is COMPLETE!🥳💪🥂👏🍾

# COMMAND ----------

retrieval_example.write.mode("overwrite").parquet(PROJECT_DATA_DIR + "/retrieval_example")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval of an Inconsistent Example

# COMMAND ----------

# Something completely unrelated and impossible to find, like a lumberjack in the software development industry

query_form = OrderedDict([
    ('field_of_interest', 'healthcare'),
    ('desired_position', 'lumberjack'),
    ('desired_companies', 'construction'),
    ('desired_industry', 'software development')
])

query_fields = list(set([val for vals in query_field_map.values() for val in vals]))
query_fields

bad_results = retrieve_complete_query(query_format=query_form, 
                                  query_fields=query_fields, 
                                  query_field_map=query_field_map, 
                                  indexed_corpus=indexed_corpus, 
                                  indexing_pipeline=complete_indexing_pipeline, 
                                  topn=50)

bad_fused_results = reciprocal_rank(retrieved_profiles=bad_results, v=1).orderBy(F.col("fused_rank"))
bad_retrieval_example = bad_fused_results.join(profiles, 'id', 'inner').select(*profiles.columns)
bad_retrieval_example.write.mode("overwrite").parquet(PROJECT_DATA_DIR + "/bad_retrieval_example")
display(bad_retrieval_example)
