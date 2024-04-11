// Databricks notebook source
import scala.io.Source
import spark.implicits._
import org.zouzias.spark.lucenerdd._
import org.zouzias.spark.lucenerdd.LuceneRDD

// Load the profiles data
val profiles = spark.read.parquet("/linkedin/people")

// Convert the necessary columns to RDD
val rdd = profiles.select("id", "about", "current_company:name", "position").rdd

// Initialize LuceneRDD with the RDD
val luceneRDD = LuceneRDD(rdd)

// COMMAND ----------

// Perform the term query
val queryString = "data"
val hits = luceneRDD.termQuery("about", queryString, 50)

val results = hits.collect() // Collect results to the driver for printing

// Loop through the results and print each record
results.foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC This Also took longer then our pyspark UDF, even though this was in scala and used a robust open-source library!
