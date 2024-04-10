// Databricks notebook source
import scala.io.Source
import org.zouzias.spark.lucenerdd._
import org.zouzias.spark.lucenerdd.LuceneRDD
val profiles = spark.read.parquet("/linkedin/people")
val rdd = profiles.select("id", "name", "about", "current_company:name", "position", "city", "followers", "following", "recommendations_count", "url").rdd
val luceneRDD = LuceneRDD(rdd)
val queryString = "data"

// COMMAND ----------

val hits = luceneRDD.prefixQuery("about", queryString, 50).collect
println(hits)
