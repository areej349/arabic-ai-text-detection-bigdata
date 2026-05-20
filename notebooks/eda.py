"""
MSBDA-801 — Task 1.4: Initial Data Exploration
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, count, when, avg, stddev

spark = SparkSession.builder \
    .appName("EDA_Task14") \
    .master("local[*]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(
    "/home/hadoop/bigdata_project/data/raw/raw_abstracts.parquet"
)

print("\n" + "="*55)
print("  TASK 1.4: Initial Data Exploration (Spark DataFrame)")
print("="*55)

print("\n>> Schema:")
df.printSchema()

print(f">> Total rows: {df.count():,}")

print("\n>> Sample rows (3):")
df.show(3, truncate=80)

print(">> Label distribution (0=Human, 1=AI):")
df.groupBy("label").count().orderBy("label").show()

print(">> Text length statistics (characters):")
df.select(length(col("text")).alias("char_len")).describe().show()

print(">> Avg text length per label:")
df.withColumn("char_len", length(col("text"))) \
  .groupBy("label") \
  .agg(
      count("*").alias("n_samples"),
      avg("char_len").alias("avg_chars"),
      stddev("char_len").alias("std_chars")
  ).orderBy("label").show()

print(">> Null / empty counts:")
df.select(
    count(when(col("text").isNull() | (col("text") == ""), "x")).alias("null_text"),
    count(when(col("label").isNull(), "x")).alias("null_label")
).show()

print("="*55)
print("  TASK 1.4 COMPLETED!")
print("="*55)

spark.stop()