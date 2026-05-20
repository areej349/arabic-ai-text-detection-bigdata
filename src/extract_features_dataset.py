"""

=============================================================
Input  : data/processed/processed_abstracts.parquet
Output : data/processed/features_extracted.parquet

Columns in output:
  text, label, clean_text,
  f3_digits_ratio,
  f24_punct_div,
  f45_adj_count,
  f66_genitive_count,
  f87_gini
=============================================================
Run on CentOS:
  python3 ~/bigdata_project/extract_features_dataset.py
=============================================================
"""

import os, sys, subprocess, re
from collections import Counter

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pyspark", "pyarrow", "pandas"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import (
    DoubleType, StructType, StructField
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/bigdata_project")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_PATH   = os.path.join(PROC_DIR, "processed_abstracts.parquet")
OUTPUT_PATH  = os.path.join(PROC_DIR, "features_extracted.parquet")
OUTPUT_CSV   = os.path.join(PROC_DIR, "features_extracted.csv")

# ── Spark ─────────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MSBDA801_FeatureExtraction") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("[INFO] Spark:", spark.version)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading: {INPUT_PATH}")
df = spark.read.parquet(INPUT_PATH)
df = df.select("text", "clean_text", col("label").cast("integer")) \
       .dropna(subset=["text", "label"]) \
       .filter(col("text") != "")

total = df.count()
print(f"  Total rows: {total:,}")
df.printSchema()

# ── 5 Assigned Features UDF ───────────────────────────────────────────────────
ARABIC_PUNCT  = set('،؛؟!:.,"\'()[]{}')
GENITIVE_PREP = re.compile(r'\b(في|من|إلى|على|عن|مع)\s')
ADJ_PATTERN   = re.compile(r'ال\w+ي\b')
DIGIT_RE      = re.compile(r'[٠-٩0-9]')

@udf(returnType=StructType([
    StructField("f3_digits_ratio",    DoubleType()),
    StructField("f24_punct_div",      DoubleType()),
    StructField("f45_adj_count",      DoubleType()),
    StructField("f66_genitive_count", DoubleType()),
    StructField("f87_gini",           DoubleType()),
]))
def compute_5_features(text):
    """
    Compute all 5 assigned stylometric features for a single row.
    Returns struct with all feature values.
    """
    if not text or len(text) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    C = len(text)

    # ── f3: Number of digits / C ──────────────────────────────────────────────
    f3 = len(DIGIT_RE.findall(text)) / C

    # ── f24: Number of different punctuation signs / C ────────────────────────
    unique_punct = {ch for ch in text if ch in ARABIC_PUNCT}
    f24 = len(unique_punct) / C

    # ── f45: Number of adjectives (ال...ي pattern) ────────────────────────────
    f45 = float(len(ADJ_PATTERN.findall(text)))

    # ── f66: Number of genitives (words after prepositions) ───────────────────
    f66 = float(len(GENITIVE_PREP.findall(text)))

    # ── f87: Gini Coefficient of Word Frequencies ─────────────────────────────
    words = text.split()
    N = len(words)
    if N == 0:
        return (f3, f24, f45, f66, 0.0)

    from collections import Counter
    freq  = Counter(words)
    freqs = sorted(freq.values())   # ascending order
    n_t   = len(freqs)
    tot_f = sum(freqs)

    if tot_f == 0 or n_t == 0:
        f87 = 0.0
    else:
        cum = sum((i + 1) * f for i, f in enumerate(freqs))
        f87 = (2 * cum) / (n_t * tot_f) - (n_t + 1) / n_t

    return (f3, f24, f45, f66, f87)

# ── Apply UDF to every row ────────────────────────────────────────────────────
print("\n[INFO] Computing 5 features for every row...")
print("       f3(digits/C) | f24(punct/C) | f45(adj) | f66(genitives) | f87(Gini)")

df_features = df.withColumn("features", compute_5_features(col("text")))

# Unpack struct columns
df_features = df_features \
    .withColumn("f3_digits_ratio",    col("features.f3_digits_ratio")) \
    .withColumn("f24_punct_div",      col("features.f24_punct_div")) \
    .withColumn("f45_adj_count",      col("features.f45_adj_count")) \
    .withColumn("f66_genitive_count", col("features.f66_genitive_count")) \
    .withColumn("f87_gini",           col("features.f87_gini")) \
    .drop("features")

# ── Show sample ───────────────────────────────────────────────────────────────
print("\n  Sample rows (5):")
df_features.select(
    "label",
    "f3_digits_ratio",
    "f24_punct_div",
    "f45_adj_count",
    "f66_genitive_count",
    "f87_gini"
).show(5)

# ── Statistics per feature per label ─────────────────────────────────────────
print("\n  Feature statistics per label (Human=0, AI=1):")
from pyspark.sql.functions import avg, stddev, min as spark_min, max as spark_max

df_features.groupBy("label").agg(
    avg("f3_digits_ratio").alias("avg_f3"),
    avg("f24_punct_div").alias("avg_f24"),
    avg("f45_adj_count").alias("avg_f45"),
    avg("f66_genitive_count").alias("avg_f66"),
    avg("f87_gini").alias("avg_f87"),
).orderBy("label").show()

# ── Row counts per label ──────────────────────────────────────────────────────
print("  Row count per label:")
df_features.groupBy("label").count().orderBy("label").show()

# ── Save as Parquet ───────────────────────────────────────────────────────────
print(f"\n[INFO] Saving as Parquet: {OUTPUT_PATH}")
df_features.write.mode("overwrite").parquet(OUTPUT_PATH)
print(f"  ✅ Parquet saved: {OUTPUT_PATH}")

# ── Also save sample as CSV (first 1000 rows for inspection) ─────────────────
print(f"\n[INFO] Saving sample CSV (1000 rows): {OUTPUT_CSV}")
df_features.limit(1000) \
    .toPandas() \
    .to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"  ✅ CSV saved: {OUTPUT_CSV}")

# ── Verify saved file ─────────────────────────────────────────────────────────
print("\n[INFO] Verifying saved Parquet...")
df_verify = spark.read.parquet(OUTPUT_PATH)
print(f"  Rows in saved file: {df_verify.count():,}")
print(f"  Columns: {df_verify.columns}")
df_verify.printSchema()

print("\n" + "="*60)
print("  FEATURE EXTRACTION COMPLETED!")
print("="*60)
print(f"  Output Parquet : {OUTPUT_PATH}")
print(f"  Output CSV     : {OUTPUT_CSV}")
print(f"  Total rows     : {total:,}")
print(f"  Features saved : f3, f24, f45, f66, f87")
print("="*60)

spark.stop()