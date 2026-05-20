

import os, sys, subprocess

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pyspark", "pandas", "pyarrow"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\bigdata_project"
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")

# ── Task 1.1: Create project directory structure ─────────────────────────────
print("\n[TASK 1.1] Creating project directory structure...")
for d in ["data/raw", "data/processed", "notebooks", "src",
          "models", "reports/figures", "reports/presentations"]:
    os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
    print(f"  OK: {os.path.join(PROJECT_ROOT, d)}")

with open(os.path.join(PROJECT_ROOT, ".gitignore"), "w") as f:
    f.write("__pycache__/\n*.pyc\ndata/raw/\ndata/processed/\nmodels/\n*.log\n")

with open(os.path.join(PROJECT_ROOT, "requirements.txt"), "w") as f:
    f.write(
        "pyspark>=3.4.0\npandas\npyarrow\nfastparquet\n"
        "matplotlib\nwordcloud\narabic_reshaper\npython-bidi\nkafka-python\n"
    )
print("  .gitignore & requirements.txt created.")

# ── Task 1.2: Start PySpark ──────────────────────────────────────────────────
print("\n[TASK 1.2] Starting PySpark environment...")
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MSBDA801_Phase1") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print(f"  Spark version       : {spark.version}")
print(f"  Master              : {spark.sparkContext.master}")
print(f"  Default parallelism : {spark.sparkContext.defaultParallelism}")

# ── Task 1.3: Load local CSV ─────────────────────────────────────────────────
print("\n[TASK 1.3] Loading dataset from local CSV...")

CSV_PATH = os.path.join(RAW_DIR, "raw_combined_abstracts.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] CSV not found at:\n  {CSV_PATH}\n\n"
        "Upload it from Windows PowerShell first:\n"
        "  scp \"C:\\Users\\areej\\OneDrive\\Desktop\\AI_Arabic_Detection_ProBDA"
        "\\Data\\raw\\raw_combined_abstracts.csv\" "
        "USER@192.168.8.94:~/bigdata_project/data/raw/\n"
    )

df = spark.read \
    .option("header",    "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "true") \
    .option("escape",    '"') \
    .option("encoding",  "UTF-8") \
    .csv(CSV_PATH)

print(f"  Rows loaded  : {df.count()}")
print(f"  Columns      : {df.columns}")
df.printSchema()

# ── Validate & clean ─────────────────────────────────────────────────────────
from pyspark.sql.functions import col, length, count, when, isnan, avg, stddev

assert "text"  in df.columns, "Column 'text' not found in CSV!"
assert "label" in df.columns, "Column 'label' not found in CSV!"

df = df.withColumn("label", col("label").cast("integer")) \
       .filter(col("text").isNotNull()) \
       .filter(col("text") != "")

print(f"  Rows after cleaning: {df.count()}")

# ── Save as Parquet ──────────────────────────────────────────────────────────
parquet_path = os.path.join(RAW_DIR, "raw_abstracts.parquet")
df.write.mode("overwrite").parquet(parquet_path)
print(f"  Saved Parquet : {parquet_path}")

# ── Task 1.4: Initial EDA ────────────────────────────────────────────────────
print("\n[TASK 1.4] Initial EDA using Spark DataFrames...")

total = df.count()
print(f"\n  Total samples : {total}")

print("\n  >> Sample rows (3):")
df.show(3, truncate=80)

print("\n  >> Label distribution  (0=Human, 1=AI):")
df.groupBy("label").count() \
  .withColumnRenamed("count", "samples") \
  .orderBy("label").show()

print("\n  >> Text length statistics (characters):")
df.select(length(col("text")).alias("char_len")).describe().show()

print("\n  >> Null / empty value counts:")
df.select(
    count(when(col("text").isNull()  | (col("text") == ""), "x")).alias("null_text"),
    count(when(col("label").isNull() | isnan(col("label")),  "x")).alias("null_label"),
).show()

print("\n  >> Average text length per label:")
df.withColumn("char_len", length(col("text"))) \
  .groupBy("label") \
  .agg(
      count("*").alias("n_samples"),
      avg("char_len").alias("avg_chars"),
      stddev("char_len").alias("std_chars"),
  ).orderBy("label").show()

print("\n" + "="*60)
print("  PHASE 1 COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"  Project root : {PROJECT_ROOT}")
print(f"  Raw Parquet  : {parquet_path}")
print("="*60)

spark.stop()