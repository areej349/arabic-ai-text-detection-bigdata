"""
=============================================================
Input  : ~/bigdata_project/data/raw/raw_abstracts.parquet
Columns: text, label
Tasks  :
  2.1 - Arabic NLP preprocessing (normalize, stopwords, stem)
  2.2 - Save as Parquet + ORC
  2.3 - EDA: unigrams, bigrams, TTR, word cloud
  M/R  - Word count + Bigram frequency (pure RDD, no pandas_udf)
=============================================================
"""

import os, sys, subprocess, re
from collections import Counter

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pyspark", "pyarrow", "pandas", "matplotlib",
            "wordcloud", "arabic_reshaper", "python-bidi"]:
    try:
        __import__(pkg.split("-")[0].replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, explode, split as spark_split,
    count, desc, collect_list, avg
)
from pyspark.sql.types import StringType
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\bigdata_project"
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR      = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Spark ────────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MSBDA801_Phase2") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("[INFO] Spark started. Version:", spark.version)

# ── Load Parquet from Phase 1 ─────────────────────────────────────────────────
print("\n[TASK 2.1] Loading data from Phase 1 Parquet...")
parquet_in = os.path.join(RAW_DIR, "raw_abstracts.parquet")

if not os.path.exists(parquet_in):
    raise FileNotFoundError(
        f"[ERROR] Run phase1_setup.py first.\n  Missing: {parquet_in}"
    )

df = spark.read.parquet(parquet_in)
print(f"  Rows: {df.count():,}")
df.printSchema()

# ── Arabic Preprocessing ──────────────────────────────────────────────────────
ARABIC_STOPWORDS = set([
    "في","من","إلى","على","عن","مع","هذا","هذه","ذلك","التي","الذي",
    "وقد","كما","إلا","أن","إن","كان","لا","ما","هو","هي","لم","لن",
    "قد","أو","حتى","أي","بين","له","لها","بما","عند","كل","بعد","قبل",
    "ثم","منذ","خلال","حول","نحو","غير","بل","لكن","وهو","وهي","فإن",
    "وأن","أما","إذا","كانت","وكان","أيضا","فقد","وفي","ومن","وعلى",
    "فإن","وإن","كذلك","ولا","ولم","وكانت","أنه","أنها","إنه","إنها",
])

def normalize_arabic(text):
    if not text:
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    if not text:
        return ""
    return " ".join(w for w in text.split()
                    if w not in ARABIC_STOPWORDS and len(w) > 1)

def light_stem(word):
    prefixes = ['ال','وال','بال','كال','فال','لل','و','ف','ب','ك','ل','س']
    suffixes = ['ها','هم','هن','كم','كن','نا','ون','ين','ان','ات','وا','ية']
    for p in prefixes:
        if word.startswith(p) and len(word) - len(p) >= 3:
            word = word[len(p):]
            break
    for s in suffixes:
        if word.endswith(s) and len(word) - len(s) >= 3:
            word = word[:-len(s)]
            break
    return word

def full_preprocess(text):
    t = normalize_arabic(text)
    t = remove_stopwords(t)
    t = " ".join(light_stem(w) for w in t.split())
    return t

normalize_udf  = udf(normalize_arabic, StringType())
stopword_udf   = udf(remove_stopwords,  StringType())
preprocess_udf = udf(full_preprocess,   StringType())

print("  Applying preprocessing pipeline...")
df_processed = df \
    .withColumn("normalized_text",   normalize_udf(col("text"))) \
    .withColumn("no_stopwords_text", stopword_udf(col("normalized_text"))) \
    .withColumn("clean_text",        preprocess_udf(col("text")))

print("  Sample (original vs clean):")
df_processed.select(
    col("text").alias("original"),
    col("clean_text")
).show(3, truncate=70)

# ── Task 2.2: Save Parquet + ORC ─────────────────────────────────────────────
print("\n[TASK 2.2] Saving processed data...")
parquet_out = os.path.join(PROC_DIR, "processed_abstracts.parquet")
orc_out     = os.path.join(PROC_DIR, "processed_abstracts.orc")

df_processed.write.mode("overwrite").parquet(parquet_out)
print(f"  Parquet saved : {parquet_out}")

df_processed.write.mode("overwrite").orc(orc_out)
print(f"  ORC saved     : {orc_out}")

# ── Task 2.3: EDA ─────────────────────────────────────────────────────────────
print("\n[TASK 2.3] EDA on processed data...")

# Top unigrams
words_df = df_processed \
    .select(explode(spark_split(col("clean_text"), " ")).alias("word")) \
    .filter(col("word") != "")

print("\n  >> Top 20 Unigrams:")
words_df.groupBy("word").count().orderBy(desc("count")).show(20)

# TTR per label
print("\n  >> Type-Token Ratio (TTR) per label:")
label_texts = df_processed.groupBy("label") \
    .agg(collect_list("clean_text").alias("all_texts")) \
    .toPandas()

for _, row in label_texts.iterrows():
    all_words    = " ".join(row["all_texts"]).split()
    total_tokens = len(all_words)
    unique_types = len(set(all_words))
    ttr          = unique_types / total_tokens if total_tokens else 0
    lname        = "Human" if int(row["label"]) == 0 else "AI"
    print(f"    Label {row['label']} ({lname}): "
          f"tokens={total_tokens:,}, types={unique_types:,}, TTR={ttr:.4f}")

# Bigrams on sample (pure Python — no pandas_udf)
print("\n  >> Top 15 Bigrams (on 1000-row sample):")
sample_texts = df_processed.select("clean_text").limit(1000).toPandas()
bigram_cnt   = Counter()
for text in sample_texts["clean_text"].dropna():
    tokens = text.split()
    bigram_cnt.update(" ".join(tokens[i:i+2]) for i in range(len(tokens)-1))
for bg, c in bigram_cnt.most_common(15):
    print(f"    {bg}: {c}")

# Word cloud
print("\n  >> Generating word cloud...")
try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import arabic_reshaper
    from bidi.algorithm import get_display

    all_text = " ".join(
        df_processed.select("clean_text").limit(2000)
        .toPandas()["clean_text"].dropna().tolist()
    )
    wc = WordCloud(
        background_color="white", width=1200, height=600,
        max_words=100
    ).generate(get_display(arabic_reshaper.reshape(all_text)))

    fig_path = os.path.join(FIG_DIR, "wordcloud_phase2.png")
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Arabic Word Cloud — Preprocessed")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Word cloud saved: {fig_path}")
except Exception as e:
    print(f"  Word cloud skipped: {e}")

# ── MapReduce Sim 1: Word Count (pure RDD) ────────────────────────────────────
print("\n[MapReduce Sim 1] Word Count (RDD M/R)...")

words_rdd = df_processed.select("clean_text").rdd \
    .flatMap(lambda r: r["clean_text"].split() if r["clean_text"] else []) \
    .filter(lambda w: w != "")

# MAP: (word, 1)  →  REDUCE: sum
word_count_rdd = words_rdd \
    .map(lambda w: (w, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: x[1], ascending=False)

total_tokens = words_rdd.count()
print(f"  Total tokens (M/R): {total_tokens:,}")
print("  Top 10 words:")
for word, cnt in word_count_rdd.take(10):
    print(f"    {word}: {cnt:,}")

# Save as Parquet
mr_wc_df  = spark.createDataFrame(word_count_rdd, ["word", "count"])
mr_wc_path = os.path.join(PROC_DIR, "mr_word_count.parquet")
mr_wc_df.write.mode("overwrite").parquet(mr_wc_path)
print(f"  Saved: {mr_wc_path}")

# ── MapReduce Sim 2: Bigram Frequency (pure RDD) ──────────────────────────────
print("\n[MapReduce Sim 2] Bigram Frequency (RDD M/R)...")

def get_bigrams(text):
    if not text:
        return []
    tokens = text.split()
    return [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]

bigram_rdd = df_processed.select("clean_text").rdd \
    .flatMap(lambda r: get_bigrams(r["clean_text"])) \
    .filter(lambda b: b.strip() != "")

# MAP: (bigram, 1)  →  REDUCE: sum
bigram_count_rdd = bigram_rdd \
    .map(lambda b: (b, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: x[1], ascending=False)

print("  Top 10 Bigrams (M/R):")
for bg, cnt in bigram_count_rdd.take(10):
    print(f"    {bg}: {cnt:,}")

# Save as Parquet
mr_bg_df   = spark.createDataFrame(bigram_count_rdd, ["bigram", "count"])
mr_bg_path = os.path.join(PROC_DIR, "mr_bigram_count.parquet")
mr_bg_df.write.mode("overwrite").parquet(mr_bg_path)
print(f"  Saved: {mr_bg_path}")

print("\n" + "="*60)
print("  PHASE 2 COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"  Processed Parquet : {parquet_out}")
print(f"  Processed ORC     : {orc_out}")
print(f"  M/R Word Count    : {mr_wc_path}")
print(f"  M/R Bigram Count  : {mr_bg_path}")
print("="*60)

spark.stop()