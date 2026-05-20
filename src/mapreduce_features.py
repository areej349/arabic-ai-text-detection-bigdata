"""
=============================================================
MSBDA-801 Big Data Analytics
MapReduce Conceptual Design & Implementation
for 5 Assigned Stylometric Features
=============================================================
Assigned Features:
  f3  → Number of digits / C
  f24 → Number of different punctuation signs / C
  f45 → Number of adjectives
  f66 → Number of genitives
  f87 → Gini Coefficient of Word Frequencies

MapReduce Design (Conceptual + Spark RDD Implementation):
─────────────────────────────────────────────────────────
Each feature is computed using M/R-style RDD operations:
  MAP    → extract (key, value) pairs per document
  REDUCE → aggregate across corpus
─────────────────────────────────────────────────────────
Run on CentOS:
  python3 ~/bigdata_project/mapreduce_features.py
=============================================================

PSEUDOCODE DESIGN
=================

──────────────────────────────────────────────────────────────
FEATURE f3: Number of digits / C
──────────────────────────────────────────────────────────────
M/R Job 1 — Per-document digit count:
  MAP   : (doc_id, text) → (doc_id, count_digits(text))
  MAP   : (doc_id, text) → (doc_id, len(text))
  REDUCE: (doc_id, [digit_count, char_count]) → digit_count / char_count

M/R Job 2 — Corpus-level aggregation:
  MAP   : (doc_id, f3_value) → ("f3", f3_value)
  REDUCE: ("f3", [v1,v2,...]) → mean(values)

──────────────────────────────────────────────────────────────
FEATURE f24: Number of different punctuation signs / C
──────────────────────────────────────────────────────────────
M/R Job 1 — Per-document unique punctuation types:
  MAP   : (doc_id, text) → (doc_id, {unique_punct_chars})
  MAP   : (doc_id, text) → (doc_id, len(text))
  REDUCE: (doc_id, [punct_set, C]) → len(punct_set) / C

M/R Job 2 — Corpus-level aggregation:
  MAP   : (doc_id, f24_value) → ("f24", f24_value)
  REDUCE: ("f24", [v1,v2,...]) → mean(values)

──────────────────────────────────────────────────────────────
FEATURE f45: Number of adjectives
──────────────────────────────────────────────────────────────
M/R Job 1 — Per-document adjective count (pattern ال...ي):
  MAP   : (doc_id, text) → (doc_id, count_regex(ADJ_PATTERN, text))
  REDUCE: identity (per-document value)

M/R Job 2 — Corpus-level stats:
  MAP   : (doc_id, f45_value) → (label, f45_value)
  REDUCE: (label, [v1,v2,...]) → mean(values) per label

──────────────────────────────────────────────────────────────
FEATURE f66: Number of genitives
──────────────────────────────────────────────────────────────
M/R Job 1 — Per-document genitive count (words after في,من,إلى...):
  MAP   : (doc_id, text) → (doc_id, count_regex(GENITIVE_PREP, text))
  REDUCE: identity (per-document value)

M/R Job 2 — Corpus-level stats:
  MAP   : (doc_id, f66_value) → (label, f66_value)
  REDUCE: (label, [v1,v2,...]) → mean(values) per label

──────────────────────────────────────────────────────────────
FEATURE f87: Gini Coefficient of Word Frequencies
──────────────────────────────────────────────────────────────
M/R Job 1 — Word Count (same as standard word count M/R):
  MAP   : (doc_id, text) → for each word: (word, 1)
  REDUCE: (word, [1,1,1,...]) → (word, total_count)

M/R Job 2 — Per-document word frequencies:
  MAP   : (doc_id, text) → (doc_id, Counter(words))
  REDUCE: (doc_id, freq_dict) → Gini(sorted_frequencies)

Gini Formula:
  freqs  = sorted(freq.values())   # ascending
  n      = len(freqs)
  total  = sum(freqs)
  cum    = sum((i+1)*f for i,f in enumerate(freqs))
  Gini   = (2 * cum) / (n * total) − (n+1)/n

M/R Job 3 — Corpus-level Gini stats:
  MAP   : (doc_id, gini_value) → (label, gini_value)
  REDUCE: (label, [v1,v2,...]) → mean(values) per label
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
from pyspark.sql.functions import col
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/bigdata_project")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Spark ─────────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MSBDA801_MapReduce_Features") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("[INFO] Spark:", spark.version)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[INFO] Loading processed data...")
proc_path = os.path.join(PROC_DIR, "processed_abstracts.parquet")
df = spark.read.parquet(proc_path)
df = df.select("text", col("label").cast("integer")) \
       .dropna().filter(col("text") != "")

total_docs = df.count()
print(f"  Total documents: {total_docs:,}")

# Convert to RDD of (doc_id, text, label)
rdd = df.rdd.zipWithIndex() \
    .map(lambda x: (x[1], x[0]["text"], x[0]["label"]))

print(f"  RDD partitions: {rdd.getNumPartitions()}")

# ── Regex patterns ────────────────────────────────────────────────────────────
ARABIC_PUNCT  = set('،؛؟!:.,"\'()[]{}')
GENITIVE_PREP = re.compile(r'\b(في|من|إلى|على|عن|مع)\s')
ADJ_PATTERN   = re.compile(r'ال\w+ي\b')
DIGIT_RE      = re.compile(r'[٠-٩0-9]')

# =============================================================================
print("\n" + "="*60)
print("  MapReduce Feature Extraction (Spark RDD M/R Style)")
print("="*60)

# =============================================================================
# FEATURE f3: Number of digits / C
# =============================================================================
print("\n[f3] M/R: Number of digits / C")
print("  Job 1 MAP: (doc_id, text) → (doc_id, (digit_count, char_count))")
print("  Job 1 REDUCE: → (doc_id, f3_ratio)")

# MAP: each doc → (doc_id, (digit_count, char_count))
f3_rdd = rdd.map(lambda x: (
    x[0],
    (len(DIGIT_RE.findall(x[1])), len(x[1]), x[2])
))
# REDUCE: compute ratio per doc
f3_vals = f3_rdd.map(lambda x: (
    x[0],
    x[1][0] / x[1][1] if x[1][1] > 0 else 0.0,
    x[1][2]  # label
))

# Corpus-level: mean per label
f3_by_label = f3_vals \
    .map(lambda x: (x[2], (x[1], 1))) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda x: (x[0], x[1][0]/x[1][1])) \
    .sortBy(lambda x: x[0])

print("  >> f3 mean per label (0=Human, 1=AI):")
for label, mean_val in f3_by_label.collect():
    name = "Human" if label == 0 else "AI"
    print(f"     Label {label} ({name}): avg f3 = {mean_val:.6f}")

# =============================================================================
# FEATURE f24: Number of different punctuation signs / C
# =============================================================================
print("\n[f24] M/R: Number of different punctuation signs / C")
print("  Job 1 MAP: (doc_id, text) → (doc_id, (unique_punct_count, char_count))")

f24_rdd = rdd.map(lambda x: (
    x[0],
    len({ch for ch in x[1] if ch in ARABIC_PUNCT}),
    len(x[1]),
    x[2]
))
f24_vals = f24_rdd.map(lambda x: (
    x[0],
    x[1] / x[2] if x[2] > 0 else 0.0,
    x[3]
))

f24_by_label = f24_vals \
    .map(lambda x: (x[2], (x[1], 1))) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda x: (x[0], x[1][0]/x[1][1])) \
    .sortBy(lambda x: x[0])

print("  >> f24 mean per label:")
for label, mean_val in f24_by_label.collect():
    name = "Human" if label == 0 else "AI"
    print(f"     Label {label} ({name}): avg f24 = {mean_val:.6f}")

# =============================================================================
# FEATURE f45: Number of adjectives
# =============================================================================
print("\n[f45] M/R: Number of adjectives (pattern ال...ي)")
print("  Job 1 MAP: (doc_id, text) → (doc_id, adj_count)")
print("  Job 2 MAP: (doc_id, adj_count) → (label, adj_count)")
print("  Job 2 REDUCE: (label, [counts]) → mean")

f45_rdd = rdd.map(lambda x: (
    x[0],
    float(len(ADJ_PATTERN.findall(x[1]))),
    x[2]
))

f45_by_label = f45_rdd \
    .map(lambda x: (x[2], (x[1], 1))) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda x: (x[0], x[1][0]/x[1][1])) \
    .sortBy(lambda x: x[0])

print("  >> f45 mean per label:")
for label, mean_val in f45_by_label.collect():
    name = "Human" if label == 0 else "AI"
    print(f"     Label {label} ({name}): avg adjectives = {mean_val:.4f}")

# =============================================================================
# FEATURE f66: Number of genitives
# =============================================================================
print("\n[f66] M/R: Number of genitives (words after prepositions)")
print("  Job 1 MAP: (doc_id, text) → (doc_id, genitive_count)")
print("  Job 2 REDUCE: (label, [counts]) → mean")

f66_rdd = rdd.map(lambda x: (
    x[0],
    float(len(GENITIVE_PREP.findall(x[1]))),
    x[2]
))

f66_by_label = f66_rdd \
    .map(lambda x: (x[2], (x[1], 1))) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda x: (x[0], x[1][0]/x[1][1])) \
    .sortBy(lambda x: x[0])

print("  >> f66 mean per label:")
for label, mean_val in f66_by_label.collect():
    name = "Human" if label == 0 else "AI"
    print(f"     Label {label} ({name}): avg genitives = {mean_val:.4f}")

# =============================================================================
# FEATURE f87: Gini Coefficient of Word Frequencies
# =============================================================================
print("\n[f87] M/R: Gini Coefficient of Word Frequencies")
print("  Job 1 MAP  : (doc_id, text) → for each word: (word, 1)")
print("  Job 1 REDUCE: (word, [1,1,...]) → (word, total_count)")
print("  Job 2 MAP  : (doc_id, text) → (doc_id, Counter(words))")
print("  Job 2 REDUCE: (doc_id, freq_dict) → Gini coefficient")

def compute_gini(text):
    """Compute Gini coefficient from word frequencies."""
    if not text:
        return 0.0
    words = text.split()
    N = len(words)
    if N == 0:
        return 0.0
    freq  = Counter(words)
    freqs = sorted(freq.values())
    n_t   = len(freqs)
    tot_f = sum(freqs)
    if tot_f == 0 or n_t == 0:
        return 0.0
    cum = sum((i+1)*f for i, f in enumerate(freqs))
    return (2*cum) / (n_t*tot_f) - (n_t+1)/n_t

# M/R Job 1: Global word count (standard M/R word count)
print("\n  >> M/R Job 1: Global Word Count")
global_word_count = rdd \
    .flatMap(lambda x: x[1].split()) \
    .filter(lambda w: w != "") \
    .map(lambda w: (w, 1)) \
    .reduceByKey(lambda a, b: a + b)

total_tokens     = global_word_count.map(lambda x: x[1]).sum()
total_vocab_size = global_word_count.count()
print(f"     Total tokens (corpus) : {int(total_tokens):,}")
print(f"     Vocabulary size        : {total_vocab_size:,}")

# M/R Job 2: Per-document Gini
print("\n  >> M/R Job 2: Per-document Gini Coefficient")
f87_rdd = rdd.map(lambda x: (
    x[0],
    compute_gini(x[1]),
    x[2]
))

f87_by_label = f87_rdd \
    .map(lambda x: (x[2], (x[1], 1))) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda x: (x[0], x[1][0]/x[1][1])) \
    .sortBy(lambda x: x[0])

print("  >> f87 Gini mean per label:")
for label, mean_val in f87_by_label.collect():
    name = "Human" if label == 0 else "AI"
    print(f"     Label {label} ({name}): avg Gini = {mean_val:.6f}")

# =============================================================================
# COMBINED SUMMARY
# =============================================================================
print("\n" + "="*60)
print("  MAPREDUCE RESULTS SUMMARY — 5 ASSIGNED FEATURES")
print("="*60)

# Collect all results
results = {}
for feat, rdd_obj in [
    ("f3_digits_ratio",    f3_vals),
    ("f24_punct_div",      f24_vals),
    ("f45_adj_count",      f45_rdd),
    ("f66_genitive_count", f66_rdd),
]:
    by_label = rdd_obj \
        .map(lambda x: (x[2], (x[1], 1))) \
        .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
        .map(lambda x: (x[0], round(x[1][0]/x[1][1], 6))) \
        .collectAsMap()
    results[feat] = by_label

# f87 separately
f87_map = f87_by_label.collectAsMap()
results["f87_gini"] = f87_map

# Print table
print(f"\n  {'Feature':<25} {'Human (0)':>12} {'AI (1)':>12} {'Diff':>10}")
print("  " + "-"*60)

descriptions = {
    "f3_digits_ratio":    "f3  - digits/C",
    "f24_punct_div":      "f24 - punct/C",
    "f45_adj_count":      "f45 - adjectives",
    "f66_genitive_count": "f66 - genitives",
    "f87_gini":           "f87 - Gini coeff",
}

summary_rows = []
for feat, desc in descriptions.items():
    human = results[feat].get(0, 0.0)
    ai    = results[feat].get(1, 0.0)
    diff  = ai - human
    print(f"  {desc:<25} {human:>12.6f} {ai:>12.6f} {diff:>+10.6f}")
    summary_rows.append({
        "feature": feat,
        "description": desc,
        "human_mean": round(human, 6),
        "ai_mean":    round(ai,    6),
        "difference": round(diff,  6),
    })

print("="*60)
print(f"  Total corpus tokens : {int(total_tokens):,}")
print(f"  Vocabulary size     : {total_vocab_size:,}")
print("="*60)

# Save results
summary_df  = pd.DataFrame(summary_rows)
summary_path = os.path.join(REPORT_DIR, "mr_features_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n  Results saved: {summary_path}")

print("\n" + "="*60)
print("  MAPREDUCE FEATURE EXTRACTION COMPLETED!")
print("="*60)

spark.stop()