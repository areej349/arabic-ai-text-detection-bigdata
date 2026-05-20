
import os, sys, subprocess, re, json

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pyspark", "pyarrow", "pandas", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, when, count
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, LinearSVC
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, BinaryClassificationEvaluator
)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\bigdata_project"
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
FIG_DIR      = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

# ── Spark ─────────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MSBDA801_Phase3") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("[INFO] Spark:", spark.version)

print("""
      ASSIGNED STYLOMETRIC FEATURES  (i=3, n=21) :
   f3  → Number of digits / C                             
   f24 → Number of different punctuation signs / C        
   f45 → Number of adjectives (pattern-based)             
   f66 → Number of genitives                              
   f87 → Gini Coefficient of Word Frequencies 

""")

# ── Task 3.1: Load pre-computed features ──────────────────────────────────────
print("[TASK 3.1] Loading pre-computed features from features_extracted.parquet...")

features_path = os.path.join(PROC_DIR, "features_extracted.parquet")
if not os.path.exists(features_path):
    raise FileNotFoundError(
        f"[ERROR] Run extract_features_dataset.py first.\n"
        f"  Missing: {features_path}"
    )

df = spark.read.parquet(features_path)
df = df.withColumn("label", col("label").cast("double")) \
       .dropna(subset=["text", "label", "f3_digits_ratio",
                       "f24_punct_div", "f45_adj_count",
                       "f66_genitive_count", "f87_gini"]) \
       .filter(col("text") != "")

total = df.count()
print(f"  Rows loaded: {total:,}")
print(f"  Columns    : {df.columns}")
df.printSchema()

# ── [NEW] Class Imbalance Analysis ───────────────────────────────────────────
print("\n[CLASS IMBALANCE] Analyzing label distribution...")

label_counts = df.groupBy("label").count().orderBy("label").collect()
count_human  = next((r["count"] for r in label_counts if r["label"] == 0.0), 0)
count_ai     = next((r["count"] for r in label_counts if r["label"] == 1.0), 0)
total_count  = count_human + count_ai
pct_human    = count_human / total_count * 100
pct_ai       = count_ai    / total_count * 100
ratio        = count_ai / count_human if count_human else float("inf")

print(f"  +-------+---------+--------+")
print(f"  | Label |  Count  |   Pct  |")
print(f"  +-------+---------+--------+")
print(f"  |   0   | {count_human:>7,} | {pct_human:>5.1f}% |  (Human)")
print(f"  |   1   | {count_ai:>7,} | {pct_ai:>5.1f}% |  (AI)")
print(f"  +-------+---------+--------+")
print(f"  Imbalance ratio (AI:Human) : {ratio:.2f}:1")
print(f"  [ACTION] Applying class weights to compensate imbalance.")

# Assigned feature columns
ASSIGNED_COLS = [
    "f3_digits_ratio",
    "f24_punct_div",
    "f45_adj_count",
    "f66_genitive_count",
    "f87_gini",
]

print("\n  Sample — 5 assigned features (from pre-computed file):")
df.select(["label"] + ASSIGNED_COLS).show(5)

print("  Feature statistics per label:")
df.groupBy("label").agg(
    avg("f3_digits_ratio").alias("avg_f3"),
    avg("f24_punct_div").alias("avg_f24"),
    avg("f45_adj_count").alias("avg_f45"),
    avg("f66_genitive_count").alias("avg_f66"),
    avg("f87_gini").alias("avg_f87"),
).orderBy("label").show()

# ── Task 3.2: TF-IDF ──────────────────────────────────────────────────────────
print("\n[TASK 3.2] Computing TF-IDF features (5000-dim)...")

text_col  = "clean_text" if "clean_text" in df.columns else "text"
print(f"  Using column: '{text_col}' for TF-IDF")

tokenizer = Tokenizer(inputCol=text_col, outputCol="tokens")
df_tok    = tokenizer.transform(df)

hashTF    = HashingTF(inputCol="tokens", outputCol="raw_tf", numFeatures=5000)
df_tf     = hashTF.transform(df_tok)

idf_model = IDF(inputCol="raw_tf", outputCol="tfidf", minDocFreq=3).fit(df_tf)
df_tfidf  = idf_model.transform(df_tf)

print("  TF-IDF (5000-dim) computed.")

# ── Assemble: TF-IDF + 5 assigned features ────────────────────────────────────
print("  Assembling feature vector: TF-IDF(5000) + f3 + f24 + f45 + f66 + f87")

assembler = VectorAssembler(
    inputCols=["tfidf"] + ASSIGNED_COLS,
    outputCol="features",
    handleInvalid="skip"
)
df_final = assembler.transform(df_tfidf)
df_final = df_final.dropna(subset=["label"]).cache()

print(f"  Final dataset: {df_final.count():,} rows")
print(f"  Feature vector dimensions: 5000 (TF-IDF) + 5 (stylometric) = 5005")

# ── Task 3.3: Train/Val/Test Split ────────────────────────────────────────────
print("\n[TASK 3.3] Splitting data 70 / 15 / 15 (no data leakage)...")
train_df, val_df, test_df = df_final.randomSplit([0.70, 0.15, 0.15], seed=42)
train_df.cache(); test_df.cache()
print(f"  Train : {train_df.count():,} rows (70%)")
print(f"  Val   : {val_df.count():,} rows (15%)")
print(f"  Test  : {test_df.count():,} rows (15%)")

# ── [NEW] Compute Class Weights on TRAIN only ─────────────────────────────────
print("\n[CLASS WEIGHTS] Computing weights from training set...")

train_total = train_df.count()
train_human = train_df.filter(col("label") == 0.0).count()
train_ai    = train_df.filter(col("label") == 1.0).count()

# weight = total / (n_classes * count_per_class)
weight_human = train_total / (2.0 * train_human) if train_human else 1.0
weight_ai    = train_total / (2.0 * train_ai)    if train_ai    else 1.0

print(f"  Train Human (0): {train_human:,}  → weight = {weight_human:.4f}")
print(f"  Train AI    (1): {train_ai:,}  → weight = {weight_ai:.4f}")

# Add classWeight column to train_df only
train_df = train_df.withColumn(
    "classWeight",
    when(col("label") == 0.0, weight_human).otherwise(weight_ai)
)

# ── Evaluators ────────────────────────────────────────────────────────────────
mc_eval  = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction")
bin_eval = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction")

def evaluate(model, data, name):
    preds = model.transform(data)
    acc   = mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"})
    f1    = mc_eval.evaluate(preds, {mc_eval.metricName: "f1"})
    prec  = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedPrecision"})
    rec   = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedRecall"})
    try:
        auc = bin_eval.evaluate(preds, {bin_eval.metricName: "areaUnderROC"})
    except Exception:
        auc = 0.0

    # Per-class metrics
    preds_pd = preds.select("label", "prediction").toPandas()
    for lbl, lname in [(0.0, "Human"), (1.0, "AI")]:
        tp = ((preds_pd["label"] == lbl) & (preds_pd["prediction"] == lbl)).sum()
        fp = ((preds_pd["label"] != lbl) & (preds_pd["prediction"] == lbl)).sum()
        fn = ((preds_pd["label"] == lbl) & (preds_pd["prediction"] != lbl)).sum()
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f  = 2*p*r / (p+r)  if (p + r)  else 0.0
        print(f"    {lname:<6} → Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

    print(f"\n  [{name}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}" if auc > 0 else "    ROC-AUC   : N/A")
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc}

results = []

# ── Task 3.4: Baseline — Logistic Regression (with class weights) ─────────────
print("\n[TASK 3.4] Baseline: Logistic Regression (with class weights)...")
lr = LogisticRegression(
    featuresCol="features", labelCol="label",
    weightCol="classWeight",          # [NEW] class weight applied
    maxIter=20, regParam=0.01)
lr_model = lr.fit(train_df)
results.append(evaluate(lr_model, test_df, "Logistic Regression (Baseline)"))
# ───────────────
print("\n>> Logistic Regression Errors:")

errors_df = lr_model.transform(test_df) \
    .filter(col("label") != col("prediction")) \
    .select("label", "prediction", "text") \
    .limit(5)

pdf = errors_df.toPandas()

for i, row in pdf.iterrows():
    print("=" * 80)
    print(f"True Label : {row['label']}")
    print(f"Prediction : {row['prediction']}")

    txt = str(row['text'])

    try:
        txt = txt.encode("latin1").decode("utf-8")
    except:
        pass

    print("Sample Text:")
    print(txt[:300])
    print()
# ───────────────
lr_model.save(os.path.join(MODEL_DIR, "lr_model"))
print(f"  Model saved: {MODEL_DIR}/lr_model")

# ── Task 3.5a: Random Forest (with class weights) ────────────────────────────
print("\n[TASK 3.5a] Advanced Model 1: Random Forest (with class weights)...")
rf = RandomForestClassifier(
    featuresCol="features", labelCol="label",
    weightCol="classWeight",          # [NEW] class weight applied
    numTrees=50, maxDepth=10, seed=42)
rf_model = rf.fit(train_df)
results.append(evaluate(rf_model, test_df, "Random Forest"))
rf_model.save(os.path.join(MODEL_DIR, "rf_model"))
print(f"  Model saved: {MODEL_DIR}/rf_model")

# Feature importance plot
print("\n  >> Feature Importances (5 assigned features):")
fi_arr  = rf_model.featureImportances.toArray()
n_tfidf = 5000
fi_vals = []
for i, sc in enumerate(ASSIGNED_COLS):
    idx = n_tfidf + i
    val = fi_arr[idx] if idx < len(fi_arr) else 0.0
    fi_vals.append(val)
    print(f"    {sc:<25}: {val:.6f}")

labels_plot = ["f3\ndigits/C","f24\npunct/C","f45\nadj","f66\ngenitives","f87\nGini"]
colors_plot = ["#e53935","#8e24aa","#1e88e5","#43a047","#fb8c00"]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels_plot, fi_vals, color=colors_plot, width=0.5, edgecolor="white")
ax.set_ylabel("Feature Importance", fontsize=11)
ax.set_title("Random Forest — 5 Assigned Stylometric Feature Importances\n(i=3, n=21)",
             fontsize=11, fontweight="bold")
for bar, val in zip(bars, fi_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
            f"{val:.5f}", ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
fi_path = os.path.join(FIG_DIR, "rf_feature_importance.png")
plt.savefig(fi_path, dpi=150); plt.close()
print(f"  Plot saved: {fi_path}")

# ── Task 3.5b: Linear SVM (no weightCol support in PySpark LinearSVC) ─────────
print("\n[TASK 3.5b] Advanced Model 2: Linear SVM...")
print("  [NOTE] LinearSVC in PySpark does not support weightCol.")
print("         Imbalance is compensated via regParam tuning.")
svm = LinearSVC(
    featuresCol="features", labelCol="label",
    maxIter=20, regParam=0.005)       # lower regParam to help minority class
svm_model = svm.fit(train_df)
svm_preds = svm_model.transform(test_df)
acc  = mc_eval.evaluate(svm_preds, {mc_eval.metricName: "accuracy"})
f1   = mc_eval.evaluate(svm_preds, {mc_eval.metricName: "f1"})
prec = mc_eval.evaluate(svm_preds, {mc_eval.metricName: "weightedPrecision"})
rec  = mc_eval.evaluate(svm_preds, {mc_eval.metricName: "weightedRecall"})
try:
    auc = bin_eval.evaluate(svm_preds, {bin_eval.metricName: "areaUnderROC"})
except Exception:
    auc = 0.0

# Per-class for SVM
svm_pd = svm_preds.select("label", "prediction").toPandas()
for lbl, lname in [(0.0, "Human"), (1.0, "AI")]:
    tp = ((svm_pd["label"] == lbl) & (svm_pd["prediction"] == lbl)).sum()
    fp = ((svm_pd["label"] != lbl) & (svm_pd["prediction"] == lbl)).sum()
    fn = ((svm_pd["label"] == lbl) & (svm_pd["prediction"] != lbl)).sum()
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f  = 2*p*r / (p+r)  if (p + r)  else 0.0
    print(f"    {lname:<6} → Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

print(f"\n  [Linear SVM]")
print(f"    Accuracy  : {acc:.4f}")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print(f"    ROC-AUC   : {auc:.4f}" if auc > 0 else "    ROC-AUC   : N/A")
results.append({"model": "Linear SVM", "accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "auc": auc})
svm_model.save(os.path.join(MODEL_DIR, "svm_model"))
print(f"  Model saved: {MODEL_DIR}/svm_model")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  MODEL COMPARISON — TEST SET (with class imbalance handling)")
print(f"  Dataset: {count_human:,} Human (20%) / {count_ai:,} AI (80%) — ratio {ratio:.1f}:1")
print(f"  Fix applied: classWeight (Human={weight_human:.3f}, AI={weight_ai:.3f})")
print("="*70)
print(f"  {'Model':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("  " + "-"*67)
for r in results:
    auc_s = f"{r['auc']:>7.4f}" if r['auc'] > 0 else "    N/A"
    print(f"  {r['model']:<35} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {auc_s}")
best = max(results, key=lambda x: x["f1"])
print("="*70)
print(f"  Best model (F1): {best['model']}  →  F1 = {best['f1']:.4f}")

# ── Comparison chart ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x        = range(len(results))
models   = [r["model"] for r in results]
metrics  = {
    "Accuracy":  [r["accuracy"]  for r in results],
    "Precision": [r["precision"] for r in results],
    "Recall":    [r["recall"]    for r in results],
    "F1-Score":  [r["f1"]        for r in results],
}
colors_m = ["#5c8de0","#e05c5c","#43a047","#fb8c00"]
width = 0.18
for idx, (metric, vals) in enumerate(metrics.items()):
    offset = (idx - 1.5) * width
    bars = ax.bar([xi + offset for xi in x], vals, width,
                  label=metric, color=colors_m[idx], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{bar.get_height():.3f}", ha='center', fontsize=7)
ax.set_xticks(list(x))
ax.set_xticklabels(models, rotation=10, ha="right")
ax.set_ylim(0.75, 1.06)
ax.set_ylabel("Score")
ax.set_title(
    "Model Comparison (Class Imbalance Handled via Weights)\n"
    "TF-IDF(5000) + f3 + f24 + f45 + f66 + f87",
    fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
cmp_path = os.path.join(PROJECT_ROOT, "reports", "figures", "model_comparison_phase3.png")
plt.savefig(cmp_path, dpi=150); plt.close()
print(f"  Chart saved: {cmp_path}")

# ── Save results CSV ──────────────────────────────────────────────────────────
res_path = os.path.join(PROJECT_ROOT, "reports", "model_results_phase3.csv")
pd.DataFrame(results).to_csv(res_path, index=False)
print(f"  Results CSV: {res_path}")

# ── M/R: Hapax Legomena ───────────────────────────────────────────────────────
print("\n[MapReduce Sim] Hapax Legomena Ratio...")
words_rdd   = df.select(text_col).rdd \
    .flatMap(lambda r: r[text_col].split() if r[text_col] else [])
word_counts = words_rdd.map(lambda w:(w,1)).reduceByKey(lambda a,b:a+b)
total_w     = word_counts.map(lambda x:x[1]).sum()
hapax_w     = word_counts.filter(lambda x:x[1]==1).count()
print(f"  Total tokens : {int(total_w):,}")
print(f"  Hapax words  : {hapax_w:,}")
print(f"  Hapax ratio  : {hapax_w/total_w:.4f}" if total_w else "  N/A")

print("\n" + "="*70)
print("  PHASE 3 COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"  Input          : features_extracted.parquet ({total:,} rows)")
print(f"  Imbalance      : {count_human:,} Human / {count_ai:,} AI  (ratio {ratio:.1f}:1)")
print(f"  Fix applied    : classWeight Human={weight_human:.3f}  AI={weight_ai:.3f}")
print(f"  Feature vector : TF-IDF(5000) + 5 assigned features = 5005 dims")
print(f"  Best model     : {best['model']}  F1={best['f1']:.4f}")
print(f"  Models saved   : {MODEL_DIR}")
print(f"  Results CSV    : {res_path}")
print("="*70)

spark.stop()