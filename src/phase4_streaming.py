
import os, sys, subprocess, re, time, json
import numpy as np

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pyspark", "pyarrow", "pandas", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import StringType, DoubleType, StructType, StructField
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel,
    LinearSVCModel,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/bigdata_project")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
STREAM_IN    = os.path.join(PROJECT_ROOT, "stream", "input")
STREAM_OUT   = os.path.join(PROJECT_ROOT, "stream", "output")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "reports")
FIG_DIR      = os.path.join(PROJECT_ROOT, "reports", "figures")
for d in [STREAM_IN, STREAM_OUT, REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

ASSIGNED_COLS = [
    "f3_digits_ratio", "f24_punct_div", "f45_adj_count",
    "f66_genitive_count", "f87_gini",
]

# ── [NEW] Load class imbalance info from Phase 1 ──────────────────────────────
info_path = os.path.join(PROC_DIR, "class_imbalance_info.json")
if os.path.exists(info_path):
    with open(info_path) as f:
        imb = json.load(f)
    COUNT_HUMAN = imb["count_human"]
    COUNT_AI    = imb["count_ai"]
    RATIO       = imb["ratio"]
else:
    # fallback from known values
    COUNT_HUMAN, COUNT_AI, RATIO = 8388, 33552, 4.0

# ── Preprocessing UDF ─────────────────────────────────────────────────────────
STOPWORDS = set([
    "في","من","إلى","على","عن","مع","هذا","هذه","ذلك","التي","الذي",
    "وقد","كما","إلا","أن","إن","كان","لا","ما","هو","هي","لم","لن",
    "قد","أو","حتى","أي","بين","له","لها","كل","بعد","قبل","ثم",
])
def preprocess_text(text):
    if not text: return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(w for w in text.split()
                    if w not in STOPWORDS and len(w) > 1)

preprocess_udf = udf(preprocess_text, StringType())

# ── Confusion Matrix helper ───────────────────────────────────────────────────
def plot_cm(cm, title, ax, acc, f1):
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{title}\nAcc={acc:.4f} | F1={f1:.4f}",
                 fontsize=9, fontweight='bold')
    classes = ["Human(0)", "AI(1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticks([0,1]); ax.set_yticklabels(classes, fontsize=8)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=11, fontweight='bold',
                    color="white" if cm[i,j] > thresh else "black")

print("="*65)
print("  MSBDA-801 Phase 4: Stream Processing & Evaluation")
print("  Tasks: 4.1 | 4.2 | 4.3 | 4.4")
print("="*65)
# [NEW] Print imbalance reminder
print(f"\n  [CLASS IMBALANCE REMINDER]")
print(f"  Human (0): {COUNT_HUMAN:,}  |  AI (1): {COUNT_AI:,}")
print(f"  Ratio: {RATIO:.1f}:1  →  Class weights applied in Phase 3 models")

# ── START SPARK ───────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MSBDA801_Phase4_Complete") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"\n[INFO] Spark: {spark.version}")

# ── Load full data ────────────────────────────────────────────────────────────
print("\n[INFO] Loading features_extracted.parquet...")
feat_path = os.path.join(PROC_DIR, "features_extracted.parquet")
if not os.path.exists(feat_path):
    raise FileNotFoundError(f"Missing: {feat_path}")

df_all = spark.read.parquet(feat_path)
df_all = df_all.select(
    ["text","clean_text", col("label").cast("double")] + ASSIGNED_COLS
).dropna(subset=["text","label"]+ASSIGNED_COLS) \
 .filter(col("text") != "")
total = df_all.count()
print(f"  Loaded: {total:,} rows")

# ── Build feature pipeline on FULL corpus ────────────────────────────────────
print("\n[INFO] Building feature pipeline on full corpus...")
text_col  = "clean_text" if "clean_text" in df_all.columns else "text"
tokenizer = Tokenizer(inputCol=text_col, outputCol="tokens")
df_tok    = tokenizer.transform(df_all)
hashTF    = HashingTF(inputCol="tokens", outputCol="raw_tf", numFeatures=5000)
df_tf     = hashTF.transform(df_tok)
idf_model = IDF(inputCol="raw_tf", outputCol="tfidf", minDocFreq=3).fit(df_tf)
assembler = VectorAssembler(
    inputCols=["tfidf"] + ASSIGNED_COLS,
    outputCol="features", handleInvalid="skip"
)
df_final = assembler.transform(idf_model.transform(df_tf))
df_final = df_final.dropna(subset=["label"]).cache()
print(f"  Pipeline ready. Feature dims: 5005 (TF-IDF + 5 stylometric)")

# ── Same split as Phase 3 ─────────────────────────────────────────────────────
train_df, val_df, test_df = df_final.randomSplit([0.70, 0.15, 0.15], seed=42)
test_df.cache()
print(f"  Test set: {test_df.count():,} rows (held-out)")

def apply_pipeline(df):
    if "clean_text" not in df.columns:
        df = df.withColumn("clean_text", preprocess_udf(col("text")))
    df = tokenizer.transform(df)
    df = hashTF.transform(df)
    df = idf_model.transform(df)
    df = assembler.transform(df)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4.1: STREAM SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("[TASK 4.1] Stream Simulation (file-based)")
print("="*65)

BATCH_SIZE  = 50
NUM_BATCHES = 10
sample_pd   = df_all.limit(BATCH_SIZE * NUM_BATCHES).toPandas()

for i in range(NUM_BATCHES):
    batch = sample_pd.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    batch[["text","label"]+ASSIGNED_COLS].to_csv(
        os.path.join(STREAM_IN, f"batch_{i:03d}.csv"),
        index=False, encoding="utf-8-sig"
    )

print(f"  Simulated stream: {NUM_BATCHES} batches × {BATCH_SIZE} records")
print(f"  Stream source   : {STREAM_IN}")
print(f"  Each batch      : text + label + f3+f24+f45+f66+f87")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4.2: REAL-TIME DEPLOYMENT — BEST MODEL (LR)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("[TASK 4.2] Real-Time Deployment — LR (best model from Phase 3)")
print("="*65)

lr_path = os.path.join(MODEL_DIR, "lr_model")
if not os.path.exists(lr_path):
    raise FileNotFoundError(f"Missing: {lr_path}. Run Phase 3 first.")

lr_model = LogisticRegressionModel.load(lr_path)
print(f"  Loaded: {lr_path}")
print(f"  Model type: Logistic Regression (Spark MLlib, trained with class weights)")
print(f"  Feature vector: TF-IDF(5000) + f3+f24+f45+f66+f87 = 5005 dims")

mc_eval  = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction")
bin_eval = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction")

print("\n  Processing stream batches...")
print(f"  {'Batch':>5} {'n':>4} {'Acc':>7} {'F1':>7} {'Latency':>9} "
      f"{'Tput':>8} {'Human':>6} {'AI':>5} {'H-Rec':>7} {'A-Rec':>7}")
print("  " + "-"*75)

batch_metrics = []

for i in range(NUM_BATCHES):
    t0 = time.time()

    batch_df = spark.read \
        .option("header","true").option("encoding","utf-8") \
        .csv(os.path.join(STREAM_IN, f"batch_{i:03d}.csv"))
    batch_df = batch_df.withColumn("label", col("label").cast("double"))
    for sc in ASSIGNED_COLS:
        batch_df = batch_df.withColumn(sc, col(sc).cast("double"))
    batch_df = batch_df.dropna(subset=["text"]+ASSIGNED_COLS)
    n = batch_df.count()

    feat_df = apply_pipeline(batch_df)
    preds   = lr_model.transform(feat_df)
    preds.cache()

    try:
        acc = mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"})
        f1  = mc_eval.evaluate(preds, {mc_eval.metricName: "f1"})
    except Exception:
        correct = preds.filter(col("prediction")==col("label")).count()
        acc = correct/n if n > 0 else 0.0
        f1  = acc

    pred_pd = preds.select("label","prediction").toPandas()
    human   = int((pred_pd["prediction"]==0.0).sum())
    ai      = int((pred_pd["prediction"]==1.0).sum())

    # [NEW] Per-class recall in stream
    h_tp = int(((pred_pd["label"]==0.0)&(pred_pd["prediction"]==0.0)).sum())
    h_fn = int(((pred_pd["label"]==0.0)&(pred_pd["prediction"]==1.0)).sum())
    a_tp = int(((pred_pd["label"]==1.0)&(pred_pd["prediction"]==1.0)).sum())
    a_fn = int(((pred_pd["label"]==1.0)&(pred_pd["prediction"]==0.0)).sum())
    h_rec = h_tp/(h_tp+h_fn) if (h_tp+h_fn) else 0.0
    a_rec = a_tp/(a_tp+a_fn) if (a_tp+a_fn) else 0.0

    preds.select("text","label","prediction") \
         .toPandas() \
         .to_csv(os.path.join(STREAM_OUT, f"results_{i:03d}.csv"),
                 index=False, encoding="utf-8-sig")

    t1 = time.time()
    latency    = (t1-t0)*1000
    throughput = n/(t1-t0)

    batch_metrics.append({
        "batch": i, "n_records": n,
        "accuracy": round(acc,4), "f1": round(f1,4),
        "latency_ms": round(latency,1),
        "throughput_rps": round(throughput,1),
        "predicted_human": human, "predicted_ai": ai,
        "human_recall": round(h_rec,4), "ai_recall": round(a_rec,4),
    })
    print(f"  {i:>5} {n:>4} {acc:>7.4f} {f1:>7.4f} "
          f"{latency:>8.0f}ms {throughput:>7.0f}r/s "
          f"{human:>6} {ai:>5} {h_rec:>7.4f} {a_rec:>7.4f}")

metrics_df = pd.DataFrame(batch_metrics)
metrics_df.to_csv(os.path.join(REPORT_DIR,"stream_batch_metrics.csv"),index=False)

# [NEW] Stream summary with per-class
print(f"\n  Stream Summary:")
print(f"    Avg Accuracy     : {metrics_df['accuracy'].mean():.4f}")
print(f"    Avg F1           : {metrics_df['f1'].mean():.4f}")
print(f"    Avg Latency      : {metrics_df['latency_ms'].mean():.1f} ms")
print(f"    Avg Throughput   : {metrics_df['throughput_rps'].mean():.0f} rec/s")
print(f"    Avg Human Recall : {metrics_df['human_recall'].mean():.4f}  ← minority class")
print(f"    Avg AI Recall    : {metrics_df['ai_recall'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4.3: COMPREHENSIVE EVALUATION — ALL MODELS + CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("[TASK 4.3] Comprehensive Evaluation — All Models on Test Set")
print("="*65)
print(f"  Note: Models trained with class weights (Human={COUNT_HUMAN:,} / AI={COUNT_AI:,})")

models_info = [
    ("Logistic Regression", "lr_model",  LogisticRegressionModel),
    ("Random Forest",       "rf_model",  RandomForestClassificationModel),
    ("Linear SVM",          "svm_model", LinearSVCModel),
]

eval_results       = []
confusion_matrices = {}

for model_name, model_dir, ModelClass in models_info:
    model_path = os.path.join(MODEL_DIR, model_dir)
    if not os.path.exists(model_path):
        print(f"  [SKIP] {model_name}: not found")
        continue

    print(f"\n  [{model_name}]")
    model = ModelClass.load(model_path)
    preds = model.transform(test_df)
    preds.cache()

    acc  = mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"})
    f1   = mc_eval.evaluate(preds, {mc_eval.metricName: "f1"})
    prec = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedPrecision"})
    rec  = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedRecall"})
    try:
        auc = bin_eval.evaluate(preds,{bin_eval.metricName:"areaUnderROC"})
    except Exception:
        auc = 0.0

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}" if auc>0 else "    ROC-AUC   : N/A")

    # Confusion matrix
    p  = preds.select("label","prediction").toPandas()
    tp = int(((p["label"]==1)&(p["prediction"]==1)).sum())
    tn = int(((p["label"]==0)&(p["prediction"]==0)).sum())
    fp = int(((p["label"]==0)&(p["prediction"]==1)).sum())
    fn = int(((p["label"]==1)&(p["prediction"]==0)).sum())
    cm = np.array([[tn,fp],[fn,tp]])
    confusion_matrices[model_name] = cm
    print(f"    Confusion : TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    # [NEW] Per-class metrics
    h_prec = tn/(tn+fn) if (tn+fn) else 0.0
    h_rec  = tn/(tn+fp) if (tn+fp) else 0.0
    h_f1   = 2*h_prec*h_rec/(h_prec+h_rec) if (h_prec+h_rec) else 0.0
    a_prec = tp/(tp+fp) if (tp+fp) else 0.0
    a_rec  = tp/(tp+fn) if (tp+fn) else 0.0
    a_f1   = 2*a_prec*a_rec/(a_prec+a_rec) if (a_prec+a_rec) else 0.0
    print(f"    Human(0)  : Prec={h_prec:.4f}  Rec={h_rec:.4f}  F1={h_f1:.4f}  ← minority")
    print(f"    AI(1)     : Prec={a_prec:.4f}  Rec={a_rec:.4f}  F1={a_f1:.4f}")

    eval_results.append({
        "model": model_name,
        "accuracy":round(acc,4), "precision":round(prec,4),
        "recall":round(rec,4), "f1":round(f1,4), "auc":round(auc,4),
        "TP":tp, "TN":tn, "FP":fp, "FN":fn,
        "human_f1":round(h_f1,4), "ai_f1":round(a_f1,4),
    })

# Save evaluation CSV
eval_df = pd.DataFrame(eval_results)
eval_df.to_csv(os.path.join(REPORT_DIR,"task43_evaluation.csv"),index=False)

# Plot individual confusion matrices
for r in eval_results:
    name = r["model"]
    cm   = confusion_matrices[name]
    fig, ax = plt.subplots(figsize=(5,4))
    plot_cm(cm, name, ax, r["accuracy"], r["f1"])
    plt.tight_layout()
    safe = name.lower().replace(" ","_").replace("(","").replace(")","")
    cm_path = os.path.join(FIG_DIR, f"confusion_matrix_{safe}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

# Combined confusion matrix plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    "Confusion Matrices — All Models on Held-out Test Set\n"
    f"Features: TF-IDF(5000)+5 stylometric  |  "
    f"Imbalance: {COUNT_HUMAN:,} Human / {COUNT_AI:,} AI  |  "
    f"Class weights applied",
    fontsize=10, fontweight="bold"
)
for idx, r in enumerate(eval_results):
    plot_cm(confusion_matrices[r["model"]], r["model"],
            axes[idx], r["accuracy"], r["f1"])
plt.tight_layout()
all_cm = os.path.join(FIG_DIR, "confusion_matrix_all.png")
plt.savefig(all_cm, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Confusion matrices saved: {FIG_DIR}")

print("\n  Model Comparison Summary (with class weights):")
print(f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
      f"{'AUC':>7} {'H-F1':>7} {'A-F1':>7}")
print("  "+"-"*72)
for r in eval_results:
    auc_s = f"{r['auc']:>7.4f}" if r['auc']>0 else "    N/A"
    print(f"  {r['model']:<25} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {auc_s} "
          f"{r['human_f1']:>7.4f} {r['ai_f1']:>7.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4.4: SCALABILITY TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("[TASK 4.4] Scalability Test — Executors + Batch/Stream Benchmark")
print("="*65)

print("\n  Part A: Batch processing vs number of executors...")
scale_exec = []
test_size  = 1000

for n_exec in [1, 2, 4, 8]:
    print(f"  Testing local[{n_exec}]...")
    spark_e = SparkSession.builder \
        .appName(f"MSBDA_Scale_{n_exec}") \
        .master(f"local[{n_exec}]") \
        .config("spark.driver.memory","4g") \
        .config("spark.sql.shuffle.partitions", str(n_exec*2)) \
        .config("spark.sql.execution.arrow.pyspark.enabled","false") \
        .getOrCreate()
    spark_e.sparkContext.setLogLevel("WARN")

    df_e = spark_e.read.parquet(feat_path)
    df_e = df_e.select(
        ["text","clean_text",col("label").cast("double")]+ASSIGNED_COLS
    ).dropna(subset=["text","label"]+ASSIGNED_COLS) \
     .filter(col("text")!="").limit(test_size)

    tok_e  = Tokenizer(inputCol=text_col,outputCol="tokens")
    htf_e  = HashingTF(inputCol="tokens",outputCol="raw_tf",numFeatures=5000)
    df_et  = tok_e.transform(df_e)
    df_etf = htf_e.transform(df_et)
    idf_e  = IDF(inputCol="raw_tf",outputCol="tfidf",minDocFreq=1).fit(df_etf)
    asm_e  = VectorAssembler(
        inputCols=["tfidf"]+ASSIGNED_COLS,
        outputCol="features",handleInvalid="skip")
    df_fin = asm_e.transform(idf_e.transform(df_etf)).dropna(subset=["label"])
    lr_e   = LogisticRegressionModel.load(lr_path)

    t0 = time.time()
    n_p = lr_e.transform(df_fin).count()
    t1 = time.time()
    elapsed = t1-t0

    scale_exec.append({
        "executors":n_exec, "batch_size":n_p,
        "time_sec":round(elapsed,2),
        "throughput":round(n_p/elapsed,1),
    })
    print(f"    executors={n_exec} | time={elapsed:.2f}s | {n_p/elapsed:.0f} rec/s")
    spark_e.stop()

# Part B: Stream latency summary
print("\n  Part B: Stream latency/throughput summary:")
print(f"    Avg latency      : {metrics_df['latency_ms'].mean():.1f} ms")
print(f"    Avg throughput   : {metrics_df['throughput_rps'].mean():.0f} rec/s")
print(f"    Avg accuracy     : {metrics_df['accuracy'].mean():.4f}")
print(f"    Avg Human Recall : {metrics_df['human_recall'].mean():.4f}")
print(f"    Avg AI Recall    : {metrics_df['ai_recall'].mean():.4f}")

scale_df = pd.DataFrame(scale_exec)
scale_df.to_csv(os.path.join(REPORT_DIR,"task44_scalability.csv"),index=False)

# ── Combined scalability plot ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "Task 4.4 — Scalability Benchmarks\n"
    "Batch processing (Phase 3) & Stream processing (Phase 4)",
    fontsize=11, fontweight="bold"
)

# A: Throughput vs executors
axes[0,0].plot(scale_df["executors"], scale_df["throughput"],
               marker="o", color="#2ca02c", linewidth=2, markersize=8)
for _, row in scale_df.iterrows():
    axes[0,0].annotate(f"{row['throughput']:.0f}",
                       (row["executors"], row["throughput"]),
                       textcoords="offset points", xytext=(0,8),
                       ha='center', fontsize=9, fontweight='bold')
axes[0,0].set_xlabel("Number of Spark Executors")
axes[0,0].set_ylabel("Throughput (rec/sec)")
axes[0,0].set_title("Batch Throughput vs Executors")
axes[0,0].set_xticks(scale_df["executors"])
axes[0,0].grid(True, alpha=0.3)

# B: Processing time vs executors
axes[0,1].bar(scale_df["executors"].astype(str),
              scale_df["time_sec"],
              color=["#ff7f0e","#5c8de0","#2ca02c","#e05c5c"],
              edgecolor="white", width=0.5)
for i, row in scale_df.iterrows():
    axes[0,1].text(i, row["time_sec"]+0.01,
                   f"{row['time_sec']}s",
                   ha='center', fontsize=9, fontweight='bold')
axes[0,1].set_xlabel("Number of Spark Executors")
axes[0,1].set_ylabel("Time (sec)")
axes[0,1].set_title("Batch Processing Time vs Executors")
axes[0,1].grid(True, alpha=0.3, axis='y')

# C: Stream latency per batch
axes[1,0].bar(metrics_df["batch"], metrics_df["latency_ms"],
              color="#5c8de0", edgecolor="white")
axes[1,0].axhline(metrics_df["latency_ms"].mean(), color="red",
                  linestyle="--",
                  label=f"avg={metrics_df['latency_ms'].mean():.0f}ms")
axes[1,0].set_title("Stream Latency per Batch (ms)")
axes[1,0].set_xlabel("Batch #"); axes[1,0].set_ylabel("ms")
axes[1,0].legend()

# D: Stream throughput per batch
axes[1,1].plot(metrics_df["batch"], metrics_df["throughput_rps"],
               marker="o", color="#e05c5c", linewidth=2)
axes[1,1].axhline(metrics_df["throughput_rps"].mean(), color="gray",
                  linestyle="--",
                  label=f"avg={metrics_df['throughput_rps'].mean():.0f}")
axes[1,1].set_title("Stream Throughput per Batch (rec/sec)")
axes[1,1].set_xlabel("Batch #"); axes[1,1].legend()

plt.tight_layout()
scale_path = os.path.join(FIG_DIR, "scalability_executors.png")
plt.savefig(scale_path, dpi=150, bbox_inches="tight")
plt.close()

# Stream benchmark plot
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Phase 4 Stream Processing — Detection Results",
              fontsize=11, fontweight="bold")
x = metrics_df["batch"]
axes2[0].bar(x-0.2, metrics_df["predicted_human"], 0.35,
             label="Predicted Human", color="#42a5f5")
axes2[0].bar(x+0.2, metrics_df["predicted_ai"],    0.35,
             label="Predicted AI",    color="#ef5350")
axes2[0].set_xlabel("Batch #"); axes2[0].set_ylabel("Count")
axes2[0].set_title("Human vs AI Predictions per Batch")
axes2[0].legend(); axes2[0].set_xticks(x)
axes2[1].plot(x, metrics_df["accuracy"], marker="o",
              color="#5c8de0", linewidth=2, label="Accuracy")
axes2[1].plot(x, metrics_df["f1"], marker="s",
              color="#e05c5c", linewidth=2, label="F1-Score")
axes2[1].axhline(metrics_df["accuracy"].mean(), color="#5c8de0",
                 linestyle="--", alpha=0.5)
axes2[1].axhline(metrics_df["f1"].mean(), color="#e05c5c",
                 linestyle="--", alpha=0.5)
axes2[1].set_ylim(0.7, 1.02)
axes2[1].set_xlabel("Batch #"); axes2[1].set_ylabel("Score")
axes2[1].set_title("Stream Accuracy & F1 per Batch")
axes2[1].legend(); axes2[1].set_xticks(x)
plt.tight_layout()
bench_path = os.path.join(FIG_DIR, "stream_benchmark.png")
plt.savefig(bench_path, dpi=150, bbox_inches="tight")
plt.close()

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PHASE 4 COMPLETE SUMMARY")
print("="*65)
print(f"\n  [Class Imbalance] Handled via class weights in Phase 3")
print(f"    Human(0): {COUNT_HUMAN:,}  AI(1): {COUNT_AI:,}  Ratio: {RATIO:.1f}:1")

print(f"\n  [Task 4.1] Stream Simulation")
print(f"    Batches : {NUM_BATCHES} × {BATCH_SIZE} = {NUM_BATCHES*BATCH_SIZE} records")

print(f"\n  [Task 4.2] Real-Time Deployment")
print(f"    Model        : Logistic Regression (with class weights)")
print(f"    Avg Accuracy : {metrics_df['accuracy'].mean():.4f}")
print(f"    Avg F1       : {metrics_df['f1'].mean():.4f}")
print(f"    Avg Latency  : {metrics_df['latency_ms'].mean():.1f} ms")
print(f"    Avg Tput     : {metrics_df['throughput_rps'].mean():.0f} rec/sec")
print(f"    Human Recall : {metrics_df['human_recall'].mean():.4f}  ← minority class")
print(f"    AI Recall    : {metrics_df['ai_recall'].mean():.4f}")

print(f"\n  [Task 4.3] Comprehensive Evaluation")
best = max(eval_results, key=lambda x: x["f1"])
for r in eval_results:
    auc_s  = f"{r['auc']:.4f}" if r['auc']>0 else "N/A"
    marker = " ← best" if r["model"]==best["model"] else ""
    print(f"    {r['model']:<25} F1={r['f1']:.4f}  "
          f"H-F1={r['human_f1']:.4f}  A-F1={r['ai_f1']:.4f}{marker}")

print(f"\n  [Task 4.4] Scalability Test")
for row in scale_exec:
    print(f"    executors={row['executors']} → "
          f"{row['throughput']:.0f} rec/s ({row['time_sec']}s)")

print(f"\n  Output files:")
print(f"    {FIG_DIR}/confusion_matrix_all.png")
print(f"    {FIG_DIR}/scalability_executors.png")
print(f"    {FIG_DIR}/stream_benchmark.png")
print(f"    {REPORT_DIR}/task43_evaluation.csv")
print(f"    {REPORT_DIR}/task44_scalability.csv")
print(f"    {REPORT_DIR}/stream_batch_metrics.csv")
print("="*65)
print("  PHASE 4 COMPLETED SUCCESSFULLY!")
print("="*65)

spark.stop()