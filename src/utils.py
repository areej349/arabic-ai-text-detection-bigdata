"""
=============================================================
MSBDA-801 Big Data Analytics
utils.py — Shared Helper Functions
=============================================================
Reusable utilities used across the pipeline:
  - Arabic text preprocessing
  - Stylometric feature extraction (f3, f24, f45, f66, f87)
  - Spark session factory
  - Evaluation helpers
=============================================================
"""

import re
import os
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Arabic NLP helpers
# ─────────────────────────────────────────────────────────────────────────────

ARABIC_STOPWORDS = set([
    "في","من","إلى","على","عن","مع","هذا","هذه","ذلك","التي","الذي",
    "وقد","كما","إلا","أن","إن","كان","لا","ما","هو","هي","لم","لن",
    "قد","أو","حتى","أي","بين","له","لها","بما","عند","كل","بعد","قبل",
    "ثم","منذ","خلال","حول","نحو","غير","بل","لكن","وهو","وهي","فإن",
    "وأن","أما","إذا","كانت","وكان","أيضا","فقد","وفي","ومن","وعلى",
])

ARABIC_PUNCT  = set('،؛؟!:.,"\'()[]{}')
GENITIVE_PREP = re.compile(r'\b(في|من|إلى|على|عن|مع)\s')
ADJ_PATTERN   = re.compile(r'ال\w+ي\b')
DIGIT_RE      = re.compile(r'[٠-٩0-9]')
DIACRITIC_RE  = re.compile(r'[\u064B-\u065F\u0670]')


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text:
    - Remove diacritics (tashkeel)
    - Normalize Alef variants → ا
    - Normalize Taa marbuta → ه
    - Normalize Yaa variants → ي
    - Remove tatweel (elongation)
    - Remove non-Arabic characters
    """
    if not text:
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)   # diacritics
    text = re.sub(r'[إأآٱ]', 'ا', text)                 # alef
    text = re.sub(r'ة', 'ه', text)                       # taa marbuta
    text = re.sub(r'ى', 'ي', text)                       # alef maqsura
    text = re.sub(r'ـ+', '', text)                       # tatweel
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)      # non-Arabic
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove Arabic stopwords from text."""
    if not text:
        return ""
    return " ".join(
        w for w in text.split()
        if w not in ARABIC_STOPWORDS and len(w) > 1
    )


def light_stem(word: str) -> str:
    """Simple Arabic light stemmer (prefix/suffix removal)."""
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


def full_preprocess(text: str) -> str:
    """Full Arabic preprocessing pipeline: normalize → stopwords → stem."""
    t = normalize_arabic(text)
    t = remove_stopwords(t)
    t = " ".join(light_stem(w) for w in t.split())
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Stylometric Feature Extraction (5 assigned features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_f3(text: str) -> float:
    """f3: Number of digits / C (total characters)."""
    if not text:
        return 0.0
    C = len(text)
    return len(DIGIT_RE.findall(text)) / C if C > 0 else 0.0


def compute_f24(text: str) -> float:
    """f24: Number of different punctuation signs / C."""
    if not text:
        return 0.0
    C = len(text)
    unique_punct = {ch for ch in text if ch in ARABIC_PUNCT}
    return len(unique_punct) / C if C > 0 else 0.0


def compute_f45(text: str) -> float:
    """f45: Number of adjectives (ال...ي pattern, approximation)."""
    if not text:
        return 0.0
    return float(len(ADJ_PATTERN.findall(text)))


def compute_f66(text: str) -> float:
    """f66: Number of genitives (words after prepositions)."""
    if not text:
        return 0.0
    return float(len(GENITIVE_PREP.findall(text)))


def compute_f87(text: str) -> float:
    """
    f87: Gini Coefficient of Word Frequencies.
    Measures inequality in word distribution.
    0 = perfectly equal, 1 = one word dominates.
    """
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
    cum = sum((i + 1) * f for i, f in enumerate(freqs))
    return (2 * cum) / (n_t * tot_f) - (n_t + 1) / n_t


def extract_all_features(text: str) -> dict:
    """
    Extract all 5 assigned stylometric features from text.
    Returns a dictionary with feature names and values.
    """
    return {
        "f3_digits_ratio":    compute_f3(text),
        "f24_punct_div":      compute_f24(text),
        "f45_adj_count":      compute_f45(text),
        "f66_genitive_count": compute_f66(text),
        "f87_gini":           compute_f87(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spark Session Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_spark_session(app_name: str = "MSBDA801",
                      memory: str = "4g",
                      partitions: int = 8):
    """
    Create or retrieve an existing SparkSession.

    Parameters
    ----------
    app_name   : Spark application name
    memory     : Driver memory (e.g. '4g', '6g')
    partitions : spark.sql.shuffle.partitions
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", memory) \
        .config("spark.sql.shuffle.partitions", str(partitions)) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, test_df, model_name: str) -> dict:
    """
    Evaluate a Spark MLlib model on test data.
    Returns dict with accuracy, precision, recall, f1, auc.
    """
    from pyspark.ml.evaluation import (
        MulticlassClassificationEvaluator,
        BinaryClassificationEvaluator,
    )
    mc_eval  = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction")
    bin_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction")

    preds = model.transform(test_df)
    acc   = mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"})
    f1    = mc_eval.evaluate(preds, {mc_eval.metricName: "f1"})
    prec  = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedPrecision"})
    rec   = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedRecall"})
    try:
        auc = bin_eval.evaluate(preds, {bin_eval.metricName: "areaUnderROC"})
    except Exception:
        auc = 0.0

    print(f"\n  [{model_name}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}" if auc > 0 else "    ROC-AUC   : N/A")

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
    }


def compute_confusion_matrix(model, test_df) -> dict:
    """
    Compute confusion matrix values (TP, TN, FP, FN).
    Returns dict with all four values.
    """
    preds = model.transform(test_df)
    p     = preds.select("label", "prediction").toPandas()
    tp = int(((p["label"] == 1) & (p["prediction"] == 1)).sum())
    tn = int(((p["label"] == 0) & (p["prediction"] == 0)).sum())
    fp = int(((p["label"] == 0) & (p["prediction"] == 1)).sum())
    fn = int(((p["label"] == 1) & (p["prediction"] == 0)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ─────────────────────────────────────────────────────────────────────────────
# Path Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_project_paths(root: str = None) -> dict:
    """Return standard project directory paths."""
    if root is None:
        root = os.path.expanduser("~/bigdata_project")
    return {
        "root":          root,
        "data_raw":      os.path.join(root, "data", "raw"),
        "data_processed":os.path.join(root, "data", "processed"),
        "models":        os.path.join(root, "models"),
        "reports":       os.path.join(root, "reports"),
        "figures":       os.path.join(root, "reports", "figures"),
        "stream_in":     os.path.join(root, "stream", "input"),
        "stream_out":    os.path.join(root, "stream", "output"),
    }


if __name__ == "__main__":
    print("utils.py — testing helper functions...")
    sample = "درست في عام 2023 وحصلت على 95 درجة في البحث العلمي التقني."
    feats  = extract_all_features(sample)
    for k, v in feats.items():
        print(f"  {k:<25}: {v:.6f}")
    print("All helpers working correctly.")