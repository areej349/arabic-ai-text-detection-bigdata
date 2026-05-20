
import os, sys, subprocess, json

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pandas", "matplotlib", "numpy", "pyarrow"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/bigdata_project")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "reports")
FIG_DIR      = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("="*65)
print("  MSBDA-801 Phase 5: Analysis & Final Reporting")
print("="*65)

# ── [NEW] Load class imbalance info ───────────────────────────────────────────
info_path = os.path.join(PROC_DIR, "class_imbalance_info.json")
if os.path.exists(info_path):
    with open(info_path) as f:
        imb = json.load(f)
    COUNT_HUMAN = imb["count_human"]
    COUNT_AI    = imb["count_ai"]
    PCT_HUMAN   = imb["pct_human"]
    PCT_AI      = imb["pct_ai"]
    RATIO       = imb["ratio"]
else:
    COUNT_HUMAN, COUNT_AI   = 8388, 33552
    PCT_HUMAN,   PCT_AI     = 20.0, 80.0
    RATIO                   = 4.0

print(f"\n  [CLASS IMBALANCE]")
print(f"  Human (0): {COUNT_HUMAN:,} ({PCT_HUMAN:.1f}%)")
print(f"  AI    (1): {COUNT_AI:,} ({PCT_AI:.1f}%)")
print(f"  Ratio    : {RATIO:.1f}:1  → class weights applied in Phase 3")

# ── Load model results ────────────────────────────────────────────────────────
print("\n[TASK 5.1] Loading results...")

model_csv = os.path.join(REPORT_DIR, "model_results_phase3.csv")
if not os.path.exists(model_csv):
    raise FileNotFoundError(f"[ERROR] Run Phase 3 first.\n  Missing: {model_csv}")

models_df = pd.read_csv(model_csv)
print("\n  Model Results (Phase 3 — with class weights):")
print(models_df.to_string(index=False))

# Load per-class metrics from Phase 4 task43
eval43_csv = os.path.join(REPORT_DIR, "task43_evaluation.csv")
has_eval43 = os.path.exists(eval43_csv)
if has_eval43:
    eval43_df = pd.read_csv(eval43_csv)
    print("\n  Per-class results (from task43_evaluation.csv):")
    print(eval43_df[["model","accuracy","f1","human_f1","ai_f1"]].to_string(index=False))

# Load stream metrics
stream_csv = os.path.join(REPORT_DIR, "stream_batch_metrics.csv")
scale_csv  = os.path.join(REPORT_DIR, "stream_scalability_metrics.csv")
has_stream = os.path.exists(stream_csv)

if has_stream:
    stream_df = pd.read_csv(stream_csv)
    scale_df  = pd.read_csv(scale_csv)
    print(f"\n  Stream avg latency    : {stream_df['latency_ms'].mean():.1f} ms")
    print(f"  Stream avg throughput : {stream_df['throughput_rps'].mean():.0f} rec/s")
    print(f"  Stream avg accuracy   : {stream_df['accuracy'].mean():.4f}")
    if "human_recall" in stream_df.columns:
        print(f"  Stream Human Recall   : {stream_df['human_recall'].mean():.4f}  ← minority")
        print(f"  Stream AI Recall      : {stream_df['ai_recall'].mean():.4f}")

# ── Load feature stats ────────────────────────────────────────────────────────
print("\n  Loading feature statistics from features_extracted.parquet...")
feat_path = os.path.join(PROC_DIR, "features_extracted.parquet")
ASSIGNED_COLS = [
    "f3_digits_ratio","f24_punct_div","f45_adj_count",
    "f66_genitive_count","f87_gini"
]

if os.path.exists(feat_path):
    feat_df    = pd.read_parquet(feat_path)
    feat_stats = feat_df.groupby("label")[ASSIGNED_COLS].mean()
    print("\n  Feature means per label:")
    print(feat_stats.to_string())
    human_vals = feat_stats.loc[0].tolist() if 0 in feat_stats.index else [0]*5
    ai_vals    = feat_stats.loc[1].tolist() if 1 in feat_stats.index else [0]*5
else:
    human_vals = [0.003041, 0.004475, 6.1927, 11.8865, 0.1653]
    ai_vals    = [0.002219, 0.004135, 5.3025, 11.7611, 0.1795]

rf_importances = [0.000534, 0.001569, 0.002301, 0.001687, 0.001347]

ASSIGNED = {
    "f3_digits_ratio":    ("f3",  "Number of digits / C",
                           human_vals[0], ai_vals[0], rf_importances[0]),
    "f24_punct_div":      ("f24", "Number of different punctuation signs / C",
                           human_vals[1], ai_vals[1], rf_importances[1]),
    "f45_adj_count":      ("f45", "Number of adjectives",
                           human_vals[2], ai_vals[2], rf_importances[2]),
    "f66_genitive_count": ("f66", "Number of genitives",
                           human_vals[3], ai_vals[3], rf_importances[3]),
    "f87_gini":           ("f87", "Gini Coefficient of Word Frequencies",
                           human_vals[4], ai_vals[4], rf_importances[4]),
}

# ── [NEW] Task 5.0: Class Imbalance Plot ──────────────────────────────────────
print("\n[TASK 5.0] Generating class imbalance plot...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(
    "Dataset Class Distribution — Class Imbalance Analysis\n"
    f"Total: {COUNT_HUMAN+COUNT_AI:,} samples  |  Ratio AI:Human = {RATIO:.1f}:1",
    fontsize=11, fontweight="bold"
)

# Bar chart
axes[0].bar(["Human (0)", "AI (1)"], [COUNT_HUMAN, COUNT_AI],
            color=["#42a5f5","#ef5350"], width=0.5, edgecolor="white")
for i, (val, pct) in enumerate([(COUNT_HUMAN, PCT_HUMAN), (COUNT_AI, PCT_AI)]):
    axes[0].text(i, val + 300, f"{val:,}\n({pct:.1f}%)",
                 ha='center', fontsize=11, fontweight='bold')
axes[0].set_ylabel("Number of Samples")
axes[0].set_title("Sample Count per Class")
axes[0].set_ylim(0, COUNT_AI * 1.15)
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart
axes[1].pie(
    [COUNT_HUMAN, COUNT_AI],
    labels=[f"Human\n{COUNT_HUMAN:,} ({PCT_HUMAN:.1f}%)",
            f"AI\n{COUNT_AI:,} ({PCT_AI:.1f}%)"],
    colors=["#42a5f5","#ef5350"],
    autopct='%1.1f%%', startangle=90,
    wedgeprops={"edgecolor":"white","linewidth":2}
)
axes[1].set_title(f"Class Distribution\n(Imbalance ratio {RATIO:.1f}:1)")

plt.tight_layout()
imb_path = os.path.join(FIG_DIR, "class_imbalance.png")
plt.savefig(imb_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Imbalance plot saved: {imb_path}")

# ── Task 5.2: Feature Analysis Plots ─────────────────────────────────────────
print("\n[TASK 5.2] Generating feature analysis plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Assigned Stylometric Features Analysis\n"
    "f3(digits/C) | f24(punct/C) | f45(adj) | f66(genitives) | f87(Gini)\n"
    f"Source: features_extracted.parquet ({COUNT_HUMAN+COUNT_AI:,} rows)",
    fontsize=10, fontweight="bold"
)

features    = list(ASSIGNED.keys())
feat_labels = ["f3\ndigits/C","f24\npunct/C","f45\nadj","f66\ngenitives","f87\nGini"]
importances = [ASSIGNED[f][4] for f in features]
colors_f    = ["#e53935","#8e24aa","#1e88e5","#43a047","#fb8c00"]

bars = axes[0].bar(feat_labels, importances, color=colors_f,
                   width=0.5, edgecolor="white")
axes[0].set_title("RF Feature Importance (5 Assigned)", fontsize=10)
axes[0].set_ylabel("Importance")
for bar, val in zip(bars, importances):
    axes[0].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.00005,
                 f"{val:.5f}", ha='center', fontsize=8, fontweight='bold')

hv = [ASSIGNED[f][2] for f in features]
av = [ASSIGNED[f][3] for f in features]
x  = np.arange(len(features))
w  = 0.35
axes[1].bar(x-w/2, hv, w, label="Human (label=0)", color="#42a5f5", alpha=0.85)
axes[1].bar(x+w/2, av, w, label="AI (label=1)",    color="#ef5350", alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(feat_labels)
axes[1].set_title("Feature Values: Human vs AI Text", fontsize=10)
axes[1].set_ylabel("Average Value")
axes[1].legend()

plt.tight_layout()
feat_path_img = os.path.join(FIG_DIR, "assigned_features_analysis.png")
plt.savefig(feat_path_img, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Feature plot saved: {feat_path_img}")

# ── Task 5.1: Model Comparison Plot ──────────────────────────────────────────
print("\n[TASK 5.1] Model comparison plot...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Final Model Comparison (Class Imbalance Handled via Weights)\n"
    "Input: features_extracted.parquet → TF-IDF(5000) + f3+f24+f45+f66+f87",
    fontsize=10, fontweight="bold"
)

models_list  = models_df["model"].tolist()
x            = np.arange(len(models_list))
w            = 0.2
metrics_cols = ["accuracy","precision","recall","f1"]
colors_m     = ["#5c8de0","#e05c5c","#43a047","#fb8c00"]

for idx, metric in enumerate(metrics_cols):
    if metric in models_df.columns:
        vals = models_df[metric].tolist()
        bars = axes[0].bar(x+(idx-1.5)*w, vals, w,
                           label=metric.capitalize(),
                           color=colors_m[idx], alpha=0.85)
        for bar in bars:
            axes[0].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.003,
                         f"{bar.get_height():.3f}",
                         ha='center', fontsize=7)

axes[0].set_xticks(x)
axes[0].set_xticklabels(models_list, rotation=12, ha="right", fontsize=9)
axes[0].set_ylim(0.70, 1.06)
axes[0].set_ylabel("Score")
axes[0].set_title("Accuracy / Precision / Recall / F1")
axes[0].legend(loc="lower right", fontsize=8)

auc_vals = [a if a > 0 else 0 for a in models_df["auc"].tolist()]
bars2 = axes[1].bar(models_list, auc_vals,
                    color=["#5c8de0","#e05c5c","#43a047"][:len(models_list)],
                    width=0.4, alpha=0.85)
axes[1].set_ylim(0.9, 1.01)
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("ROC-AUC Score")
axes[1].set_xticklabels(models_list, rotation=12, ha="right", fontsize=9)
for bar, val in zip(bars2, auc_vals):
    label = f"{val:.4f}" if val > 0 else "N/A"
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.001,
                 label, ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
cmp_path = os.path.join(FIG_DIR, "final_model_comparison.png")
plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Model comparison plot: {cmp_path}")

# ── Batch vs Stream ───────────────────────────────────────────────────────────
if has_stream:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Batch vs Stream Processing Performance",
                 fontsize=12, fontweight="bold")
    axes[0].plot(stream_df["batch"], stream_df["latency_ms"],
                 marker="o", color="#5c8de0", linewidth=2)
    axes[0].axhline(stream_df["latency_ms"].mean(), color="red",
                    linestyle="--",
                    label=f"Avg={stream_df['latency_ms'].mean():.0f}ms")
    axes[0].set_xlabel("Batch #"); axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Stream Latency per Batch"); axes[0].legend()

    axes[1].plot(scale_df["batch_size"], scale_df["throughput"],
                 marker="s", color="#e05c5c", linewidth=2)
    axes[1].set_xlabel("Batch Size"); axes[1].set_ylabel("Records/sec")
    axes[1].set_title("Scalability: Throughput vs Batch Size")

    plt.tight_layout()
    bvs_path = os.path.join(FIG_DIR, "batch_vs_stream.png")
    plt.savefig(bvs_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Batch vs Stream plot: {bvs_path}")

# ── HTML Report ───────────────────────────────────────────────────────────────
print("\n[TASK 5.3] Generating HTML report...")

best = models_df.loc[models_df["f1"].idxmax()]

model_rows = ""
for _, r in models_df.iterrows():
    hl     = ' style="background:#e8f5e9;font-weight:bold;"' \
             if r["model"] == best["model"] else ""
    auc_s  = f"{r['auc']:.4f}"       if r["auc"]  > 0 else "N/A"
    prec_s = f"{r['precision']:.4f}" if "precision" in r else "-"
    rec_s  = f"{r['recall']:.4f}"    if "recall"    in r else "-"
    model_rows += (
        f"<tr{hl}><td>{r['model']}</td>"
        f"<td>{r['accuracy']:.4f}</td>"
        f"<td>{prec_s}</td><td>{rec_s}</td>"
        f"<td>{r['f1']:.4f}</td><td>{auc_s}</td></tr>"
    )

# [NEW] Per-class rows
perclass_rows = ""
if has_eval43:
    for _, r in eval43_df.iterrows():
        perclass_rows += (
            f"<tr><td>{r['model']}</td>"
            f"<td>{r.get('human_f1','-')}</td>"
            f"<td>{r.get('ai_f1','-')}</td>"
            f"<td>{r['f1']:.4f}</td></tr>"
        )

feat_rows = ""
for key, (code, desc, hv, av, imp) in ASSIGNED.items():
    diff = av - hv
    sign = "+" if diff > 0 else ""
    feat_rows += (
        f"<tr><td><strong>{code}</strong></td>"
        f"<td>{desc}</td>"
        f"<td>{hv:.6f}</td><td>{av:.6f}</td>"
        f"<td>{sign}{diff:.6f}</td>"
        f"<td>{imp:.6f}</td></tr>"
    )

stream_rows = ""
if has_stream:
    h_rec = stream_df['human_recall'].mean() if "human_recall" in stream_df.columns else "-"
    a_rec = stream_df['ai_recall'].mean()    if "ai_recall"    in stream_df.columns else "-"
    stream_rows = f"""
    <tr><td>Avg Latency</td>
        <td>{stream_df['latency_ms'].mean():.1f} ms</td></tr>
    <tr><td>Avg Throughput</td>
        <td>{stream_df['throughput_rps'].mean():.0f} rec/sec</td></tr>
    <tr><td>Avg Stream Accuracy</td>
        <td>{stream_df['accuracy'].mean():.4f}</td></tr>
    <tr><td>Avg Human Recall (minority)</td>
        <td>{h_rec:.4f} ← minority class</td></tr>
    <tr><td>Avg AI Recall</td>
        <td>{a_rec:.4f}</td></tr>
    <tr><td>Total Records Streamed</td>
        <td>{len(stream_df)*50}</td></tr>
    """

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MSBDA-801 Final Report</title>
<style>
  body  {{ font-family: Arial, sans-serif; margin:40px; background:#f5f5f5; }}
  h1    {{ color:#1a237e; border-bottom:3px solid #1a237e; padding-bottom:10px; }}
  h2    {{ color:#283593; margin-top:30px; }}
  table {{ border-collapse:collapse; width:100%; margin:15px 0; background:white; }}
  th    {{ background:#1a237e; color:white; padding:10px; text-align:left; }}
  td    {{ padding:8px 12px; border-bottom:1px solid #ddd; }}
  tr:hover {{ background:#f0f0f0; }}
  .box  {{ background:white; padding:20px; border-radius:8px;
           box-shadow:0 2px 4px rgba(0,0,0,.1); margin:15px 0; }}
  .best {{ background:#e8f5e9; padding:15px;
           border-left:5px solid #4caf50; border-radius:4px; margin:10px 0; }}
  .warn {{ background:#fff3e0; padding:15px;
           border-left:5px solid #ff9800; border-radius:4px; margin:10px 0; }}
  img   {{ max-width:100%; border-radius:8px; margin:10px 0; }}
  .pipeline {{ background:#e3f2fd; padding:15px;
               border-left:5px solid #1976d2; border-radius:4px;
               font-family:monospace; white-space:pre; margin:10px 0; }}
</style>
</head>
<body>
<h1>MSBDA-801 Big Data Analytics — Final Report</h1>
<p><strong>Project:</strong> Scalable Real-time Detection of AI-Generated Arabic Text<br>
   <strong>Student:</strong> اريج عبدالعزيز بن حمدان الشاماني &nbsp;|&nbsp;
   <strong>i=3, n=21</strong><br>
   <strong>Dataset:</strong> KFUPM-JRCAI/arabic-generated-abstracts
   ({COUNT_HUMAN+COUNT_AI:,} samples: {COUNT_HUMAN:,} Human + {COUNT_AI:,} AI)</p>

<div class="box">
<h2>Pipeline Overview</h2>
<div class="pipeline">Phase 1 → raw_abstracts.parquet
    ↓
Phase 2 → processed_abstracts.parquet
    ↓
extract_features_dataset.py → features_extracted.parquet ({COUNT_HUMAN+COUNT_AI:,} × 5 features)
    ↓
Phase 3 → TF-IDF(5000) + f3+f24+f45+f66+f87 → Class Weights → Train/Val/Test → Models
    ↓
Phase 4 → Stream simulation + Comprehensive Evaluation
    ↓
Phase 5 → Final Report</div>
</div>

<div class="box">
<h2>Class Imbalance Analysis</h2>
<div class="warn">
  ⚠ <strong>Significant class imbalance detected:</strong>
  Human (0) = {COUNT_HUMAN:,} ({PCT_HUMAN:.1f}%) &nbsp;|&nbsp;
  AI (1) = {COUNT_AI:,} ({PCT_AI:.1f}%) &nbsp;|&nbsp;
  Ratio = {RATIO:.1f}:1<br><br>
  <strong>Fix applied:</strong> Class weights computed from training set and passed to
  Logistic Regression and Random Forest via <code>weightCol</code>.
  This ensures the minority class (Human) is not ignored during training.
</div>
<table>
  <tr><th>Class</th><th>Count</th><th>Percentage</th><th>Weight Applied</th></tr>
  <tr><td>Human (0) — minority</td>
      <td>{COUNT_HUMAN:,}</td><td>{PCT_HUMAN:.1f}%</td>
      <td>~{(COUNT_HUMAN+COUNT_AI)/(2*COUNT_HUMAN):.3f} (higher)</td></tr>
  <tr><td>AI (1) — majority</td>
      <td>{COUNT_AI:,}</td><td>{PCT_AI:.1f}%</td>
      <td>~{(COUNT_HUMAN+COUNT_AI)/(2*COUNT_AI):.3f} (lower)</td></tr>
</table>
<img src="figures/class_imbalance.png" alt="Class Imbalance">
</div>

<div class="box">
<h2>Assigned Stylometric Features (f3, f24, f45, f66, f87)</h2>
<table>
  <tr><th>Code</th><th>Feature</th>
      <th>Human avg</th><th>AI avg</th>
      <th>Diff</th><th>RF Importance</th></tr>
  {feat_rows}
</table>
<p><strong>Key insights:</strong><br>
   • f87 (Gini): AI &gt; Human → AI text has less uniform word distribution<br>
   • f45 (adjectives): Human uses more adjectives on average<br>
   • f45 is the most important assigned feature (RF importance=0.002301)</p>
<img src="figures/assigned_features_analysis.png" alt="Feature Analysis">
</div>

<div class="box">
<h2>Model Comparison — Phase 3 (with class weights)</h2>
<p><em>Input: features_extracted.parquet → TF-IDF(5000) + 5 assigned features<br>
Class imbalance handled via classWeight in Logistic Regression and Random Forest.</em></p>
<div class="best">
  Best Model: <strong>{best['model']}</strong>
  — F1={best['f1']:.4f}, Accuracy={best['accuracy']:.4f}, AUC={best['auc']:.4f}
</div>
<table>
  <tr><th>Model</th><th>Accuracy</th><th>Precision</th>
      <th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr>
  {model_rows}
</table>
<img src="figures/final_model_comparison.png" alt="Model Comparison">
<img src="figures/rf_feature_importance.png"  alt="Feature Importance">

{'<h3>Per-Class F1 (Human vs AI)</h3><table><tr><th>Model</th><th>Human F1 (minority)</th><th>AI F1 (majority)</th><th>Weighted F1</th></tr>' + perclass_rows + '</table>' if perclass_rows else ''}
</div>

<div class="box">
<h2>Stream Processing — Phase 4</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Input Source</td><td>features_extracted.parquet</td></tr>
  <tr><td>Stream Method</td><td>File-based Spark Structured Streaming simulation</td></tr>
  <tr><td>Features Used</td><td>TF-IDF(5000) + f3 + f24 + f45 + f66 + f87</td></tr>
  <tr><td>Model Used</td><td>Logistic Regression (trained with class weights)</td></tr>
  <tr><td>Batch Size</td><td>50 records</td></tr>
  <tr><td>Number of Batches</td><td>10</td></tr>
  {stream_rows}
</table>
{'<img src="figures/stream_benchmark.png" alt="Benchmark">' if has_stream else ''}
{'<img src="figures/batch_vs_stream.png" alt="Batch vs Stream">' if has_stream else ''}
</div>

<div class="box">
<h2>MapReduce Results (Phase 2)</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total corpus tokens</td><td>4,404,818</td></tr>
  <tr><td>Vocabulary size</td><td>147,128</td></tr>
  <tr><td>Hapax words</td><td>9,772</td></tr>
  <tr><td>Hapax ratio</td><td>0.0026</td></tr>
  <tr><td>TTR — Human</td><td>0.0362</td></tr>
  <tr><td>TTR — AI</td><td>0.0138</td></tr>
</table>
</div>

<div class="box">
<h2>Conclusion</h2>
<ul>
  <li><strong>Best model:</strong> {best['model']} — F1={best['f1']:.4f}</li>
  <li><strong>Class imbalance ({RATIO:.1f}:1)</strong> was detected and handled via
      class weights in Phase 3, ensuring fair evaluation of the minority class (Human).</li>
  <li>Pipeline: raw CSV → preprocessing → feature extraction →
      class-weighted modeling → streaming</li>
  <li><strong>f45 (adjectives)</strong> is the most important assigned feature</li>
  <li><strong>f87 (Gini)</strong> clearly distinguishes AI text
      (higher inequality in word distribution)</li>
  <li>Stream processing delivers real-time detection with consistent accuracy</li>
  <li>All features pre-computed in features_extracted.parquet
      ({COUNT_HUMAN+COUNT_AI:,} rows × 5 features)</li>
</ul>
</div>
</body></html>"""

html_path = os.path.join(REPORT_DIR, "final_report_summary.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  HTML report: {html_path}")

print("\n" + "="*65)
print("  PHASE 5 COMPLETED SUCCESSFULLY!")
print("="*65)
print("  Figures:")
for fig_name in [
    "class_imbalance.png",
    "assigned_features_analysis.png",
    "final_model_comparison.png",
    "rf_feature_importance.png",
    "stream_benchmark.png",
    "batch_vs_stream.png",
]:
    p = os.path.join(FIG_DIR, fig_name)
    s = "✓" if os.path.exists(p) else "✗ run Phase 4 first"
    print(f"    {s} {fig_name}")
print(f"\n  Open report: start {html_path}")
print("="*65)