import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/hadoop/bigdata_project/reports/lr_coefficients.csv")

colors = ["#e53935" if v < 0 else "#1e88e5" for v in df["cofficient"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Logistic Regression — 5 Assigned Feature Coefficients\n"
    "(positive = AI signal, negative = Human signal)",
    fontsize=11, fontweight="bold"
)

axes[0].barh(df["feature"], df["cofficient"], color=colors, edgecolor="white")
axes[0].axvline(x=0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("Coefficient value")
axes[0].set_title("Raw coefficients")
for i, (val, feat) in enumerate(zip(df["cofficient"], df["feature"])):
    offset = 1 if val >= 0 else -1
    axes[0].text(val + offset, i, f"{val:.3f}", va="center", fontsize=9)

df_abs = df.sort_values("abs_coef", ascending=True)
colors2 = ["#e53935" if v < 0 else "#1e88e5" for v in df_abs["cofficient"]]
axes[1].barh(df_abs["feature"], df_abs["abs_coef"], color=colors2, edgecolor="white")
axes[1].set_xlabel("|Coefficient| — importance")
axes[1].set_title("Absolute coefficients (ranked)")
for i, val in enumerate(df_abs["abs_coef"]):
    axes[1].text(val + 0.3, i, f"{val:.3f}", va="center", fontsize=9)

fig.legend(handles=[
    plt.Rectangle((0,0),1,1, color="#1e88e5", label="Positive → AI signal"),
    plt.Rectangle((0,0),1,1, color="#e53935", label="Negative → Human signal"),
], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out = "/home/hadoop/bigdata_project/reports/figures/lr_coefficients.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")