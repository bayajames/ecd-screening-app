# %% 6.1 Load everything from Steps 3-5
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)

d3 = joblib.load("artifacts/step3_data.joblib")
d4 = joblib.load("artifacts/step4_base_models.joblib")
d5 = joblib.load("artifacts/step5_stacked_model.joblib")

X_train, y_train = d3["X_train"], d3["y_train"]
X_test, y_test = d3["X_test"], d3["y_test"]
SEED = d3["SEED"]

models = dict(d4["fitted_base_models"])
models["Stacked Ensemble"] = d5["stacked_model"]

# Each model gets its OWN screening threshold (recall >= 80%), from its out-of-fold predictions
def recall_threshold(y, p, target=0.80):
    prec, rec, thr = precision_recall_curve(y, p)
    return float(thr[np.where(rec[:-1] >= target)[0][-1]])

oof = d4["oof_predictions"].copy()
oof["Stacked Ensemble"] = d5["oof_stack"]
thresholds = {name: recall_threshold(y_train, oof[name].to_numpy()) for name in models}

print("Test set:", X_test.shape, f"| {y_test.sum()} vulnerable children ({y_test.mean():.1%})")
print("Thresholds:", {k: round(v, 3) for k, v in thresholds.items()})


# %% 6.2 Predict on the test set (first time the models see these children)
probas = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}
pd.DataFrame(probas).describe().round(3)

# %% 6.3 Test-set performance with 95% bootstrap confidence intervals
def bootstrap_ci(y, p, metric, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"{metric(y, p):.3f} ({lo:.3f}-{hi:.3f})"

rows = []
for name, p in probas.items():
    yhat = (p >= thresholds[name]).astype(int)
    rows.append({"Model": name,
                 "ROC-AUC (95% CI)": bootstrap_ci(y_test, p, roc_auc_score),
                 "PR-AUC (95% CI)": bootstrap_ci(y_test, p, average_precision_score),
                 "Brier": round(brier_score_loss(y_test, p), 3),
                 "Sensitivity": round(recall_score(y_test, yhat), 3),
                 "Specificity": round(recall_score(y_test, yhat, pos_label=0), 3),
                 "Precision": round(precision_score(y_test, yhat), 3),
                 "F1": round(f1_score(y_test, yhat), 3),
                 "Accuracy": round(accuracy_score(y_test, yhat), 3)})

results = pd.DataFrame(rows).set_index("Model")
results = results.loc[results["ROC-AUC (95% CI)"].str[:5].astype(float).sort_values(ascending=False).index]
results.to_csv("figures/step6_test_results.csv")
results


# %% 6.4 Is the stack significantly better? (paired bootstrap)
base_names = [n for n in probas if n != "Stacked Ensemble"]
yt = y_test.to_numpy()
rng = np.random.default_rng(SEED)
idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]

rows = []
for b in base_names:
    d_auc = [roc_auc_score(yt[i], probas["Stacked Ensemble"][i]) - roc_auc_score(yt[i], probas[b][i]) for i in idx]
    d_bri = [brier_score_loss(yt[i], probas["Stacked Ensemble"][i]) - brier_score_loss(yt[i], probas[b][i]) for i in idx]
    rows.append({"Stack vs": b,
                 "AUC difference": f"{np.mean(d_auc):+.3f} ({np.percentile(d_auc, 2.5):+.3f} to {np.percentile(d_auc, 97.5):+.3f})",
                 "AUC better?": "Yes" if np.percentile(d_auc, 2.5) > 0 else "No sig. difference",
                 "Brier difference": f"{np.mean(d_bri):+.3f} ({np.percentile(d_bri, 2.5):+.3f} to {np.percentile(d_bri, 97.5):+.3f})",
                 "Brier better?": "Yes" if np.percentile(d_bri, 97.5) < 0 else "No sig. difference"})

sig = pd.DataFrame(rows).set_index("Stack vs")
sig.to_csv("figures/step6_significance.csv")
sig

# %% 6.5 ROC, precision-recall and calibration curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for name, p in probas.items():
    lw = 3 if name == "Stacked Ensemble" else 1.3
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[0].plot(fpr, tpr, lw=lw, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, p)
    axes[1].plot(rc, pr, lw=lw, label=f"{name} ({average_precision_score(y_test, p):.3f})")
    frac, mean_p = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    axes[2].plot(mean_p, frac, marker="o", lw=lw, label=name)

axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="Sensitivity")
axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1)
axes[1].set(title="Precision-recall curves (test set)", xlabel="Recall", ylabel="Precision")
axes[2].plot([0, 1], [0, 1], "k--", lw=1)
axes[2].set(title="Calibration (test set)", xlabel="Predicted probability", ylabel="Observed proportion vulnerable")
for ax in axes:
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/step6_roc_pr_calibration.png", dpi=150)
plt.show()


# %% 6.6 Confusion matrix for the stacked ensemble
T = thresholds["Stacked Ensemble"]
yhat = (probas["Stacked Ensemble"] >= T).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, yhat), display_labels=["On track", "Vulnerable"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Stacked ensemble on test set (threshold = {T:.2f})")
plt.tight_layout()
plt.savefig("figures/step6_confusion_matrix.png", dpi=150)
plt.show()

print(f"Of {tp + fn} vulnerable children, the model correctly flagged {tp} ({tp / (tp + fn):.0%}) and missed {fn}.")
print(f"Of {tn + fp} on-track children, {tn} ({tn / (tn + fp):.0%}) were correctly cleared; {fp} were flagged for follow-up.")
print(f"Of the {tp + fp} children flagged, {tp} ({tp / (tp + fp):.0%}) were truly vulnerable "
      f"(vs {y_test.mean():.0%} if children were picked at random).")


# %% 6.7 Save test-set predictions for Steps 7-8
joblib.dump({"probas": probas, "thresholds": thresholds, "results": results},
            "artifacts/step6_test_predictions.joblib")
print("Saved: artifacts/step6_test_predictions.joblib")