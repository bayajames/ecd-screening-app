# %% 5.1 Load data (Step 3) and base models (Step 4)
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_curve)

sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)

d3 = joblib.load("artifacts/step3_data.joblib")
d4 = joblib.load("artifacts/step4_base_models.joblib")
X_train, y_train = d3["X_train"], d3["y_train"]
preprocessor, SEED = d3["preprocessor"], d3["SEED"]
base_models = d4["base_models"]
cv_results_base = d4["cv_results"]
print("Base learners:", list(base_models))


# %% 5.2 Build the stacked ensemble
SHORT = {"Logistic Regression": "lr", "Random Forest": "rf", "XGBoost": "xgb", "MLP (Neural Net)": "mlp"}
estimators = [(SHORT[name], clone(m)) for name, m in base_models.items()]

stack = StackingClassifier(
    estimators=estimators,                                            # Level 0: four base learners
    final_estimator=LogisticRegression(C=1.0, max_iter=2000),         # Level 1: meta-learner
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),  # out-of-fold predictions
    stack_method="predict_proba",                                     # pass probabilities, not 0/1
    passthrough=False,                                                # meta-learner sees ONLY base predictions
    n_jobs=-1,
)
stacked_model = Pipeline([("prep", clone(preprocessor)), ("stack", stack)])
stacked_model


# %% 5.3 Train the stacked ensemble
t0 = time.time()
stacked_model.fit(X_train, y_train)
print(f"Stacked ensemble trained in {time.time() - t0:.1f} s")


# %% 5.4 Meta-learner weights
meta = stacked_model.named_steps["stack"].final_estimator_
NAMES = {v: k for k, v in SHORT.items()}
weights = pd.Series(meta.coef_[0], index=[NAMES[n] for n, _ in estimators]).sort_values()
print("Meta-learner coefficients:"); print(weights.round(3).to_string())
print(f"Intercept: {meta.intercept_[0]:.3f}")

plt.figure(figsize=(7, 3.5))
weights.plot.barh(color=["#C44E52" if w < 0 else "#55A868" for w in weights])
plt.axvline(0, color="black", lw=0.8)
plt.title("Meta-learner coefficients (weight given to each base model)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig("figures/step5_meta_weights.png", dpi=150)
plt.show()


# %% 5.5 Cross-validate the stack the SAME way as the base models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train, y_train))
t0 = time.time()
oof_stack = cross_val_predict(stacked_model, X_train, y_train, cv=folds,
                              method="predict_proba", n_jobs=-1)[:, 1]
fold_auc = [roc_auc_score(y_train.iloc[te], oof_stack[te]) for _, te in folds]
print(f"Cross-validation finished in {time.time() - t0:.1f} s")

stack_row = pd.DataFrame([{
    "ROC-AUC (mean)": np.mean(fold_auc), "ROC-AUC (SD)": np.std(fold_auc),
    "PR-AUC": average_precision_score(y_train, oof_stack),
    "Brier": brier_score_loss(y_train, oof_stack)}], index=["Stacked Ensemble"])
comparison = (pd.concat([stack_row, cv_results_base.drop(columns="Time (s)")])
                .round(3).sort_values("ROC-AUC (mean)", ascending=False))
comparison.to_csv("figures/step5_cv_comparison.csv")
comparison


# %% 5.6 Choose the screening threshold from out-of-fold predictions
prec, rec, thr = precision_recall_curve(y_train, oof_stack)
prec, rec = prec[:-1], rec[:-1]
f1 = 2 * prec * rec / (prec + rec + 1e-12)

TARGET_RECALL = 0.80
THRESHOLD = float(thr[np.where(rec >= TARGET_RECALL)[0][-1]])   # highest threshold keeping recall >= 80%
F1_THRESHOLD = float(thr[f1.argmax()])

def at(t):
    pred = oof_stack >= t
    tp = (pred & (y_train == 1)).sum(); fp = (pred & (y_train == 0)).sum()
    fn = (~pred & (y_train == 1)).sum()
    return {"Threshold": round(t, 3), "Recall": round(tp / (tp + fn), 3),
            "Precision": round(tp / (tp + fp), 3), "% children flagged": round(pred.mean() * 100, 1)}

print(pd.DataFrame([at(0.5), at(F1_THRESHOLD), at(THRESHOLD)],
                   index=["Default 0.5", "Best F1", f"Screening (recall >= {TARGET_RECALL:.0%})"]))

plt.figure(figsize=(8, 4.5))
plt.plot(thr, rec, label="Recall (vulnerable children caught)", lw=2)
plt.plot(thr, prec, label="Precision (flags that are correct)", lw=2)
plt.plot(thr, f1, label="F1", lw=1.5, ls="--")
plt.axvline(THRESHOLD, color="red", ls=":", label=f"Chosen threshold = {THRESHOLD:.2f}")
plt.xlabel("Decision threshold (predicted probability)")
plt.ylabel("Score")
plt.title("Choosing the screening threshold (out-of-fold, training set)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/step5_threshold.png", dpi=150)
plt.show()

# %% 5.7 Save the stacked model
joblib.dump({"stacked_model": stacked_model, "threshold": THRESHOLD, "f1_threshold": F1_THRESHOLD,
             "oof_stack": oof_stack, "cv_comparison": comparison},
            "artifacts/step5_stacked_model.joblib")
print("Saved: artifacts/step5_stacked_model.joblib")