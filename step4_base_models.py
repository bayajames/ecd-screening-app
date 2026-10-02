# %% 4.1 Load the prepared data from Step 3
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)

data = joblib.load("artifacts/step3_data.joblib")
X_train, y_train = data["X_train"], data["y_train"]
preprocessor, SEED = data["preprocessor"], data["SEED"]
print("Training set:", X_train.shape)


# %% 4.2 Define the four base learners
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class ratio (on-track per vulnerable child): {pos_weight:.2f}")

base_models = {
    "Logistic Regression": LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=pos_weight,
                             eval_metric="logloss", random_state=SEED, n_jobs=-1),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                      early_stopping=True, max_iter=500, random_state=SEED),
}

# Each model gets the Step 3 preprocessor in front of it
pipelines = {name: Pipeline([("prep", preprocessor), ("model", m)]) for name, m in base_models.items()}
print(list(pipelines))

# %% 4.3 5-fold cross-validation on the TRAINING set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train, y_train))
oof = {}      # out-of-fold predicted probabilities
rows = []

for name, pipe in pipelines.items():
    t0 = time.time()
    p = cross_val_predict(pipe, X_train, y_train, cv=folds, method="predict_proba", n_jobs=-1)[:, 1]
    oof[name] = p
    fold_auc = [roc_auc_score(y_train.iloc[te], p[te]) for _, te in folds]
    rows.append({"Model": name,
                 "ROC-AUC (mean)": np.mean(fold_auc), "ROC-AUC (SD)": np.std(fold_auc),
                 "PR-AUC": average_precision_score(y_train, p),
                 "Brier": brier_score_loss(y_train, p),
                 "Time (s)": time.time() - t0})
    print(f"{name:<20} AUC = {np.mean(fold_auc):.3f} ± {np.std(fold_auc):.3f}")

cv_results = (pd.DataFrame(rows).set_index("Model").round(3)
                .sort_values("ROC-AUC (mean)", ascending=False))
cv_results.to_csv("figures/step4_cv_results.csv")
cv_results

# %% 4.4 Plot fold-by-fold AUC
fold_df = pd.DataFrame([{"Model": n, "Fold": k + 1, "ROC-AUC": roc_auc_score(y_train.iloc[te], oof[n][te])}
                        for n in oof for k, (_, te) in enumerate(folds)])

plt.figure(figsize=(8, 4.5))
sns.boxplot(data=fold_df, x="Model", y="ROC-AUC", color="#9ecae1")
sns.stripplot(data=fold_df, x="Model", y="ROC-AUC", color="black", size=6)
plt.title("5-fold cross-validated ROC-AUC of base learners (training set)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("figures/step4_cv_auc_boxplot.png", dpi=150)
plt.show()


# %% 4.5 Model diversity: do the models make DIFFERENT predictions?
oof_df = pd.DataFrame(oof)
corr = oof_df.corr(method="spearman")

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0.7, vmax=1, square=True)
plt.title("Correlation of out-of-fold predictions")
plt.tight_layout()
plt.savefig("figures/step4_prediction_correlation.png", dpi=150)
plt.show()

print(f"Best single model AUC:      {max(roc_auc_score(y_train, oof_df[c]) for c in oof_df):.3f}")
print(f"Simple average of all four: {roc_auc_score(y_train, oof_df.mean(axis=1)):.3f}")



# %% 4.6 Fit each base model on the full training set and save
fitted = {}
for name, pipe in pipelines.items():
    fitted[name] = pipe.fit(X_train, y_train)
    print("Fitted:", name)

joblib.dump({"fitted_base_models": fitted, "base_models": base_models,
             "oof_predictions": oof_df, "cv_results": cv_results},
            "artifacts/step4_base_models.joblib")
print("Saved: artifacts/step4_base_models.joblib")

