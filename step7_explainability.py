# %% 7.1 Load model and data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)

d3 = joblib.load("artifacts/step3_data.joblib")
d5 = joblib.load("artifacts/step5_stacked_model.joblib")
X_train, X_test, y_test = d3["X_train"], d3["X_test"], d3["y_test"]
SEED = d3["SEED"]
stacked_model = d5["stacked_model"]
FEATURES = list(X_train.columns)
print("SHAP version:", shap.__version__, "| Features:", len(FEATURES))



# %% 7.2 Permutation importance on the test set
pi = permutation_importance(stacked_model, X_test, y_test, scoring="roc_auc",
                            n_repeats=10, random_state=SEED, n_jobs=-1)
perm_imp = (pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean, "SD": pi.importances_std})
              .sort_values("Importance", ascending=False).reset_index(drop=True))

top = perm_imp.head(15).iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top.Feature, top.Importance, xerr=top.SD, color="#4C72B0")
plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
plt.title("Permutation importance - stacked ensemble (test set)")
plt.tight_layout()
plt.savefig("figures/step7_permutation_importance.png", dpi=150)
plt.show()
perm_imp.head(10).round(4)


# %% 7.3 Compute SHAP values for the whole stacked ensemble
# Fill gaps with training medians (the model does this internally anyway, so predictions are unchanged)
med = X_train.median(numeric_only=True)
X_bg = shap.utils.sample(X_train.fillna(med), 50, random_state=SEED)   # "average child" baseline
X_explain = X_test.fillna(med).sample(200, random_state=SEED)          # children to explain

def predict_risk(data):
    return stacked_model.predict_proba(pd.DataFrame(data, columns=FEATURES))[:, 1]

explainer = shap.PermutationExplainer(predict_risk, shap.maskers.Independent(X_bg), seed=SEED)
shap_values = explainer(X_explain, max_evals=4 * len(FEATURES) + 1)

# Check: base value + sum of contributions should equal the model's prediction
recon = shap_values.base_values + shap_values.values.sum(axis=1)
print(f"Average predicted risk (base value): {shap_values.base_values[0]:.3f}")
print(f"Max difference between SHAP sum and prediction: {np.abs(recon - predict_risk(X_explain)).max():.6f}")


# %% 7.4 SHAP beeswarm and bar plots
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("SHAP summary - what drives predicted vulnerability")
plt.tight_layout()
plt.savefig("figures/step7_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Mean |SHAP| - overall feature importance")
plt.tight_layout()
plt.savefig("figures/step7_shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 7.5 Compare SHAP importance with permutation importance
shap_imp = (pd.Series(np.abs(shap_values.values).mean(axis=0), index=FEATURES)
              .sort_values(ascending=False).rename("Mean |SHAP|"))
compare = (shap_imp.to_frame()
             .join(perm_imp.set_index("Feature")["Importance"].rename("Permutation"))
             .assign(SHAP_rank=lambda t: t["Mean |SHAP|"].rank(ascending=False).astype(int),
                     Perm_rank=lambda t: t["Permutation"].rank(ascending=False).astype(int)))
rho, p = spearmanr(compare["Mean |SHAP|"], compare["Permutation"])
print(f"Rank agreement between SHAP and permutation importance: Spearman rho = {rho:.2f} (p = {p:.4f})")
compare.to_csv("figures/step7_importance_comparison.csv")
compare.round(4).head(12)



# %% 7.6 SHAP dependence plots for key modifiable factors
key = ["age_months", "early_stimulation", "children_books", "haz"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, f in zip(axes, key):
    shap.plots.scatter(shap_values[:, f], ax=ax, show=False)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title(f)
plt.suptitle("SHAP dependence: effect of each factor on predicted risk (above 0 = raises risk)", y=1.03)
plt.tight_layout()
plt.savefig("figures/step7_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 7.7 Explain individual children
risk = predict_risk(X_explain)
cases = {"Highest-risk child": int(np.argmax(risk)), "Lowest-risk child": int(np.argmin(risk))}

for label, i in cases.items():
    cid = X_explain.index[i]
    actual = "vulnerable" if y_test.loc[cid] == 1 else "on track"
    print(f"\n{label} (ID {cid}): predicted risk {risk[i]:.1%} | actual: {actual}")
    contrib = pd.Series(shap_values.values[i], index=FEATURES).sort_values(key=abs, ascending=False).head(5)
    for f, v in contrib.items():
        direction = "raises" if v > 0 else "lowers"
        print(f"   {f} = {X_explain.iloc[i][f]:g}  -> {direction} risk by {abs(v) * 100:.1f} percentage points")

    plt.figure()
    shap.plots.waterfall(shap_values[i], max_display=12, show=False)
    plt.title(f"{label}: predicted risk {risk[i]:.0%}")
    plt.tight_layout()
    plt.savefig(f"figures/step7_waterfall_{label.split('-')[0].lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()


    # %% 7.8 Partial dependence + ICE curves
pd_feats = ["early_stimulation", "haz", "age_months", "children_books"]
X_pd = X_train.dropna(subset=pd_feats).sample(1000, random_state=SEED).astype(float)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
PartialDependenceDisplay.from_estimator(stacked_model, X_pd, pd_feats, kind="both", subsample=100,
                                        random_state=SEED, ax=ax, ice_lines_kw={"alpha": 0.08},
                                        pd_line_kw={"color": "red", "lw": 3})
plt.suptitle("Partial dependence (red = average) and ICE curves (grey = individual children)", y=1.04)
plt.tight_layout()
plt.savefig("figures/step7_partial_dependence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% 7.9 Save explanations
joblib.dump({"shap_values": shap_values, "X_explain": X_explain, "perm_importance": perm_imp,
             "importance_comparison": compare}, "artifacts/step7_explanations.joblib")
print("Saved: artifacts/step7_explanations.joblib")