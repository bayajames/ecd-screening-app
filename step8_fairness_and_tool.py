# %% 8.1 Load the model, test data and test predictions
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_auc_score, confusion_matrix

sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

d3 = joblib.load("artifacts/step3_data.joblib")
d5 = joblib.load("artifacts/step5_stacked_model.joblib")
d6 = joblib.load("artifacts/step6_test_predictions.joblib")

X_train, X_test, y_test = d3["X_train"], d3["X_test"], d3["y_test"]
FEATURES, SEED = d3["FEATURES"], d3["SEED"]
stacked_model, THRESHOLD = d5["stacked_model"], d5["threshold"]
p_test = d6["probas"]["Stacked Ensemble"]
yhat = (p_test >= THRESHOLD).astype(int)

REGIONS = {0: "Nairobi", 1: "Central", 2: "Coast", 3: "Eastern", 4: "North Eastern",
           5: "Nyanza", 6: "Rift Valley", 7: "Western"}
print(f"Threshold: {THRESHOLD:.3f} | Test children: {len(y_test)}")


# %% 8.2 Performance within each subgroup
groups = {
    "Sex": X_test.sex_female.map({0: "Boy", 1: "Girl"}),
    "Residence": X_test.rural.map({0: "Urban", 1: "Rural"}),
    "Wealth": X_test.wealth_quintile.map({1: "Q1 Poorest", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 Richest"}),
    "Region": X_test.region.map(REGIONS),
}

rows = []
yt = y_test.to_numpy()
for gname, g in groups.items():
    for level in sorted(g.unique()):
        m = (g == level).to_numpy()
        tn, fp, fn, tp = confusion_matrix(yt[m], yhat[m], labels=[0, 1]).ravel()
        rows.append({
            "Attribute": gname, "Group": level, "n": int(m.sum()),
            "Observed % vulnerable": round(yt[m].mean() * 100, 1),
            "Mean predicted %": round(p_test[m].mean() * 100, 1),
            "AUC": round(roc_auc_score(yt[m], p_test[m]), 3) if 0 < yt[m].sum() < m.sum() else np.nan,
            "Sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else np.nan,
            "False positive rate": round(fp / (fp + tn), 3) if (fp + tn) else np.nan,
            "% flagged": round(yhat[m].mean() * 100, 1),
        })
fair = pd.DataFrame(rows)
fair.to_csv("figures/step8_fairness_by_subgroup.csv", index=False)
fair

# %% 8.3 Chart: sensitivity and false-positive rate by subgroup
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
for ax, (gname, sub) in zip(axes, fair.groupby("Attribute", sort=False)):
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["Sensitivity"], 0.4, label="Sensitivity (caught)", color="#4C72B0")
    ax.bar(x + 0.2, sub["False positive rate"], 0.4, label="False positive rate", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Group"], rotation=45, ha="right")
    ax.set_title(gname)
    ax.set_ylim(0, 1.05)
axes[0].legend(loc="upper left", fontsize=8)
plt.suptitle("Screening performance by subgroup (stacked ensemble, test set)", y=1.02)
plt.tight_layout()
plt.savefig("figures/step8_fairness.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 8.4 Summarise the gaps between groups
fair["Calibration error (pp)"] = (fair["Mean predicted %"] - fair["Observed % vulnerable"]).round(1)
gaps = fair.groupby("Attribute", sort=False).agg(
    Sensitivity_gap=("Sensitivity", lambda s: s.max() - s.min()),
    AUC_gap=("AUC", lambda s: s.max() - s.min()),
    Max_calibration_error_pp=("Calibration error (pp)", lambda s: s.abs().max()))
gaps["Concern?"] = np.where((gaps.Sensitivity_gap > 0.10) | (gaps.AUC_gap > 0.10), "Review", "OK")
gaps.to_csv("figures/step8_fairness_gaps.csv")
print(gaps.round(3))

# %% 8.5 Save the final model package
package = {
    "model": stacked_model,
    "threshold": THRESHOLD,
    "features": FEATURES,
    "regions": REGIONS,
    "train_medians": X_train.median(numeric_only=True),
    "background": X_train.fillna(X_train.median(numeric_only=True)).sample(50, random_state=SEED),
    "test_metrics": d6["results"].loc["Stacked Ensemble"].to_dict(),
    "created": datetime.date.today().isoformat(),
    "note": "Trained on SIMULATED data - retrain on KDHS 2022 before real use.",
}
joblib.dump(package, "models/ecd_stacked_model.joblib")
print("Saved: models/ecd_stacked_model.joblib")
print(json.dumps(package["test_metrics"], indent=2))


# %% 8.6 Prediction tool for a new child
import shap

pkg = joblib.load("models/ecd_stacked_model.joblib")

def assess_child(child: dict, top_k: int = 5) -> dict:
    """Return risk, screening decision and the main reasons for one child."""
    missing = [f for f in pkg["features"] if f not in child]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    x = pd.DataFrame([child])[pkg["features"]]
    p = float(pkg["model"].predict_proba(x)[:, 1][0])
    band = "High" if p >= 0.40 else "Moderate" if p >= pkg["threshold"] else "Low"
    result = {"predicted_risk": f"{p:.1%}",
              "risk_band": band,
              "decision": "REFER for developmental assessment" if p >= pkg["threshold"]
                          else "On track - routine follow-up"}

    # Explain the prediction with SHAP
    f = lambda d: pkg["model"].predict_proba(pd.DataFrame(d, columns=pkg["features"]))[:, 1]
    expl = shap.PermutationExplainer(f, shap.maskers.Independent(pkg["background"]), seed=0)
    sv = expl(x.fillna(pkg["train_medians"]), max_evals=4 * len(pkg["features"]) + 1)
    contrib = pd.Series(sv.values[0], index=pkg["features"]).sort_values(key=abs, ascending=False).head(top_k)
    result["main_reasons"] = [f"{k} = {child[k]}: {'raises' if v > 0 else 'lowers'} risk by {abs(v) * 100:.1f} pts"
                              for k, v in contrib.items()]
    return result

print("assess_child() is ready")


# %% 8.7 Try it on two example children
child_A = dict(age_months=30, sex_female=0, low_birth_weight=1, haz=-2.6, whz=-0.8, recent_illness=1,
               fully_vaccinated=0, mother_education=0, mother_age=19, region=4, rural=1, wealth_quintile=1,
               household_size=8, improved_water=0, improved_sanitation=0, early_stimulation=1,
               father_engaged=0, children_books=0, playthings=0, inadequate_supervision=1,
               violent_discipline=1, ecde_attendance=0)

child_B = dict(age_months=50, sex_female=1, low_birth_weight=0, haz=0.2, whz=0.1, recent_illness=0,
               fully_vaccinated=1, mother_education=2, mother_age=31, region=1, rural=0, wealth_quintile=4,
               household_size=4, improved_water=1, improved_sanitation=1, early_stimulation=5,
               father_engaged=1, children_books=2, playthings=3, inadequate_supervision=0,
               violent_discipline=0, ecde_attendance=1)

for name, child in [("Child A", child_A), ("Child B", child_B)]:
    print(f"\n{name}:")
    print(json.dumps(assess_child(child), indent=2))