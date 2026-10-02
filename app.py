# ============================================================
# STAGE 1 - Imports and page set-up
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st

st.set_page_config(page_title="ECD Vulnerability Screening", page_icon="🧒", layout="wide")

st.title("🧒 Early Childhood Developmental Vulnerability Screening")
st.caption("Explainable stacked ensemble (Logistic Regression + Random Forest + XGBoost + MLP) "
           "- children aged 24-59 months, Kenya")
st.warning("Research prototype trained on **simulated data**. Not for real screening decisions.", icon="⚠️")


# ============================================================
# STAGE 2 - Load the saved model and create the tabs
# ============================================================
MODEL_PATH = "models/ecd_stacked_model.joblib"


@st.cache_resource            # load the model once, not on every click
def load_package(path=MODEL_PATH):
    return joblib.load(path)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Run Step 8 first and start the app from the project folder.")
    st.stop()

pkg = load_package()
REGIONS = pkg["regions"]
THRESHOLD = pkg["threshold"]

tab_assess, tab_perf, tab_about = st.tabs(["Assess a child", "Model performance", "About"])

with tab_about:
    st.markdown(f"""
**Project:** An Explainable Stacked Ensemble Model for Predicting Early Childhood Developmental Vulnerability in Kenya

**Outcome:** child aged 24-59 months *not developmentally on track* (ECDI2030).

**Model:** four base learners (Logistic Regression, Random Forest, XGBoost, MLP) combined by a
logistic-regression meta-learner trained on out-of-fold predictions.
Screening threshold = **{THRESHOLD:.2f}** (catches at least 80% of vulnerable children in cross-validation).

**Explanations:** SHAP permutation explainer on the full pipeline, so reasons use the original factors.

**Limitations:** trained on simulated data; associations are not causes; a single national threshold
misses more vulnerable children in wealthier/urban households - use alongside professional judgement.

*Model file created: {pkg.get('created', 'n/a')}*
""")


# ============================================================
# STAGE 3 - The input form (22 questions)
# ============================================================
EDU = {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"}
BOOKS = {0: "None", 1: "1-2 books", 2: "3 or more"}
WEALTH = {1: "1 - Poorest", 2: "2", 3: "3 - Middle", 4: "4", 5: "5 - Richest"}
YESNO = {0: "No", 1: "Yes"}

with tab_assess:
    with st.form("child_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Child")
            age = st.slider("Age (months)", 24, 59, 36)
            sex = st.radio("Sex", [0, 1], format_func=lambda v: "Boy" if v == 0 else "Girl", horizontal=True)
            lbw = st.radio("Low birth weight (<2.5 kg)?", [0, 1], format_func=YESNO.get, horizontal=True)
            haz_known = st.checkbox("Height measured", value=True)
            haz = st.number_input("Height-for-age z-score (HAZ)", -6.0, 6.0, -0.5, 0.1)
            whz_known = st.checkbox("Weight measured", value=True)
            whz = st.number_input("Weight-for-height z-score (WHZ)", -6.0, 6.0, 0.0, 0.1)
            ill = st.radio("Diarrhoea/fever in last 2 weeks?", [0, 1], format_func=YESNO.get, horizontal=True)
            vacc = st.radio("Fully vaccinated?", [1, 0], format_func=YESNO.get, horizontal=True)

        with c2:
            st.subheader("Mother & household")
            m_edu = st.selectbox("Mother's education", list(EDU), index=1, format_func=EDU.get)
            m_age = st.number_input("Mother's age", 15, 49, 28)
            region = st.selectbox("Region", list(REGIONS), index=6, format_func=REGIONS.get)
            rural = st.radio("Residence", [1, 0], format_func=lambda v: "Rural" if v == 1 else "Urban", horizontal=True)
            wealth = st.select_slider("Wealth quintile", options=list(WEALTH), value=3, format_func=WEALTH.get)
            hh = st.number_input("Household size", 2, 15, 5)
            water = st.radio("Improved drinking water?", [1, 0], format_func=YESNO.get, horizontal=True)
            sani = st.radio("Improved sanitation?", [1, 0], format_func=YESNO.get, horizontal=True)

        with c3:
            st.subheader("Care & learning")
            stim = st.slider("Early stimulation activities with an adult (past 3 days)", 0, 6, 3,
                             help="Reading, storytelling, singing, outings, playing, naming/counting/drawing")
            father = st.radio("Father took part in activities?", [0, 1], format_func=YESNO.get, horizontal=True)
            books = st.selectbox("Children's books at home", list(BOOKS), format_func=BOOKS.get)
            toys = st.slider("Types of playthings (0-3)", 0, 3, 2)
            superv = st.radio("Left alone or with a child <10 in past week?", [0, 1],
                              format_func=YESNO.get, horizontal=True)
            disc = st.radio("Physical discipline used?", [0, 1], format_func=YESNO.get, horizontal=True)
            ecde = st.radio("Attends ECDE / pre-primary?", [0, 1], format_func=YESNO.get, horizontal=True,
                            help="Only possible from 36 months")

        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        child = dict(age_months=age, sex_female=sex, low_birth_weight=lbw,
                     haz=haz if haz_known else np.nan, whz=whz if whz_known else np.nan,
                     recent_illness=ill, fully_vaccinated=vacc, mother_education=m_edu, mother_age=m_age,
                     region=region, rural=rural, wealth_quintile=wealth, household_size=hh,
                     improved_water=water, improved_sanitation=sani, early_stimulation=stim,
                     father_engaged=father, children_books=books, playthings=toys,
                     inadequate_supervision=superv, violent_discipline=disc,
                     ecde_attendance=ecde if age >= 36 else 0)
        with st.expander("Data sent to the model"):
            st.json({k: (None if pd.isna(v) else v) for k, v in child.items()})

            # ============================================================
# STAGE 4 - Predict risk and show the decision
# ============================================================
def predict(pkg, child):
    x = pd.DataFrame([child])[pkg["features"]]
    return float(pkg["model"].predict_proba(x)[:, 1][0])


def risk_band(p, threshold):
    if p >= 0.40:
        return "High", "#C0392B"
    if p >= threshold:
        return "Moderate", "#E67E22"
    return "Low", "#27AE60"


with tab_assess:
    if submitted:
        p = predict(pkg, child)
        band, colour = risk_band(p, THRESHOLD)

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted risk of vulnerability", f"{p:.1%}")
        r2.markdown(f"**Risk band**<br><span style='font-size:2rem;color:{colour}'>{band}</span>",
                    unsafe_allow_html=True)
        if p >= THRESHOLD:
            r3.error(f"**REFER** for developmental assessment\n\n(risk ≥ screening threshold of {THRESHOLD:.0%})")
        else:
            r3.success(f"**On track** - routine follow-up\n\n(risk below screening threshold of {THRESHOLD:.0%})")
        st.progress(min(p, 1.0))

        # ============================================================
# STAGE 5 - Explain the prediction with SHAP
# ============================================================
LABELS = {
    "age_months": "Child's age (months)", "sex_female": "Sex", "low_birth_weight": "Low birth weight",
    "haz": "Height-for-age z-score", "whz": "Weight-for-height z-score", "recent_illness": "Recent illness",
    "fully_vaccinated": "Fully vaccinated", "mother_education": "Mother's education", "mother_age": "Mother's age",
    "region": "Region", "rural": "Rural residence", "wealth_quintile": "Wealth quintile",
    "household_size": "Household size", "improved_water": "Improved water", "improved_sanitation": "Improved sanitation",
    "early_stimulation": "Early stimulation activities", "father_engaged": "Father engaged in activities",
    "children_books": "Children's books", "playthings": "Types of playthings",
    "inadequate_supervision": "Inadequate supervision", "violent_discipline": "Violent discipline",
    "ecde_attendance": "Attends ECDE",
}
YESNO_FEATURES = ("low_birth_weight", "recent_illness", "fully_vaccinated", "rural", "improved_water",
                  "improved_sanitation", "father_engaged", "inadequate_supervision",
                  "violent_discipline", "ecde_attendance")


def explain(pkg, child):
    """SHAP contribution of each factor for one child."""
    import shap
    feats = pkg["features"]
    f = lambda d: pkg["model"].predict_proba(pd.DataFrame(d, columns=feats))[:, 1]
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(pkg["background"]), seed=0)
    x = pd.DataFrame([child])[feats].fillna(pkg["train_medians"])
    sv = explainer(x, max_evals=4 * len(feats) + 1)
    return pd.Series(sv.values[0], index=feats), float(sv.base_values[0])


def describe(feature, value):
    """Turn a coded value into readable text, e.g. mother_education 0 -> 'None'."""
    if pd.isna(value):
        return "not measured"
    lookups = {"region": REGIONS, "mother_education": EDU, "children_books": BOOKS, "wealth_quintile": WEALTH}
    if feature in lookups:
        return lookups[feature][int(value)]
    if feature == "sex_female":
        return "Girl" if value == 1 else "Boy"
    if feature in YESNO_FEATURES:
        return YESNO[int(value)]
    return f"{value:g}"


def contribution_chart(contrib, child, top_k=10):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_k)[::-1]
    labels = [f"{LABELS[f]} = {describe(f, child[f])}" for f in top.index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(top) + 1))
    ax.barh(labels, top.values * 100, color=["#C0392B" if v > 0 else "#2E86C1" for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Change in predicted risk (percentage points)")
    ax.set_title("What drives this child's risk (red = raises, blue = lowers)")
    fig.tight_layout()
    return fig


with tab_assess:
    if submitted:
        with st.spinner("Explaining the prediction with SHAP..."):
            contrib, base = explain(pkg, child)

        st.subheader("Why this result?")
        st.caption(f"Starting point: the average child's risk is {base:.0%}. "
                   "Each bar shows how much a factor moves this child's risk up or down.")
        left, right = st.columns([3, 2])
        left.pyplot(contribution_chart(contrib, child))

        right.markdown("**Main factors raising risk**")
        for f, v in contrib[contrib > 0].sort_values(ascending=False).head(3).items():
            right.markdown(f"- {LABELS[f]} ({describe(f, child[f])}): +{v * 100:.1f} pts")
        right.markdown("**Main protective factors**")
        for f, v in contrib[contrib < 0].sort_values().head(3).items():
            right.markdown(f"- {LABELS[f]} ({describe(f, child[f])}): {v * 100:.1f} pts")


            # ============================================================
# STAGE 6 - Model performance tab
# ============================================================
with tab_perf:
    st.subheader("Test-set performance (n = 1,200 children)")
    metrics = pkg.get("test_metrics", {})
    if metrics:
        cols = st.columns(4)
        for col, k in zip(cols, ["ROC-AUC (95% CI)", "Sensitivity", "Specificity", "Brier"]):
            col.metric(k, str(metrics.get(k, "-")))

    for title, path in [("ROC, precision-recall and calibration", "figures/step6_roc_pr_calibration.png"),
                        ("SHAP summary: what drives risk", "figures/step7_shap_beeswarm.png"),
                        ("Fairness across subgroups", "figures/step8_fairness.png"),
                        ("Meta-learner weights", "figures/step5_meta_weights.png")]:
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path)

    for title, path in [("Model comparison", "figures/step6_test_results.csv"),
                        ("Fairness by subgroup", "figures/step8_fairness_by_subgroup.csv")]:
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(path), hide_index=True)
            