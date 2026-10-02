"""
STEP 1 - Simulated dataset for:
An Explainable Stacked Ensemble Model for Predicting Early Childhood
Developmental Vulnerability in Kenya

Generates 6,000 synthetic children aged 24-59 months, structured like the
KDHS 2022 children's file. Synthetic data only: not real findings.
Run:  python step1_simulate_data.py   ->  writes ecd_kenya_simulated.csv
"""
import numpy as np
import pandas as pd

SEED = 42

REGIONS = ["Nairobi", "Central", "Coast", "Eastern", "North Eastern", "Nyanza", "Rift Valley", "Western"]

def simulate_ecd_data(n=6000, seed=SEED):
    rng = np.random.default_rng(seed)
    sig = lambda z: 1 / (1 + np.exp(-z))

    region = rng.choice(len(REGIONS), n, p=[.10, .12, .09, .14, .05, .14, .25, .11])
    rural_p = np.array([0.0, .70, .60, .75, .80, .72, .70, .78])[region]
    rural = rng.binomial(1, rural_p)

    # wealth: poorer in rural & North Eastern
    w_lat = rng.normal(0, 1, n) - 0.9 * rural - 0.8 * (region == 4) + 0.5 * (region == 0) + 0.3 * (region == 1)
    wealth = pd.qcut(w_lat, 5, labels=False) + 1

    m_edu_lat = 0.6 * (wealth - 3) - 0.8 * (region == 4) + rng.normal(0, 1, n)
    mother_education = np.digitize(m_edu_lat, [-1.6, -0.1, 1.3])          # 0..3
    mother_age = np.clip(rng.normal(29, 6, n), 15, 49).round()
    household_size = np.clip(rng.poisson(4 + 1.2 * rural + 0.8 * (region == 4)), 2, 15)

    age_months = rng.integers(24, 60, n)
    sex_female = rng.binomial(1, .5, n)
    low_birth_weight = rng.binomial(1, sig(-2.3 - 0.25 * (wealth - 3)))
    haz = rng.normal(-0.9 + 0.22 * (wealth - 3) + 0.15 * mother_education - 0.5 * low_birth_weight, 1.15)
    whz = rng.normal(-0.1 + 0.08 * (wealth - 3) - 0.6 * (region == 4), 1.0)
    recent_illness = rng.binomial(1, sig(-1.0 - 0.15 * (wealth - 3)))
    fully_vaccinated = rng.binomial(1, sig(1.4 + 0.3 * (wealth - 3) + 0.3 * mother_education - 1.0 * (region == 4)))

    improved_water = rng.binomial(1, sig(0.8 + 0.6 * (wealth - 3) - 0.6 * rural))
    improved_sanitation = rng.binomial(1, sig(0.2 + 0.7 * (wealth - 3) - 0.5 * rural))

    early_stimulation = rng.binomial(6, sig(-0.3 + 0.35 * mother_education + 0.2 * (wealth - 3)))
    father_engaged = rng.binomial(1, sig(-0.8 + 0.25 * mother_education + 0.15 * (wealth - 3)))
    children_books = np.digitize(0.7 * (wealth - 3) + 0.5 * mother_education - 0.5 * rural
                                 + rng.normal(0, 1, n), [0.6, 1.8])         # 0..2
    playthings = rng.binomial(3, sig(0.3 + 0.2 * (wealth - 3)))
    inadequate_supervision = rng.binomial(1, sig(-1.0 - 0.3 * (wealth - 3) + 0.3 * rural + 0.06 * (household_size - 5)))
    violent_discipline = rng.binomial(1, sig(0.4 - 0.15 * mother_education))
    ecde_attendance = np.where(age_months >= 36,
                               rng.binomial(1, sig(-1.2 + 0.09 * (age_months - 36) + 0.35 * (wealth - 3)
                                                   - 0.4 * rural - 1.0 * (region == 4))), 0)

    # ---- outcome model (non-linear, with interactions) ----
    stim_effect = -0.9 * (1 - np.exp(-early_stimulation / 2))                  # saturating benefit
    stunt_effect = 0.55 * np.clip(-2 - haz, 0, None) + 0.35 * (haz < -2)       # threshold effect
    logit = (-0.55
             - 0.045 * (age_months - 42)                       # older children more likely on track
             + 0.25 * (1 - sex_female)
             - 0.20 * (wealth - 3)
             - 0.30 * mother_education
             + stim_effect
             + stunt_effect
             + 0.45 * low_birth_weight
             - 0.35 * children_books
             - 0.15 * playthings
             + 0.55 * inadequate_supervision
             + 0.30 * violent_discipline
             - 0.55 * ecde_attendance
             - 0.25 * father_engaged
             + 0.25 * recent_illness
             - 0.20 * fully_vaccinated
             + 0.40 * (region == 4)
             + 0.30 * (whz < -2)
             + 0.35 * inadequate_supervision * (mother_education == 0)   # interaction
             + rng.normal(0, 0.6, n))                                    # unobserved factors
    vulnerable = rng.binomial(1, sig(logit))

    df = pd.DataFrame(dict(
        age_months=age_months, sex_female=sex_female, low_birth_weight=low_birth_weight,
        haz=haz.round(2), whz=whz.round(2), recent_illness=recent_illness, fully_vaccinated=fully_vaccinated,
        mother_education=mother_education, mother_age=mother_age,
        region=region, rural=rural, wealth_quintile=wealth, household_size=household_size,
        improved_water=improved_water, improved_sanitation=improved_sanitation,
        early_stimulation=early_stimulation, father_engaged=father_engaged, children_books=children_books,
        playthings=playthings, inadequate_supervision=inadequate_supervision,
        violent_discipline=violent_discipline, ecde_attendance=ecde_attendance,
        vulnerable=vulnerable))

    # realistic missingness (anthropometry & maternal age often incomplete in surveys)
    for col, rate in [("haz", .07), ("whz", .07), ("mother_age", .03), ("children_books", .02)]:
        df.loc[rng.random(n) < rate, col] = np.nan
    return df


if __name__ == "__main__":
    df = simulate_ecd_data()
    df.to_csv("ecd_kenya_simulated.csv", index=False)
    print("Shape:", df.shape)
    print(f"Developmentally vulnerable: {df.vulnerable.mean():.1%}")
    print("\nMissing values:\n", df.isna().sum()[lambda s: s > 0])
    print("\nFirst rows:\n", df.head())
