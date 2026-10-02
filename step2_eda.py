# %% 2.1 Load libraries and data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("ecd_kenya_simulated.csv")
TARGET = "vulnerable"

REGIONS = {0: "Nairobi", 1: "Central", 2: "Coast", 3: "Eastern", 4: "North Eastern",
           5: "Nyanza", 6: "Rift Valley", 7: "Western"}
EDU = {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"}

print("Rows:", df.shape[0], "| Columns:", df.shape[1])
df.head(10)

# %% 2.2 Data types and missing values
df.info()

missing = (df.isna().sum().to_frame("n_missing")
             .assign(pct_missing=lambda t: (t.n_missing / len(df) * 100).round(2))
             .query("n_missing > 0")
             .sort_values("n_missing", ascending=False))
print("\nMissing values:")
print(missing)


# %% 2.3 Target variable
counts = df[TARGET].value_counts().sort_index()
pct = (counts / counts.sum() * 100).round(1)
print(pd.DataFrame({"n": counts, "%": pct}).rename(index={0: "On track", 1: "Vulnerable"}))

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["On track", "Vulnerable"], counts.values, color=["#4C72B0", "#C44E52"])
for i, (n, p) in enumerate(zip(counts.values, pct.values)):
    ax.text(i, n + 50, f"{n}\n({p}%)", ha="center")
ax.set_ylabel("Number of children")
ax.set_title("Developmental status (ECDI2030)")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.savefig("figures/eda_01_target.png", dpi=150)
plt.show()

# %% 2.4 Descriptive statistics
df.describe().T.round(2)


# %% 2.5 Table 1 - characteristics by developmental status
categorical = ["sex_female", "low_birth_weight", "recent_illness", "fully_vaccinated", "mother_education",
               "region", "rural", "wealth_quintile", "improved_water", "improved_sanitation",
               "father_engaged", "children_books", "inadequate_supervision", "violent_discipline",
               "ecde_attendance", "playthings", "early_stimulation"]
continuous = ["age_months", "haz", "whz", "mother_age", "household_size"]

rows = []
# Continuous variables: mean (SD) + Mann-Whitney U test
for col in continuous:
    a = df.loc[df[TARGET] == 0, col].dropna()
    b = df.loc[df[TARGET] == 1, col].dropna()
    p = stats.mannwhitneyu(a, b).pvalue
    rows.append({"Variable": col, "Level": "mean (SD)",
                 "On track": f"{a.mean():.1f} ({a.std():.1f})",
                 "Vulnerable": f"{b.mean():.1f} ({b.std():.1f})",
                 "% vulnerable": "", "p-value": p})

# Categorical variables: n (%) + chi-square test
for col in categorical:
    tab = pd.crosstab(df[col], df[TARGET])
    p = stats.chi2_contingency(tab).pvalue
    for lvl in tab.index:
        n0, n1 = tab.loc[lvl, 0], tab.loc[lvl, 1]
        rows.append({"Variable": col, "Level": lvl,
                     "On track": f"{n0} ({n0 / tab[0].sum() * 100:.1f}%)",
                     "Vulnerable": f"{n1} ({n1 / tab[1].sum() * 100:.1f}%)",
                     "% vulnerable": f"{n1 / (n0 + n1) * 100:.1f}", "p-value": p})

table1 = pd.DataFrame(rows)
table1["p-value"] = table1["p-value"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
table1.loc[table1.Variable.duplicated(), ["Variable", "p-value"]] = ""
table1.to_csv("figures/table1_characteristics.csv", index=False)
table1