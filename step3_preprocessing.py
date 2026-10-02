# %% 3.1 Load data and define features
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SEED = 42
os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("ecd_kenya_simulated.csv")
TARGET = "vulnerable"
CATEGORICAL = ["region"]
NUMERIC = [c for c in df.columns if c not in CATEGORICAL + [TARGET]]
FEATURES = NUMERIC + CATEGORICAL

X = df[FEATURES]
y = df[TARGET]
print(f"{len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical = {len(FEATURES)} predictors")
print("Numeric:", NUMERIC)

# %% 3.2 Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

summary = pd.DataFrame({
    "Children": [len(y_train), len(y_test)],
    "Vulnerable": [y_train.sum(), y_test.sum()],
    "% vulnerable": [round(y_train.mean() * 100, 1), round(y_test.mean() * 100, 1)]},
    index=["Training set", "Test set"])
print(summary)

# %% 3.3 Build the preprocessing pipeline
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # fill gaps with the training median
    ("scale", StandardScaler()),                    # mean 0, SD 1
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL),
])
preprocessor


# %% 3.4 Fit on training data ONLY, then transform both sets
X_train_prep = preprocessor.fit_transform(X_train)   # learn medians/means from TRAIN
X_test_prep = preprocessor.transform(X_test)         # apply the same values to TEST

feature_names = preprocessor.get_feature_names_out()
print("Shape before:", X_train.shape, "-> after:", X_train_prep.shape)
print("Missing values left - train:", np.isnan(X_train_prep).sum(), "| test:", np.isnan(X_test_prep).sum())
print("\nNew columns created from region:", [f for f in feature_names if f.startswith("cat__")])

check = pd.DataFrame(X_train_prep, columns=feature_names)
print("\nAfter scaling (training set): mean ~0, SD ~1")
print(check[["num__age_months", "num__haz", "num__wealth_quintile"]].describe().loc[["mean", "std"]].round(2))

# %% 3.5 Save everything for the next steps
joblib.dump({
    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    "preprocessor": preprocessor, "NUMERIC": NUMERIC, "CATEGORICAL": CATEGORICAL,
    "FEATURES": FEATURES, "SEED": SEED,
}, "artifacts/step3_data.joblib")
print("Saved: artifacts/step3_data.joblib")