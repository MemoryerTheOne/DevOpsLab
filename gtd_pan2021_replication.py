# -*- coding: utf-8 -*-
"""
GTD terrorist-organization prediction — replication of Pan (2021), version 1.

Paper:   Pan, X. (2021). Quantitative Analysis and Prediction of Global Terrorist
         Attacks Based on Machine Learning. Scientific Programming, 2021, 7890923.
         https://doi.org/10.1155/2021/7890923
Dataset: Global Terrorism Database, 1970-2017 (Kaggle: START-UMD/gtd,
         file globalterrorismdb_0718dist.csv, ~181,691 rows x 135 columns)

Paper setup reproduced here:
  * target   = responsible terrorist organization (gname)
  * classes  = the 32 organizations with the most attacks ("Unknown" excluded)
  * features = selected with SelectKBest
  * models   = Decision Tree, Bagging, Random Forest, Extra Trees, XGBoost
  * eval     = 10-fold cross-validation
  * reported = XGBoost 97.16% accuracy, Random Forest 96.82% accuracy

How to run in Google Colab:
  1. Upload this file (or paste it into a cell) and run it:  !python gtd_pan2021_replication.py
     or open it as cells - the "# %%" markers split it into cells.
  2. The dataset is downloaded automatically with kagglehub. If that fails,
     upload globalterrorismdb_0718dist.csv to /content and run again.
"""

# %% 0. Install / imports
import subprocess
import sys

for pkg in ("xgboost", "kagglehub"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# %% 1. Configuration
RANDOM_STATE = 42
TOP_N_GROUPS = 32          # paper: 32 most active organizations
N_SPLITS = 10              # paper: 10-fold cross-validation
K_BEST = 25                # number of features kept by SelectKBest
CSV_NAME = "globalterrorismdb_0718dist.csv"

# Accuracy (%) reported in the paper's abstract; None = not available to us yet
PAPER_ACCURACY = {
    "Decision Tree": None,
    "Bagging": None,
    "Random Forest": 96.82,
    "Extra Trees": None,
    "XGBoost": 97.16,
}


# %% 2. Load the dataset
def find_csv():
    """Return the path to the GTD CSV: local upload first, then kagglehub."""
    for path in (CSV_NAME, os.path.join("/content", CSV_NAME)):
        if os.path.exists(path):
            return path
    import kagglehub

    folder = kagglehub.dataset_download("START-UMD/gtd")
    matches = glob.glob(os.path.join(folder, "**", CSV_NAME), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{CSV_NAME} not found in {folder}")
    return matches[0]


csv_path = find_csv()
print(f"Loading {csv_path}")
try:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", low_memory=False)
except pd.errors.ParserError:
    size_mb = os.path.getsize(csv_path) / 1e6
    sys.exit(
        f"Could not parse {csv_path} ({size_mb:.0f} MB). The file looks incomplete - "
        "if you uploaded it to Colab, wait until the upload finishes (full file is ~163 MB) and run again."
    )
print(f"Full dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

# %% 3. Select the 32 most active known organizations
known = df[df["gname"] != "Unknown"]
top_groups = known["gname"].value_counts().head(TOP_N_GROUPS)
data = known[known["gname"].isin(top_groups.index)].copy()

print(f"\nTop {TOP_N_GROUPS} organizations: {len(data):,} attacks")
print(top_groups.to_string())

# %% 4. Feature engineering
# Coded (numeric) GTD variables describing when, where and how the attack happened.
# Free text, *_txt duplicates, perpetrator fields (gname2, gsubname, ...) and
# post-hoc fields are excluded to avoid leaking the answer.
NUMERIC_FEATURES = [
    "iyear", "imonth", "iday", "extended", "country", "region",
    "latitude", "longitude", "specificity", "vicinity",
    "crit1", "crit2", "crit3", "doubtterr", "multiple", "success", "suicide",
    "attacktype1", "attacktype2", "targtype1", "targsubtype1", "natlty1",
    "weaptype1", "weapsubtype1", "nperps", "claimed",
    "nkill", "nkillter", "nwound", "nwoundte",
    "property", "propextent", "ishostkid", "nhostkid", "ransom",
    "INT_LOG", "INT_IDEO", "INT_MISC", "INT_ANY",
]
# Location text fields, label-encoded so tree models can split on them
CATEGORICAL_FEATURES = ["provstate", "city"]

features = [c for c in NUMERIC_FEATURES if c in data.columns]
X = data[features].apply(pd.to_numeric, errors="coerce")

# GTD uses -9 / -99 as "unknown" codes; treat them as missing
X = X.replace([-9, -99], np.nan)
for col in X.columns:
    X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

for col in CATEGORICAL_FEATURES:
    X[col] = LabelEncoder().fit_transform(data[col].fillna("Unknown").astype(str))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["gname"])

print(f"\nFeature matrix: {X.shape[0]:,} rows x {X.shape[1]} features, {len(label_encoder.classes_)} classes")

# %% 5. Models (the five classifiers from the paper)
import shutil

# Use the Colab GPU for XGBoost when one is attached
xgb_device = "cuda" if shutil.which("nvidia-smi") else "cpu"

MODELS = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device=xgb_device,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

# %% 6. 10-fold cross-validation
# SelectKBest sits inside the pipeline so features are chosen on each training fold only.
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
}

rows = []
for name, model in MODELS.items():
    pipeline = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=min(K_BEST, X.shape[1]))),
        ("model", model),
    ])
    print(f"\nTraining {name} ({N_SPLITS}-fold CV)...")
    start = time.time()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
    elapsed = time.time() - start

    acc = scores["test_accuracy"] * 100
    row = {
        "Model": name,
        "Accuracy (%)": acc.mean(),
        "Acc std": acc.std(),
        "Precision (%)": scores["test_precision"].mean() * 100,
        "Recall (%)": scores["test_recall"].mean() * 100,
        "F1 (%)": scores["test_f1"].mean() * 100,
        "Paper accuracy (%)": PAPER_ACCURACY[name],
        "Time (s)": elapsed,
    }
    paper = PAPER_ACCURACY[name]
    row["Diff vs paper"] = row["Accuracy (%)"] - paper if paper is not None else None
    rows.append(row)
    print(f"  accuracy {acc.mean():.2f}% (+/- {acc.std():.2f})  in {elapsed:.0f}s")

# %% 7. Results
results = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)
pd.set_option("display.width", 160)
print("\n" + "=" * 100)
print(f"Results: top {TOP_N_GROUPS} GTD organizations, {N_SPLITS}-fold CV, SelectKBest k={K_BEST}")
print("=" * 100)
print(results.to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep="-"))

results.to_csv("pan2021_replication_results.csv", index=False)
print("\nSaved pan2021_replication_results.csv")

# Features SelectKBest keeps when fitted on the full data (for the report)
selector = SelectKBest(score_func=f_classif, k=min(K_BEST, X.shape[1])).fit(X, y)
kept = (
    pd.Series(selector.scores_, index=X.columns)
    .loc[selector.get_support()]
    .sort_values(ascending=False)
)
print(f"\nSelected features (k={K_BEST}), by F-score:")
print(kept.round(1).to_string())
