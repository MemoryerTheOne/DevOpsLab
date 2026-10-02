# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

#added by irfan

# ============================================================
# [1/6] Load dataset
# ============================================================
DATA_PATH = Path(r"games.csv")

print("[1/6] Checking dataset path...")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# [2/6] Handle missing values
# ============================================================
print("[2/6] Filling missing values...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(0 if pd.isna(median_val) else median_val)

text_cols = df.select_dtypes(include=["object"]).columns.tolist()
df[text_cols] = df[text_cols].fillna("Unknown")
print(f"      Missing values remaining: {df.isnull().sum().sum()}")

# ============================================================
# [3/6] Build both target labels
# ============================================================
print("[3/6] Building target labels...")
ENGAGE_HOURS = 10          # playtime columns are in minutes
SUCCESS_THRESHOLD = 0.7

df["total_reviews"] = df["Positive"] + df["Negative"]
df["positive_ratio"] = np.where(df["total_reviews"] > 0,
                                df["Positive"] / df["total_reviews"], np.nan)
df["success"] = (df["positive_ratio"] >= SUCCESS_THRESHOLD).astype(int)
df["engaging"] = (df["Median playtime forever"] / 60 >= ENGAGE_HOURS).astype(int)

# Optional: top-50 Steam tags as 0/1 features (all known at launch)
USE_TAGS = True
if USE_TAGS and "Tags" in df.columns:
    tag_sets = df["Tags"].apply(lambda s: {t.strip() for t in s.split(",")})
    top_tags = [t for t, _ in Counter(t for s in tag_sets for t in s).most_common(50)
                if t != "Unknown"]
    tag_df = pd.DataFrame({f"tag_{t}": tag_sets.apply(lambda s, t=t: int(t in s))
                           for t in top_tags}, index=df.index)
    df = pd.concat([df, tag_df], axis=1)
    print(f"      Added {len(top_tags)} tag features")

# ============================================================
# [4/6] Per-target configuration
# ============================================================
# Columns never used as features for either target
COMMON_EXCLUDE = [
    "AppID", "Name",
    "total_reviews", "positive_ratio", "success", "engaging",
    "Score rank", "Metacritic score", "Metacritic url", "User score",
    "Recommendations",
]
# Extra columns that would leak the engagement label
PLAYTIME_LEAKS = [
    "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks", "Peak CCU",
]
# Review counts leak the engagement label too (and success is built from them)
REVIEW_COLS = ["Positive", "Negative"]

TARGETS = {
    "success": {
        "filter": lambda d: d["total_reviews"] > 0,   # drop unreviewed games
        "exclude": COMMON_EXCLUDE + REVIEW_COLS,
        "labels": ["Not Success (0)", "Success (1)"],
        "title": "Success (positive ratio >= 70%)",
    },
    "engaging": {
        "filter": lambda d: d["Median playtime forever"] > 0,  # need playtime data
        "exclude": COMMON_EXCLUDE + REVIEW_COLS + PLAYTIME_LEAKS,
        "labels": ["Not Engaging (0)", "Engaging (1)"],
        "title": f"Engaging (median playtime >= {ENGAGE_HOURS}h)",
    },
}

# ============================================================
# [5/6] Train and evaluate both models for one target
# ============================================================
def run_target(target, cfg):
    print("\n" + "=" * 60)
    print(f"TARGET: {cfg['title']}")
    print("=" * 60)

    d = df[cfg["filter"](df)].copy()
    num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c not in cfg["exclude"]]
    X, y = d[features], d[target]

    print(f"      {len(d)} rows | {len(features)} features")
    print(f"      Class balance -> 1: {y.sum()}, 0: {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Model 1: Gradient Boosting
    print("      Training Gradient Boosting...")
    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
    )
    gbc.fit(X_train_norm, y_train)
    gbc_preds = gbc.predict(X_test_norm)

    # Model 2: Random Forest
    print("      Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf.fit(X_train_norm, y_train)
    rf_preds = rf.predict(X_test_norm)

    res = {
        "GB test recall": recall_score(y_test, gbc_preds),
        "GB train recall": recall_score(y_train, gbc.predict(X_train_norm)),
        "RF test recall": recall_score(y_test, rf_preds),
        "RF train recall": recall_score(y_train, rf.predict(X_train_norm)),
    }
    for k, v in res.items():
        print(f"      {k}: {v:.4f}")

    print("\nGradient Boosting report:\n", classification_report(y_test, gbc_preds))
    print("\nRandom Forest report:\n", classification_report(y_test, rf_preds))

    # ---- Charts ----
    labels = cfg["labels"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(cfg["title"], fontsize=15)

    sns.heatmap(confusion_matrix(y_test, gbc_preds), annot=True, fmt="d", cmap="Blues",
                ax=axes[0, 0], xticklabels=labels, yticklabels=labels)
    axes[0, 0].set_title("Gradient Boosting - Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt="d", cmap="Greens",
                ax=axes[0, 1], xticklabels=labels, yticklabels=labels)
    axes[0, 1].set_title("Random Forest - Confusion Matrix")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")

    x = np.arange(2)
    width = 0.35
    axes[0, 2].bar(x - width / 2, recall_score(y_test, gbc_preds, average=None),
                   width, label="Gradient Boosting")
    axes[0, 2].bar(x + width / 2, recall_score(y_test, rf_preds, average=None),
                   width, label="Random Forest")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(["Class 0", "Class 1"])
    axes[0, 2].set_ylabel("Recall")
    axes[0, 2].set_title("Recall by Class")
    axes[0, 2].legend()
    axes[0, 2].set_ylim(0, 1)

    def plot_importances(ax, model, title, color):
        imp = pd.Series(model.feature_importances_, index=features).nlargest(10).sort_values()
        ax.barh(imp.index, imp.values, color=color)
        ax.set_title(title)
        ax.set_xlabel("Importance")

    plot_importances(axes[1, 0], gbc, "Gradient Boosting - Top 10 Features", "#337ab7")
    plot_importances(axes[1, 1], rf, "Random Forest - Top 10 Features", "#5cb85c")

    counts = y.value_counts().sort_index()
    axes[1, 2].bar(labels, counts.values, color=["#d9534f", "#5cb85c"])
    axes[1, 2].set_title("Class Distribution")
    axes[1, 2].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[1, 2].text(i, v + counts.max() * 0.01, str(v), ha="center")

    plt.tight_layout()
    out = f"charts_{target}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"      Charts saved to {out}")
    return res

# ============================================================
# [6/6] Run both targets and summarize
# ============================================================
print("[4-6/6] Running experiments...")
summary = {t: run_target(t, cfg) for t, cfg in TARGETS.items()}

print("\n" + "=" * 60)
print("SUMMARY (recall on positive class)")
print("=" * 60)
print(pd.DataFrame(summary).T.round(4))
