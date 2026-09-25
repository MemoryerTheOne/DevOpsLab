import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# ============================================================
# [1/9] Mount Google Drive and load dataset
# ============================================================
print("[1/9] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Update this path to match where your file actually sits in Drive
DATA_PATH = "/content/drive/MyDrive/games.csv"

print("[2/9] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# [3/9] Fill empty rows (handle missing values)
# ============================================================
print("[3/9] Filling missing values...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    median_val = df[col].median()
    if pd.isna(median_val):
        print(f"      Column '{col}' is entirely empty — filling with 0 instead of median")
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(median_val)

text_cols = df.select_dtypes(include=["object"]).columns.tolist()
df[text_cols] = df[text_cols].fillna("Unknown")

print(f"      Missing values remaining: {df.isnull().sum().sum()}")

# ============================================================
# [4/9] Build the success / not-success label
# ============================================================
print("[4/9] Building target label (success)...")
df["total_reviews"] = df["Positive"] + df["Negative"]
df = df[df["total_reviews"] > 0]  # can't judge games with zero reviews
df["positive_ratio"] = df["Positive"] / df["total_reviews"]
df["success"] = (df["positive_ratio"] >= 0.7).astype(int)

print(f"      {len(df)} rows remain after dropping unreviewed games")
print(f"      Class balance -> success=1: {df['success'].sum()}, success=0: {(df['success']==0).sum()}")

# ============================================================
# [5/9] Auto-extract features (no manual feature list)
# ============================================================
print("[5/9] Auto-detecting features...")

# Columns to exclude: identifiers + anything that leaks the target
exclude_cols = [
    "AppID", "Name",
    "Positive", "Negative", "total_reviews", "positive_ratio", "success",
    "Score rank", "Metacritic score", "Metacritic url", "User score",
    "Recommendations",
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_cols if col not in exclude_cols]
print(f"      Auto-selected {len(features)} features: {features}")

X = df[features]
y = df["success"]

print("      NaN check before training:", X.isnull().sum().sum())

# ============================================================
# [6/9] Split 80/20 and normalize
# ============================================================
print("[6/9] Splitting train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train size: {len(X_train)} ({len(X_train)/len(X):.0%}), Test size: {len(X_test)} ({len(X_test)/len(X):.0%})")

print("      Normalizing features (Min-Max scaling to 0-1 range)...")
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
print("      Done")

# ============================================================
# [7/9] Model 1: Gradient Boosting
# ============================================================
print("[7/9] Training Gradient Boosting model...")
gbc = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    verbose=1,
)
gbc.fit(X_train_norm, y_train)
print("      Training complete. Predicting on test set...")
gbc_preds = gbc.predict(X_test_norm)
gbc_recall = recall_score(y_test, gbc_preds)
print(f"      Gradient Boosting TEST recall: {gbc_recall:.4f}")

gbc_train_preds = gbc.predict(X_train_norm)
gbc_train_recall = recall_score(y_train, gbc_train_preds)
print(f"      Gradient Boosting TRAIN recall: {gbc_train_recall:.4f}")

# ============================================================
# [8/9] Model 2: Cosine similarity classifier
# ============================================================
print("[8/9] Running cosine similarity classifier...")

def cosine_knn_predict(X_train, y_train, X_test, k=10, log_every=200):
    n = X_test.shape[0]
    sims = cosine_similarity(X_test, X_train)
    top_k_idx = np.argsort(-sims, axis=1)[:, :k]
    preds = []
    for i, idx_row in enumerate(top_k_idx):
        neighbor_labels = y_train.iloc[idx_row]
        preds.append(neighbor_labels.mode()[0])
        if (i + 1) % log_every == 0 or (i + 1) == n:
            print(f"      Classified {i + 1}/{n} test samples")
    return np.array(preds)

cosine_preds = cosine_knn_predict(X_train_norm, y_train, X_test_norm, k=10)
cosine_recall = recall_score(y_test, cosine_preds)
print(f"      Cosine similarity recall: {cosine_recall:.4f}")

# --- Results ---
print("\nGradient Boosting report:\n", classification_report(y_test, gbc_preds))
print("\nCosine similarity report:\n", classification_report(y_test, cosine_preds))

# ============================================================
# [9/9] Charts
# ============================================================
print("[9/9] Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

cm_gbc = confusion_matrix(y_test, gbc_preds)
sns.heatmap(cm_gbc, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["Not Success (0)", "Success (1)"],
            yticklabels=["Not Success (0)", "Success (1)"])
axes[0, 0].set_title("Gradient Boosting - Confusion Matrix")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("Actual")

cm_cos = confusion_matrix(y_test, cosine_preds)
sns.heatmap(cm_cos, annot=True, fmt="d", cmap="Greens", ax=axes[0, 1],
            xticklabels=["Not Success (0)", "Success (1)"],
            yticklabels=["Not Success (0)", "Success (1)"])
axes[0, 1].set_title("Cosine Similarity - Confusion Matrix")
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("Actual")

recall_gbc_per_class = recall_score(y_test, gbc_preds, average=None)
recall_cos_per_class = recall_score(y_test, cosine_preds, average=None)

x = np.arange(2)
width = 0.35
axes[1, 0].bar(x - width/2, recall_gbc_per_class, width, label="Gradient Boosting")
axes[1, 0].bar(x + width/2, recall_cos_per_class, width, label="Cosine Similarity")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(["Class 0 (Not Success)", "Class 1 (Success)"])
axes[1, 0].set_ylabel("Recall")
axes[1, 0].set_title("Recall by Class")
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1)

class_counts = y.value_counts().sort_index()
axes[1, 1].bar(["Not Success (0)", "Success (1)"], class_counts.values, color=["#d9534f", "#5cb85c"])
axes[1, 1].set_title("Class Distribution (Full Dataset)")
axes[1, 1].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[1, 1].text(i, v + 50, str(v), ha="center")

plt.tight_layout()
plt.savefig("model_comparison_charts.png", dpi=150)
plt.show()
print("      Charts saved to model_comparison_charts.png")