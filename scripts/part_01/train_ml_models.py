
# train_ml_models.py

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Suppress convergence warnings — handled explicitly below
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
file_path = "diabetes_clean.csv"
df = pd.read_csv(file_path)

print(f"Dataset shape : {df.shape}")
print(f"Target counts :\n{df['Outcome'].value_counts()}\n")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. SCALE FEATURES
# ─────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. DEFINE CLASSIFIERS
#
# LogisticRegression:
#   solver="lbfgs"   — default, good for small/medium datasets
#   max_iter=2000    — doubled from 1000 to ensure convergence
#   tol=1e-4         — explicit convergence tolerance
#
# KNeighborsClassifier:
#   n_neighbors=5    — standard starting point
#   metric="minkowski" — default (equivalent to Euclidean for p=2)
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        solver   = "lbfgs",
        max_iter = 2000,
        tol      = 1e-4,
        random_state = 42,
    ),
    "KNN (k=5)": KNeighborsClassifier(
        n_neighbors = 5,
        metric      = "minkowski",
        p           = 2,
    ),
}

# ─────────────────────────────────────────────
# 5. TRAIN, EVALUATE, REPORT
# ─────────────────────────────────────────────
results = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = acc

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["No Diabetes (0)", "Diabetes (1)"],
        digits=3,
    ))

    # ── Confusion matrix plot ──────────────────────────────────────
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = ["No Diabetes", "Diabetes"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAccuracy: {acc:.4f}", fontsize=12, pad=12)
    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=","")
    plt.savefig(f"confusion_{safe_name}.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → confusion_{safe_name}.png")

# ─────────────────────────────────────────────
# 6. SUMMARY COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 52)
print("  SUMMARY — Model Accuracy Comparison")
print("=" * 52)

results_df = (
    pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
    .sort_values("Accuracy", ascending=False)
)
results_df["Accuracy %"] = (results_df["Accuracy"] * 100).round(2).astype(str) + "%"
print(results_df.to_string())

# ── Bar chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(
    results_df.index,
    results_df["Accuracy"],
    color=["#4C72B0", "#DD8452"],
    edgecolor="white",
    height=0.5,
)
ax.set_xlim(0, 1)
ax.set_xlabel("Accuracy", fontsize=11)
ax.set_title("Model Accuracy Comparison", fontsize=13, pad=12)
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax.text(
        row["Accuracy"] + 0.01, i,
        f"{row['Accuracy']*100:.2f}%",
        va="center", fontsize=10,
    )
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("\nComparison chart saved → model_comparison.png")






                        