"""
Smartphone Usage & Addiction Prediction — 10-Step EDA
Target: addicted_label (0 = Not Addicted, 1 = Addicted)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
import sqlite3
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
OUT       = BASE_DIR / "data" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TARGET      = "addicted_label"
DROP_COLS   = ["transaction_id", "user_id", "addiction_level"]
NUM_FEATS   = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
               "work_study_hours", "sleep_hours", "notifications_per_day",
               "app_opens_per_day", "weekend_screen_time"]
CAT_FEATS   = ["gender", "stress_level", "academic_work_impact"]

df = pd.read_csv(DATA_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Shape & Info
# ══════════════════════════════════════════════════════════════════════════════
print("── STEP 1: HIGH-LEVEL VIEW ──")
print(f"Shape: {df.shape}")
df.info()
print(f"\nNumerical : {df.select_dtypes(include=np.number).columns.tolist()}")
print(f"Categorical: {df.select_dtypes(include='object').columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Data Cleaning
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 2: DATA CLEANING ──")
print(df.isnull().sum()[df.isnull().sum() > 0])          # nulls
print(f"Duplicates: {df.duplicated().sum()}")            # dupes
# addiction_level: 819 nulls but dropped as leakage. No imputation needed.
# Outlier analysis deferred to Step 8.


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Descriptive Statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 3: DESCRIPTIVE STATISTICS ──")
print(df.describe().round(2))
print(df.describe(include="all").round(2))
for col in CAT_FEATS:
    print(f"\n{col}:\n{df[col].value_counts()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Univariate Graphs
# ══════════════════════════════════════════════════════════════════════════════
# Numerical: histogram + boxplot
fig, axes = plt.subplots(len(NUM_FEATS), 2, figsize=(14, 3.5 * len(NUM_FEATS)))
for i, col in enumerate(NUM_FEATS):
    axes[i][0].hist(df[col], bins=30, edgecolor="white")
    axes[i][0].set_title(f"{col} — Distribution")
    axes[i][1].boxplot(df[col], vert=False, patch_artist=True)
    axes[i][1].set_title(f"{col} — Boxplot")
plt.suptitle("Step 4 — Univariate: Numerical", fontsize=13, fontweight="bold", y=1.002)
plt.tight_layout()
plt.savefig(OUT / "step4_univariate_numerical.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Categorical: count bars
fig, axes = plt.subplots(1, len(CAT_FEATS), figsize=(14, 4))
for i, col in enumerate(CAT_FEATS):
    df[col].value_counts().plot(kind="bar", ax=axes[i], edgecolor="white")
    axes[i].set_title(col)
    axes[i].tick_params(axis="x", rotation=15)
plt.suptitle("Step 4 — Univariate: Categorical", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step4_univariate_categorical.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Univariate Analysis (Written)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 5: UNIVARIATE ANALYSIS ──")
print("""
- All numerical features are near-uniformly distributed — strong signal
  that this dataset is synthetically generated.
- Real-world screen time would right-skew; sleep would cluster near 7-8hrs.
- All categorical features are artificially balanced (~equal splits).
- Takeaway: distributions themselves aren't informative here — the
  relationships between features and the target are what matter.
""")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Multivariate Graphs
# ══════════════════════════════════════════════════════════════════════════════
# 6A: Numerical correlation heatmap
corr = df[NUM_FEATS + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
            annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Step 6A — Correlation Heatmap (Numerical)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step6a_corr_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# 6B: Numerical features vs target (boxplots)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, col in zip(axes.flatten(), NUM_FEATS):
    bp = ax.boxplot([df[df[TARGET]==0][col], df[df[TARGET]==1][col]],
                    labels=["Not Addicted", "Addicted"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#55A868", "#C44E52"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(col)
plt.suptitle("Step 6B — Numerical Features vs. Target", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step6b_boxplots_vs_target.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# 6C: Categorical features vs target (stacked bar %)
fig, axes = plt.subplots(1, len(CAT_FEATS), figsize=(15, 5))
for ax, col in zip(axes, CAT_FEATS):
    pd.crosstab(df[col], df[TARGET], normalize="index").mul(100).plot(
        kind="bar", ax=ax, edgecolor="white", color=["#55A868", "#C44E52"])
    ax.set_title(f"{col} vs. Target"); ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=20)
plt.suptitle("Step 6C — Categorical Features vs. Target", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step6c_cat_vs_target.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# 6D: Full heatmap with factorized categoricals
df_fact = df[NUM_FEATS + CAT_FEATS + [TARGET]].copy()
for col in CAT_FEATS:
    df_fact[col] = pd.factorize(df_fact[col])[0]
corr_full = df_fact.corr()
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr_full, mask=np.triu(np.ones_like(corr_full, dtype=bool)),
            annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Step 6D — Full Heatmap (Categoricals Factorized)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step6d_corr_heatmap_full.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("\nTarget correlations (numerical):")
print(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Multivariate Analysis (Written)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 7: MULTIVARIATE ANALYSIS ──")
print("""
- daily_screen_time_hours (r=0.58) and weekend_screen_time (r=0.56) are by
  far the strongest predictors — total time on device is the clearest signal.
- social_media_hours (r=0.41) is a meaningful secondary predictor.
- Everything else (age, sleep, gaming, notifications, etc.) shows near-zero
  correlation with the target (r < 0.04).
- Categorical features (gender, stress, academic impact) show ~0 correlation
  with addiction — none meaningfully separate the classes.
- Caveat: the factorized heatmap assigns arbitrary integers to categories,
  so those correlations should be interpreted carefully.
- Bottom line: screen time is king. SelectKBest in Step 9 will confirm.
""")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Feature Engineering (Encoding, Outliers, Scaling)
# ══════════════════════════════════════════════════════════════════════════════
print("── STEP 8: FEATURE ENGINEERING ──")

df_model = df.drop(columns=DROP_COLS).copy()

# Outlier analysis (IQR)
print("\nOutlier summary:")
for col in NUM_FEATS:
    Q1, Q3 = df_model[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = ((df_model[col] < Q1 - 1.5*IQR) | (df_model[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col}: {n_out} outliers")

# Encoding
df_model["stress_level"]         = df_model["stress_level"].map({"Low":0, "Medium":1, "High":2})
df_model["academic_work_impact"] = df_model["academic_work_impact"].map({"No":0, "Yes":1})
df_model = pd.get_dummies(df_model, columns=["gender"], drop_first=True)
print(f"\nEncoded shape: {df_model.shape}")

# Scaling
X_raw = df_model.drop(columns=[TARGET])
y     = df_model[TARGET]
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_raw), columns=X_raw.columns)
print("StandardScaler applied.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Feature Selection (SelectKBest + Chi-Square)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 9: FEATURE SELECTION ──")

X_chi2 = X_raw.copy()
X_chi2[X_chi2.select_dtypes(include="bool").columns] = \
    X_chi2.select_dtypes(include="bool").astype(int)
for col in X_chi2.columns:
    if X_chi2[col].min() < 0:
        X_chi2[col] -= X_chi2[col].min()

selector = SelectKBest(chi2, k="all").fit(X_chi2, y)
chi2_df = pd.DataFrame({"Feature": X_raw.columns,
                         "Chi2": selector.scores_.round(2),
                         "p-value": selector.pvalues_.round(4)
                        }).sort_values("Chi2", ascending=False).reset_index(drop=True)
print(chi2_df.to_string())

SELECTED = chi2_df[chi2_df["p-value"] < 0.05]["Feature"].tolist()
print(f"\nSelected features (p < 0.05): {SELECTED}")

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#55A868" if p < 0.05 else "#AAAAAA" for p in chi2_df["p-value"]]
ax.barh(chi2_df["Feature"], chi2_df["Chi2"], color=colors, edgecolor="white")
ax.set_title("Step 9 — Feature Selection: Chi-Square Scores\n(green = p < 0.05)",
             fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT / "step9_feature_selection.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Train/Test Split + Save to SQLite
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 10: TRAIN/TEST SPLIT + SQLITE ──")

X_final = X_scaled[SELECTED]
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"y_train: {y_train.shape} | y_test:  {y_test.shape}")

train_df = X_train.copy(); train_df[TARGET] = y_train.values
test_df  = X_test.copy();  test_df[TARGET]  = y_test.values

DB_PATH = OUT / "smartphone_addiction.db"
conn = sqlite3.connect(DB_PATH)
train_df.to_sql("train_data", conn, if_exists="replace", index=False)
test_df.to_sql("test_data",   conn, if_exists="replace", index=False)
conn.close()
print(f"Saved to SQLite: {DB_PATH}")

print("\n✓ EDA complete. Ready for modeling.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — Logistic Regression Model 
# ══════════════════════════════════════════════════════════════════════════════

# ── Load ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUT      = BASE_DIR / "data" / "outputs"

conn     = sqlite3.connect(OUT / "smartphone_addiction.db")
train_df = pd.read_sql("SELECT * FROM train_data", conn)
test_df  = pd.read_sql("SELECT * FROM test_data",  conn)
conn.close()

TARGET  = "addicted_label"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

# ── GridSearch ────────────────────────────────────────────────────────────────
# C          : regularization strength — lower = stronger regularization
# l1_ratio   : 0 = pure L2 (shrinks coefficients), 1 = pure L1 (zeros out weak features)
# class_weight: balanced corrects for the 70/30 class imbalance
param_grid = {
    "C":            [0.01, 0.1, 1, 10, 100],
    "l1_ratio":     [0, 0.5, 1],
    "class_weight": ["balanced", None],
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42, solver="saga", penalty="elasticnet"),
    param_grid, cv=5, scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params : {grid.best_params_}")
print(f"Best CV AUC : {grid.best_score_:.4f}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
model  = grid.best_estimator_
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Addicted", "Addicted"]))
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Addicted", "Addicted"],
            yticklabels=["Not Addicted", "Addicted"], ax=axes[0])
axes[0].set_title("Confusion Matrix", fontweight="bold")
axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
axes[1].set_title("ROC Curve", fontweight="bold")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# Coefficients
coef_df = pd.DataFrame({"Feature": X_train.columns,
                         "Coefficient": model.coef_[0]}
                       ).sort_values("Coefficient", ascending=False)
colors = ["#55A868" if c > 0 else "#C44E52" for c in coef_df["Coefficient"]]
axes[2].barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white")
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("Feature Coefficients", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


import pickle

# Save model and scaler to disk
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

with open(MODELS_DIR / "smartphone_addiction_model.sav", "wb") as f:
    pickle.dump(model, f)

# Re-fit scaler on selected features only so app.py can scale raw user input
scaler = StandardScaler().fit(X_raw[SELECTED])
with open(MODELS_DIR / "scaler.sav", "wb") as f:
    pickle.dump(scaler, f)

print("Model saved → models/smartphone_addiction_model.sav")
print("Scaler saved → models/scaler.sav")
print("\n✓ Done.")

