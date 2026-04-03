# =====================================================================
# PERSONALITY PREDICTOR — MODEL TRAINING PIPELINE
# Dataset : personality_dataset.csv
# Target  : Personality (Introvert / Extrovert)
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  PERSONALITY PREDICTION — TRAINING PIPELINE")
print("=" * 55)

df = pd.read_csv("personality_dataset.csv")
print(f"\n[DATA]  Shape  : {df.shape}")
print(f"[DATA]  Nulls  :\n{df.isnull().sum()}")
print(f"\n[DATA]  Class distribution:\n{df['Personality'].value_counts()}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

# 2a. Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n[PREP]  Dropped {before - len(df)} duplicate rows → {len(df)} remain")

# 2b. Clip numeric outliers using IQR
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance',
                'Going_outside', 'Friends_circle_size', 'Post_frequency']

for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    clipped = df[col].clip(lower, upper)
    n_clipped = (df[col] != clipped).sum()
    df[col] = clipped
    if n_clipped:
        print(f"[PREP]  Clipped {n_clipped} outliers in '{col}'")

# 2c. Feature engineering
df['Social_score']     = df['Social_event_attendance'] + df['Going_outside']
df['Alone_ratio']      = df['Time_spent_Alone'] / (df['Social_score'] + 1)
df['Network_activity'] = df['Friends_circle_size'] * df['Post_frequency']
print("\n[PREP]  Engineered 3 features: Social_score, Alone_ratio, Network_activity")

# 2d. Encode binary categoricals
le_stage   = LabelEncoder()
le_drained = LabelEncoder()
df['Stage_fear']                = le_stage.fit_transform(df['Stage_fear'])
df['Drained_after_socializing'] = le_drained.fit_transform(df['Drained_after_socializing'])

# 2e. Encode target
le_target = LabelEncoder()
df['Personality'] = le_target.fit_transform(df['Personality'])
print(f"[PREP]  Target classes: {le_target.classes_}")

# ─────────────────────────────────────────────
# 3. FEATURES & TARGET
# ─────────────────────────────────────────────
X = df.drop('Personality', axis=1)
y = df['Personality']
print(f"\n[FEAT]  {len(X.columns)} features: {X.columns.tolist()}")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (stratified)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 6. BASELINE MODEL COMPARISON (5-fold CV)
# ─────────────────────────────────────────────
print("\n[BASELINE] 5-fold Stratified CV comparison …")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
}
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_train_sc, y_train, cv=skf, scoring='accuracy', n_jobs=1)
    print(f"  {name:<25}  CV Acc: {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING — Random Forest
# ─────────────────────────────────────────────
print("\n[TUNE]  RandomizedSearchCV on Random Forest …")

param_grid = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2'],
    'class_weight'     : ['balanced', None],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, n_iter=15, cv=skf,
    scoring='accuracy', random_state=42, n_jobs=1, verbose=0,
)
search.fit(X_train_sc, y_train)
print(f"[TUNE]  Best params : {search.best_params_}")
print(f"[TUNE]  Best CV Acc : {search.best_score_:.4f}")

model = search.best_estimator_

# ─────────────────────────────────────────────
# 8. FINAL TEST EVALUATION
# ─────────────────────────────────────────────
y_pred      = model.predict(X_test_sc)
y_pred_prob = model.predict_proba(X_test_sc)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 55)
print("  TEST RESULTS")
print("=" * 55)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  Precision : {tp/(tp+fp):.4f}")
print(f"  Recall    : {tp/(tp+fn):.4f}")
print(f"  F1 Score  : {2*tp/(2*tp+fp+fn):.4f}")
print("=" * 55)
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# ─────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)\
    .plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix", fontsize=13, fontweight='bold')

feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
feat_imp.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title("Feature Importances", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
plt.show()
print("Saved → model_evaluation.png")

# ─────────────────────────────────────────────
# 10. SAVE ARTIFACTS
# ─────────────────────────────────────────────
pickle.dump(model,     open("personality_model.pkl", "wb"))
pickle.dump(scaler,    open("scaler.pkl",             "wb"))
pickle.dump(le_target, open("label_encoder.pkl",      "wb"))

# Save feature column order so app.py can reconstruct correctly
pickle.dump(list(X.columns), open("feature_columns.pkl", "wb"))

print("\n✅ Saved: personality_model.pkl | scaler.pkl | label_encoder.pkl | feature_columns.pkl")
print("   Run the app with: streamlit run app.py")