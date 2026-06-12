import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/processed/fraud_with_features.csv')
features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 'user_txn_count', 'ip_int']
X = df[features].fillna(0)
y = df['class']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Fraud rate:", round(y.mean()*100, 2), "%")

# Logistic Regression baseline
print("\n=== Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]
print(f"F1: {f1_score(y_test, lr_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, lr_prob):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# Random Forest
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
print(f"F1: {f1_score(y_test, rf_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, rf_prob):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Cross validation
print("\n=== Cross Validation (5-fold) ===")
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print(f"RF CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Model comparison
print("\n=== MODEL COMPARISON ===")
print(f"Model              F1      AUC")
print(f"Logistic Regression  {f1_score(y_test, lr_pred):.4f}   {roc_auc_score(y_test, lr_prob):.4f}")
print(f"Random Forest        {f1_score(y_test, rf_pred):.4f}   {roc_auc_score(y_test, rf_prob):.4f}")

# Save best model
joblib.dump(rf, 'models/best_rf_model.pkl')
print("\nBest model saved to models/best_rf_model.pkl")
print("Done")
