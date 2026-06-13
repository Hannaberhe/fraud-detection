import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Load model and data
rf = joblib.load('models/best_rf_model.pkl')
df = pd.read_csv('data/processed/fraud_with_features.csv')
features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 'user_txn_count', 'ip_int']
X = df[features].fillna(0)
y = df['class']

# Sample for SHAP (use 500 rows for speed)
X_sample = X.sample(500, random_state=42)

# SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# Summary plot
print("Creating summary plot...")
shap.summary_plot(shap_values[1], X_sample, show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
print("Summary plot saved!")

# Feature importance from model
print("\nBuilt-in Feature Importance:")
importances = rf.feature_importances_
for i, feat in enumerate(features):
    print(f"  {feat}: {importances[i]:.4f}")

# Top 5 SHAP features
print("\nTop 5 Fraud Drivers:")
shap_mean = np.abs(shap_values[1]).mean(axis=0)
top5_idx = np.argsort(shap_mean)[-5:][::-1]
for i in top5_idx:
    print(f"  {features[i]}")

print("\n BUSINESS RECOMMENDATIONS ")
print("1. Flag transactions within 24 hours of signup for extra verification")
print("2. Monitor high-value purchases during late night hours")
print("3. Track users with unusually high transaction counts in short periods")
print("\nDone!")
