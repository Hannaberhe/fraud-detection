import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

rf = joblib.load('models/best_rf_model.pkl')
df = pd.read_csv('data/processed/fraud_with_features.csv')
features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 'user_txn_count', 'ip_int']
X = df[features].fillna(0)

X_sample = X.sample(200, random_state=42)

print("Building SHAP explainer")
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X_sample)

if isinstance(shap_vals, list):
    sv = shap_vals[1]
else:
    sv = shap_vals

print("Creating summary plot")
shap.summary_plot(sv, X_sample, feature_names=features, show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
print("Summary plot saved")

# Top features
print("\nTop fraud drivers by SHAP:")
mean_shap = np.abs(sv).mean(axis=0)
for i in range(len(features)):
    print(f"  {features[i]}: {mean_shap[i]:.4f}")

print("\nDone")
