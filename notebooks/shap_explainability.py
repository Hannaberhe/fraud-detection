import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

rf = joblib.load('models/best_rf_model.pkl')
df = pd.read_csv('data/processed/fraud_with_features.csv')
features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 'user_txn_count', 'ip_int']
X = df[features].fillna(0)

print("Built-in Feature Importance:")
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)
print(feat_imp)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_imp['feature'], feat_imp['importance'], color='teal')
ax.set_title('Feature Importance - Random Forest')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=150)
print("Feature importance chart saved!")

print("\nTop 5 Fraud Drivers:")
for i, row in feat_imp.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n=== BUSINESS RECOMMENDATIONS ===")
print("1. Check transactions within 24 hours of signup - these show higher fraud risk")
print("2. Monitor high-value purchases during unusual hours for extra verification")
print("3. Track users with many transactions in short time periods")
print("4. Consider additional verification for purchases from new devices")
print("5. Flag transactions where user age is unusual for the purchase type")
print("\nDone!")
