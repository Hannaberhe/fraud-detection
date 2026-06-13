import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

cc = pd.read_csv('data/raw/creditcard.csv')
features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
            'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
            'V21','V22','V23','V24','V25','V26','V27','V28','Amount']
X = cc[features]
y = cc['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Credit Card Data")
print("Fraud rate:", round(y.mean()*100, 2), "%")
print("Train:", len(X_train), "Test:", len(X_test))

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("\nLogistic Regression:")
print("F1:", round(f1_score(y_test, lr_pred), 4))
print("AUC:", round(roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]), 4))

rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\nRandom Forest:")
print("F1:", round(f1_score(y_test, rf_pred), 4))
print("AUC:", round(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]), 4))
print(confusion_matrix(y_test, rf_pred))

joblib.dump(rf, 'models/creditcard_rf_model.pkl')
print("\nModel saved")
