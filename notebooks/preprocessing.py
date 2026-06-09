import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/processed/fraud_with_features.csv')

# Features for modeling
num_features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 'user_txn_count', 'ip_int']
cat_features = ['source', 'browser', 'sex']

X_num = df[num_features].fillna(0)
X_cat = df[cat_features].fillna('Unknown')
y = df['class']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

X_processed = preprocessor.fit_transform(pd.concat([X_num, X_cat], axis=1))
print("Before SMOTE - Class distribution:")
print(y.value_counts())

# SMOTE on training set only
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE - Class distribution:")
print(pd.Series(y_train_smote).value_counts())
print("Preprocessing complete")
