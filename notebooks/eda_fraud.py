import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load e-commerce fraud data
df = pd.read_csv('data/raw/Fraud_Data.csv')
print("E-commerce data:", df.shape)
print(df.head())
print("\nFraud count:")
print(df['class'].value_counts())
print(f"Fraud rate: {df['class'].mean()*100:.2f}%")

# Load credit card data
cc = pd.read_csv('data/raw/creditcard.csv')
print("\nCredit card data:", cc.shape)
print("\nFraud count:")
print(cc['Class'].value_counts())
print(f"Fraud rate: {cc['Class'].mean()*100:.2f}%")

# Missing values
print("\nMissing values in e-commerce:")
print(df.isnull().sum())
print("\nMissing values in credit card:")
print(cc.isnull().sum())

# Chart
df['class'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('E-commerce: Fraud vs Non-Fraud')
plt.savefig('reports/fraud_distribution.png', dpi=100)
print("Chart saved!")
