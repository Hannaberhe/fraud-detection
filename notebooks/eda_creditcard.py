import pandas as pd
import matplotlib.pyplot as plt

cc = pd.read_csv('data/raw/creditcard.csv')
print("Credit Card Data:", cc.shape)
print("\nFraud count:")
print(cc['Class'].value_counts())
print(f"Fraud rate: {cc['Class'].mean()*100:.2f}%")

cc['Amount'].hist(bins=50, color='steelblue')
plt.title('Credit Card Transaction Amounts')
plt.xlabel('Amount')
plt.savefig('reports/creditcard_amounts.png', dpi=100)
print("Chart saved")
