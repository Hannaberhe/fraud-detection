import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/Fraud_Data.csv')
df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])

df['hour_of_day'] = df['purchase_time'].dt.hour
df['day_of_week'] = df['purchase_time'].dt.dayofweek
df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
df['user_txn_count'] = df.groupby('user_id')['purchase_time'].transform('count')

import ipaddress
def ip_to_int(ip):
    try:
        return int(ipaddress.IPv4Address(ip))
    except:
        return 0

df['ip_int'] = df['ip_address'].apply(ip_to_int)

ip_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')
ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('int64')
ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('int64')

ip_df = ip_df.sort_values('lower_bound_ip_address')
df = df.sort_values('ip_int')

df = pd.merge_asof(df, ip_df, left_on='ip_int', right_on='lower_bound_ip_address', direction='backward')
df['country'] = df.apply(lambda x: x['country'] if x['ip_int'] <= x['upper_bound_ip_address'] else 'Unknown', axis=1)

print("Features created")
print("Countries found:", df['country'].nunique())
print(df['country'].value_counts().head(10))

df.to_csv('data/processed/fraud_with_features.csv', index=False)
print("Saved")
