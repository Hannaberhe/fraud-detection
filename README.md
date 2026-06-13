# Fraud Detection for E-commerce and Bank Transactions

Adey Innovations Inc. - Fraud Detection System

## Project Overview
Build machine learning models to detect fraudulent transactions across e-commerce and bank credit card data.

## Datasets
- Fraud_Data.csv - E-commerce transactions with user and device info
- IpAddress_to_Country.csv - IP address to country mapping
- creditcard.csv - Anonymized credit card transactions

## Results
- Random Forest: F1=0.68, AUC=0.76
- Top fraud driver: time_since_signup (97.6% importance)

## Setup
pip install -r requirements.txt

## Structure
- notebooks/ - EDA, feature engineering, modeling, explainability
- models/ - Trained model artifacts
- reports/ - Visualizations
- tests/ - Unit tests
