import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import dump

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\Apple_Quality_project\apple_quality.csv")

# Handle null values
df['Size'].fillna(df['Size'].mean(), inplace=True)
df['Weight'].fillna(df['Weight'].mean(), inplace=True)
df['Sweetness'].fillna(df['Sweetness'].mean(), inplace=True)
df['Crunchiness'].fillna(df['Crunchiness'].mean(), inplace=True)
df['Juiciness'].fillna(df['Juiciness'].mean(), inplace=True)
df['Ripeness'].fillna(df['Ripeness'].mean(), inplace=True)

# Convert Acidity and fill NaN
df['Acidity'] = pd.to_numeric(df['Acidity'], errors='coerce')
df['Acidity'].fillna(df['Acidity'].mean(), inplace=True)

# Fill target null
df['Quality'].fillna(df['Quality'].mode()[0], inplace=True)

# Encoding
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop(['Quality_good', 'A_id'], axis=1)
y = df['Quality_good']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model_xg = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xg.fit(X_train_scaled, y_train)

# Save model & scaler (IMPORTANT)
dump(model_xg, r"C:\Users\HP\Downloads\Apple_Quality_project\apple_quality_model.joblib")
dump(scaler, r"C:\Users\HP\Downloads\Apple_Quality_project\scaler.joblib")

print("✅ Model and Scaler saved successfully")
