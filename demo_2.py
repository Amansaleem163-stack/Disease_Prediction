import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic Medical Dataset
np.random.seed(42)
n_patients = 1000

medical_data = pd.DataFrame({
    'age': np.random.randint(20, 80, n_patients),
    'blood_pressure': np.random.normal(120, 15, n_patients),
    'cholesterol': np.random.normal(200, 30, n_patients),
    'glucose_level': np.random.normal(100, 25, n_patients),
    'symptom_severity': np.random.randint(1, 10, n_patients)
})

# Target: Disease status (1 = Present, 0 = Absent)
medical_data['disease_present'] = np.where(
    (medical_data['age'] > 50) & (medical_data['glucose_level'] > 110) | (medical_data['symptom_severity'] > 7), 1, 0
)

# 2. Split Data
X_med = medical_data.drop(columns=['disease_present'])
y_med = medical_data['disease_present']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_med, y_med, test_size=0.2, random_state=42)

scaler_m = StandardScaler()
X_train_m_scaled = scaler_m.fit_transform(X_train_m)
X_test_m_scaled = scaler_m.transform(X_test_m)

# 3. Model Training: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_m_scaled, y_train_m)
lr_preds = lr_model.predict(X_test_m_scaled)

# 4. Model Training: XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_m_scaled, y_train_m)
xgb_preds = xgb_model.predict(X_test_m_scaled)

# 5. Output Results
print("--- Task 4: Disease Prediction Evaluation ---")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test_m, lr_preds):.4f}")
print(f"XGBoost Accuracy: {accuracy_score(y_test_m, xgb_preds):.4f}\n")
print("XGBoost Detailed Report:")
print(classification_report(y_test_m, xgb_preds))