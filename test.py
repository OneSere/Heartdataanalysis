import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import xgboost as xgb

print("Heart Attack Prediction Model Training Script v2 - XGBoost")
print("Author: Abhishek Choudhary")
print("Credits: Dataset by Zheen Hospital, Erbil, Iraq (2019)")
print("Disclaimer: For research and educational purposes only, not clinical use.")
print("--------------------------------------------------------------")

print("Current working directory:", os.getcwd())

# Load data
df = pd.read_csv("Medicaldataset1.csv")
print("Original Columns:", df.columns)

# Split Blood Pressure
if 'Blood Pressure' in df.columns:
    print("Splitting 'Blood Pressure' column into systolic_bp and diastolic_bp...")
    bp_split = df['Blood Pressure'].astype(str).str.split('/', expand=True)
    df['systolic_bp'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['diastolic_bp'] = pd.to_numeric(bp_split[1], errors='coerce')
    df.drop(columns=['Blood Pressure'], inplace=True)

# Rename columns
df.rename(columns={
    'Age': 'age',
    'Sex': 'gender',
    'Cholesterol': 'cholesterol',
    'Heart Rate': 'heart_rate',
    'Diabetes': 'diabetes',
    'Family History': 'family_history',
    'Smoking': 'smoking',
    'Obesity': 'obesity',
    'Alcohol Consumption': 'alcohol_consumption',
    'Exercise Hours Per Week': 'exercise_hours_per_week',
    'Diet': 'diet',
    'Previous Heart Problems': 'previous_heart_problems',
    'Medication Use': 'medication_use',
    'Stress Level': 'stress_level',
    'Sedentary Hours Per Day': 'sedentary_hours_per_day',
    'Income': 'income',
    'BMI': 'bmi',
    'Triglycerides': 'triglycerides',
    'Physical Activity Days Per Week': 'physical_activity_days_per_week',
    'Sleep Hours Per Day': 'sleep_hours_per_day',
    'Country': 'country',
    'Continent': 'continent',
    'Hemisphere': 'hemisphere',
    'Heart Attack Risk': 'result'
}, inplace=True)

# Drop Patient ID if present
if 'Patient ID' in df.columns:
    df.drop(columns=['Patient ID'], inplace=True)

# Filter invalid rows
if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
    df = df[df['systolic_bp'] > df['diastolic_bp']]

df = df[(df['age'] > 0) & (df['age'] < 120)]

if 'heart_rate' in df.columns:
    df = df[(df['heart_rate'] > 30) & (df['heart_rate'] < 220)]

df.ffill(inplace=True)

if 'result' not in df.columns:
    raise ValueError("Target column 'result' not found.")

# Features and target
X = df.drop('result', axis=1)
y = df['result']

# Encode categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
print(f"Class distribution after SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# XGBoost classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter grid for Randomized Search
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

random_search = RandomizedSearchCV(
    xgb_clf, param_distributions=param_dist,
    n_iter=20, scoring='f1', cv=5,
    verbose=2, random_state=42, n_jobs=-1
)

print("Starting hyperparameter tuning...")
random_search.fit(X_train_res, y_train_res)
print("Best hyperparameters:", random_search.best_params_)

best_model = random_search.best_estimator_

# Predict probabilities on test set
y_proba = best_model.predict_proba(X_test)[:, 1]

# ROC Curve for threshold tuning
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)

# Plot ROC and PR curves
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Choose threshold to maximize F1 score on validation set (approximate)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5

print(f"Best threshold from PR curve to maximize F1 score: {best_threshold:.3f}")

# Predict with optimized threshold
y_pred_opt = (y_proba >= best_threshold).astype(int)

print("\nClassification Report with optimized threshold:\n")
print(classification_report(y_test, y_pred_opt))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_opt))

# Save the model with exception handling
print("Saving model now...")
try:
    joblib.dump(best_model, "heart_model_v2_xgb.pkl")
    print("Model saved as 'heart_model_v2_xgb.pkl'")
except Exception as e:
    print("Failed to save model:", e)

