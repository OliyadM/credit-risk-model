# src/train.py
# This script handles model training, hyperparameter tuning, and evaluation for Task 5 without MLflow.
# Assumptions:
# - Processed data is available at '../data/processed/processed_data.csv' from data_processing.py.
# - Models to train: Logistic Regression and Random Forest.
# - Hyperparameter tuning uses GridSearchCV for simplicity.
# - Evaluation metrics include Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")

# Prepare features and target
X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
y = df['is_high_risk']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Logistic Regression
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'max_iter': [100, 200]
}
lr = LogisticRegression(random_state=42)
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)

# Best model
best_lr = grid_search_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)
y_pred_proba_lr = best_lr.predict_proba(X_test)[:, 1]

# Evaluate
metrics_lr = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
}
print("Logistic Regression - Best Params:", grid_search_lr.best_params_)
print("Logistic Regression - Metrics:", metrics_lr)

# Model 2: Random Forest
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best model
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]

# Evaluate
metrics_rf = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
}
print("Random Forest - Best Params:", grid_search_rf.best_params_)
print("Random Forest - Metrics:", metrics_rf)

# Compare and select best model (based on ROC-AUC)
if metrics_lr['roc_auc'] > metrics_rf['roc_auc']:
    print("Best model: Logistic Regression")
else:
    print("Best model: Random Forest")

print("Training completed.")