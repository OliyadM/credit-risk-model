# tests/test_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.train import X, y  # Assuming X and y are defined globally or imported correctly

def test_prediction_shape():
    """Test if the prediction shape matches the test set size."""
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0], "Prediction length does not match test set length"

def test_roc_auc_range():
    """Test if ROC-AUC is between 0 and 1."""
    from sklearn.metrics import roc_auc_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    assert 0 <= roc_auc <= 1, "ROC-AUC should be between 0 and 1"

if __name__ == "__main__":
    test_prediction_shape()
    test_roc_auc_range()
    print("All tests passed!")