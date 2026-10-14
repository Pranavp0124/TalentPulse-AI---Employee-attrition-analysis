"""
attrition_model.py
==================
Enterprise Employee Attrition Model Pipeline.
Trains a high-performing Soft-Voting Ensemble (Logistic Regression + Random Forest + Gradient Boosting)
with ColumnTransformer preprocessing (Scaling + One-Hot Encoding) on the IBM HR Analytics Dataset.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Core curated features representing demographics, job role, compensation, and work environment
FEATURES_NUM = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
    'YearsWithCurrManager', 'DistanceFromHome', 'StockOptionLevel',
    'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
    'JobInvolvement', 'NumCompaniesWorked', 'PercentSalaryHike'
]

FEATURES_CAT = [
    'Department', 'JobRole', 'MaritalStatus', 'OverTime', 'BusinessTravel'
]

ALL_FEATURES = FEATURES_NUM + FEATURES_CAT
TARGET = 'Attrition'


class AttritionPredictor:
    """Wrapper class providing inference, probability thresholding, and input feature alignment."""
    def __init__(self, model_dict):
        self.pipeline = model_dict['pipeline']
        self.features_num = model_dict['features_num']
        self.features_cat = model_dict['features_cat']
        self.all_features = model_dict['all_features']
        self.metrics = model_dict['metrics']
        self.feature_importances = model_dict['feature_importances']
        self.categories = model_dict['categories']
        self.threshold = model_dict.get('threshold', 0.40)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_clean = self._align_features(X)
        else:
            X_clean = pd.DataFrame(X, columns=self.all_features)
        return self.pipeline.predict_proba(X_clean)

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def _align_features(self, df):
        aligned = df.copy()
        for col in self.features_num:
            if col not in aligned.columns:
                aligned[col] = 0
            aligned[col] = pd.to_numeric(aligned[col], errors='coerce').fillna(0)
        for col in self.features_cat:
            if col not in aligned.columns:
                aligned[col] = self.categories.get(col, ['Unknown'])[0]
            aligned[col] = aligned[col].astype(str)
        return aligned[self.all_features]


def train_and_save_model(data_path="HR_Employee_Attrition.csv", model_path="model.pkl"):
    print(f"Loading dataset from: {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found.")

    df = pd.read_csv(data_path)
    
    # Clean target
    if df[TARGET].dtype == 'object':
        y = (df[TARGET].str.strip().str.lower() == 'yes').astype(int)
    else:
        y = df[TARGET].astype(int)

    X = df[ALL_FEATURES]

    # Gather categorical options for validation & UI forms
    categories = {col: sorted(df[col].dropna().unique().tolist()) for col in FEATURES_CAT}

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), FEATURES_NUM),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), FEATURES_CAT)
        ]
    )

    # Classifiers in soft-voting ensemble
    clf_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=250, max_depth=8, class_weight='balanced', random_state=42)
    clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('lr', clf_lr), ('rf', clf_rf), ('gb', clf_gb)],
        voting='soft'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])

    print("Fitting model ensemble...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    threshold = 0.40
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Model Evaluation ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(cm))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Compute Feature Importances from RF estimator
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(FEATURES_CAT).tolist()
    all_encoded_names = FEATURES_NUM + cat_feature_names

    rf_estimator = pipeline.named_steps['classifier'].named_estimators_['rf']
    raw_importances = rf_estimator.feature_importances_

    # Aggregate one-hot back to feature groups for intuitive display
    aggregated_imp = {feat: 0.0 for feat in ALL_FEATURES}
    for name, imp in zip(all_encoded_names, raw_importances):
        matched = False
        for cat in FEATURES_CAT:
            if name.startswith(cat + "_"):
                aggregated_imp[cat] += float(imp)
                matched = True
                break
        if not matched and name in aggregated_imp:
            aggregated_imp[name] += float(imp)

    # Normalize to sum 100%
    total_imp = sum(aggregated_imp.values())
    aggregated_imp = {k: round((v / total_imp) * 100, 2) for k, v in aggregated_imp.items()}
    sorted_imp = dict(sorted(aggregated_imp.items(), key=lambda item: item[1], reverse=True))

    metrics = {
        'accuracy': round(acc, 4),
        'roc_auc': round(roc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm,
        'test_size': len(y_test),
        'train_size': len(y_train)
    }

    # Store as a plain dictionary to ensure zero pickle namespace issues across modules
    model_payload = {
        'pipeline': pipeline,
        'features_num': FEATURES_NUM,
        'features_cat': FEATURES_CAT,
        'all_features': ALL_FEATURES,
        'metrics': metrics,
        'feature_importances': sorted_imp,
        'categories': categories,
        'threshold': threshold
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)

    print(f"\nModel and metadata successfully serialized to: {model_path}")
    return AttritionPredictor(model_payload)


def load_predictor(model_path="model.pkl"):
    if not os.path.exists(model_path):
        return train_and_save_model(model_path=model_path)
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        return AttritionPredictor(payload)
    return payload


if __name__ == "__main__":
    train_and_save_model()
