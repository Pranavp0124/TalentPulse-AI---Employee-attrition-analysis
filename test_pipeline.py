"""
test_pipeline.py
================
Automated Test Suite for TalentPulse AI.
Verifies data loading, model unpickling, single inference, batch scoring, and metric contracts.
Can be run via `python test_pipeline.py` or through `pytest`.
"""

import os
import unittest
import pandas as pd
import numpy as np
from attrition_model import load_predictor, ALL_FEATURES, FEATURES_NUM, FEATURES_CAT


class TestTalentPulsePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = "HR_Employee_Attrition.csv"
        cls.sample_path = "sample_test_cohort.csv"
        cls.predictor = load_predictor("model.pkl")

    def test_01_dataset_exists_and_valid(self):
        self.assertTrue(os.path.exists(self.data_path), f"Missing {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.assertGreaterEqual(len(df), 1000, "Dataset should have at least 1,000 records")
        self.assertIn("Attrition", df.columns, "Target column 'Attrition' not found")
        for col in ALL_FEATURES:
            self.assertIn(col, df.columns, f"Required feature '{col}' missing from dataset")

    def test_02_model_metrics(self):
        metrics = self.predictor.metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("precision", metrics)
        self.assertGreater(metrics["accuracy"], 0.70, "Model accuracy should exceed 70%")
        self.assertGreater(metrics["roc_auc"], 0.70, "Model ROC-AUC should exceed 0.70")

    def test_03_single_employee_inference(self):
        sample_emp = {
            'Age': 32,
            'MonthlyIncome': 4500,
            'TotalWorkingYears': 6,
            'YearsAtCompany': 3,
            'YearsWithCurrManager': 2,
            'DistanceFromHome': 12,
            'StockOptionLevel': 0,
            'EnvironmentSatisfaction': 2,
            'JobSatisfaction': 2,
            'WorkLifeBalance': 2,
            'JobInvolvement': 3,
            'NumCompaniesWorked': 2,
            'PercentSalaryHike': 14,
            'Department': 'Sales',
            'JobRole': 'Sales Executive',
            'MaritalStatus': 'Single',
            'OverTime': 'Yes',
            'BusinessTravel': 'Travel_Frequently'
        }
        df_single = pd.DataFrame([sample_emp])
        proba = self.predictor.predict_proba(df_single)
        
        self.assertEqual(proba.shape, (1, 2), "Inference should return 2-class probability array")
        risk_score = proba[0][1]
        self.assertTrue(0.0 <= risk_score <= 1.0, "Risk score must be a probability between 0 and 1")
        
        pred = self.predictor.predict(df_single)
        self.assertIn(pred[0], [0, 1], "Prediction class should be binary (0 or 1)")

    def test_04_batch_inference(self):
        if not os.path.exists(self.sample_path):
            from generate_dummy_data import generate_cohort
            df_gen = generate_cohort(30)
            df_gen.to_csv(self.sample_path, index=False)
            
        df_batch = pd.read_csv(self.sample_path)
        probas = self.predictor.predict_proba(df_batch)
        self.assertEqual(len(probas), len(df_batch))
        preds = self.predictor.predict(df_batch)
        self.assertEqual(len(preds), len(df_batch))
        self.assertTrue(np.all((probas[:, 1] >= 0.0) & (probas[:, 1] <= 1.0)))

    def test_05_feature_importances_contract(self):
        importances = self.predictor.feature_importances
        self.assertIsInstance(importances, dict)
        self.assertGreater(len(importances), 5)
        # Sum of importances should be close to 100%
        total = sum(importances.values())
        self.assertAlmostEqual(total, 100.0, delta=1.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
