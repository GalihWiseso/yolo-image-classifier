"""
Linear Regression Inference Utilities
Loads a trained LinearRegression model and runs predictions
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
import os


MODEL_PATH = "models/linear_regression.pkl"
FEATURES_PATH = "models/feature_names.pkl"


def load_model():
	if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)):
		raise FileNotFoundError("Model artifacts not found. Run train_model.py first.")
	model = joblib.load(MODEL_PATH)
	feature_names: List[str] = joblib.load(FEATURES_PATH)
	return model, feature_names


def make_dataframe(records: List[dict], feature_names: List[str]) -> pd.DataFrame:
	df = pd.DataFrame(records)
	# Fill missing features with 0.0 and order columns
	for f in feature_names:
		if f not in df.columns:
			df[f] = 0.0
	df = df[feature_names]
	return df


def predict(model, df: pd.DataFrame) -> np.ndarray:
	return model.predict(df)


if __name__ == "__main__":
	# Quick local smoke test using an online image path or local file
	# Example using PIL load from a local file:
	# img = Image.open("example.jpg").convert("RGB")
	# model = load_model()
	# labels, scores = predict_topk(model, img)
	# print(list(zip(labels, scores)))
	pass

