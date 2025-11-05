"""
Simple Linear Regression Training Script
Trains a scikit-learn LinearRegression model on synthetic data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


def create_sample_data(n_samples: int = 1000, n_features: int = 10, noise: float = 10.0):
	X, y = make_regression(
		n_samples=n_samples,
		n_features=n_features,
		n_informative=n_features,
		noise=noise,
		random_state=42
	)
	feature_names = [f"feature_{i}" for i in range(n_features)]
	df = pd.DataFrame(X, columns=feature_names)
	df["target"] = y
	return df, feature_names


def train_and_evaluate(X_train, y_train, X_test, y_test):
	model = LinearRegression()
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	mse = mean_squared_error(y_test, preds)
	r2 = r2_score(y_test, preds)
	return model, mse, r2


def save_artifacts(model, feature_names):
	os.makedirs("models", exist_ok=True)
	joblib.dump(model, "models/linear_regression.pkl")
	joblib.dump(feature_names, "models/feature_names.pkl")


def main():
	print("Starting linear regression training...")
	df, feature_names = create_sample_data()

	X = df[feature_names]
	y = df["target"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	model, mse, r2 = train_and_evaluate(X_train, y_train, X_test, y_test)
	save_artifacts(model, feature_names)

	print(f"Samples: {len(df)} | Features: {len(feature_names)}")
	print(f"Test MSE: {mse:.4f} | R2: {r2:.4f}")
	print("Artifacts saved to models/: linear_regression.pkl, feature_names.pkl")


if __name__ == "__main__":
	main()

