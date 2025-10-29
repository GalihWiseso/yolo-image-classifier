"""
Test script to validate the trained model locally
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

def test_model():
    """Test the trained model with sample data"""
    
    # Check if model exists
    model_path = "models/simple_classifier.pkl"
    if not os.path.exists(model_path):
        print("Model not found. Please run train_model.py first.")
        return
    
    # Load model and feature names
    print("Loading model...")
    model = joblib.load(model_path)
    feature_names = joblib.load("models/feature_names.pkl")
    
    print(f"Model loaded successfully!")
    print(f"Features: {feature_names}")
    
    # Create test data
    print("\nCreating test data...")
    np.random.seed(42)
    test_data = pd.DataFrame(
        np.random.randn(100, len(feature_names)),
        columns=feature_names
    )
    
    # Add some pattern to make predictions meaningful
    test_data['feature_0'] = test_data['feature_0'] + np.random.choice([-2, 2], 100)
    
    # Create true labels (simplified)
    true_labels = (test_data['feature_0'] > 0).astype(int)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of test samples: {len(test_data)}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  Features: {test_data.iloc[i].values}")
        print(f"  True label: {true_labels[i]}")
        print(f"  Prediction: {predictions[i]}")
        print(f"  Probability: {probabilities[i]}")
        print()
    
    # Test with single prediction (like API call)
    print("Testing single prediction (API-like)...")
    single_sample = test_data.iloc[0:1]
    single_pred = model.predict(single_sample)
    single_prob = model.predict_proba(single_sample)
    
    print(f"Single prediction result:")
    print(f"  Input shape: {single_sample.shape}")
    print(f"  Prediction: {single_pred[0]}")
    print(f"  Probability: {single_prob[0]}")
    
    print("\nModel test completed successfully!")

if __name__ == "__main__":
    test_model()

