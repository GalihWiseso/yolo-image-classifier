"""
Simple ML Model Training Script
This script trains a basic scikit-learn model for classification
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def create_sample_data():
    """Create sample classification dataset"""
    print("Creating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names

def train_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def save_model_locally(model, feature_names):
    """Save model locally for testing"""
    print("Saving model locally...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/simple_classifier.pkl"
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_path = "models/feature_names.pkl"
    joblib.dump(feature_names, feature_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Feature names saved to: {feature_path}")

def main():
    """Main training pipeline"""
    print("Starting simple ML model training...")
    
    # Create sample data
    df, feature_names = create_sample_data()
    
    # Prepare features and target
    X = df[feature_names]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train model
    model, train_accuracy = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    test_accuracy = evaluate_model(model, X_test, y_test)
    
    # Save locally
    save_model_locally(model, feature_names)
    
    print(f"\nTraining completed successfully!")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Model saved to models/ directory")

if __name__ == "__main__":
    main()

