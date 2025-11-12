#!/usr/bin/env python3
"""
train_model.py

Trains a RandomForestClassifier model on the preprocessed workload dataset
to predict whether a workload should be deployed on a traditional
infrastructure or serverless platform.

Input:
  - CSV file from preprocess.py (with cost ratio < 5 rule applied)
Output:
  - orchestrator_model.pkl  (saved trained model)
  - feature_importance.png  (feature contribution visualization)
  - printed accuracy, precision, recall, and F1 score

Usage:
  python train_model.py --input ../data/workload_dataset_sample.csv --output orchestrator_model.pkl
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import os

# --------------------------
# Training Function
# --------------------------
def train_model(input_path, model_output):
    print(f" Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    # Clean and prepare dataset
    df = df.dropna(subset=["target platform"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Define features and target
    features = [
        "cpu cores",
        "memory mb",
        "latency sensitive",
        "execution time",
        "data size mb",
        "cost ratio"
    ]
    target = "target platform"

    # Encode target labels
    df[target] = df[target].map({"serverless": 0, "traditional": 1})

    X = df[features]
    y = df[target]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train RandomForest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        n_jobs=-1
    )

    print("\n Training RandomForest model...")
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Serverless", "Traditional"]))

    # Save model
    joblib.dump(model, model_output)
    print(f"\n Model saved to: {model_output}")

    # Plot feature importance
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    importances.plot.barh(color="steelblue")
    plt.title("Feature Importance (Hybrid Cloud Classifier)")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    feature_plot_path = os.path.splitext(model_output)[0] + "_feature_importance.png"
    plt.savefig(feature_plot_path)
    print(f" Feature importance plot saved to: {feature_plot_path}")

    # Print top features
    print("\n Top Contributing Features:")
    for feature, score in importances.sort_values(ascending=False).items():
        print(f" - {feature}: {score:.4f}")

    return model


# --------------------------
# Main Function
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Train RandomForest model for workload classification.")
    parser.add_argument("--input", "-i", required=True, help="Path to preprocessed dataset CSV")
    parser.add_argument("--output", "-o", default="orchestrator_model.pkl", help="Output model filename")
    args = parser.parse_args()

    model = train_model(args.input, args.output)


if __name__ == "__main__":
    main()
