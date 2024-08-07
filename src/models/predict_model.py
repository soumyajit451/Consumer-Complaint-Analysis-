import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import sys
from pathlib import Path
from scipy.sparse import load_npz

# Define evaluation function
def evaluate_clf(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, f1, precision, recall, roc_auc

def load_data(test_features_path, test_target_path):
    # Load the .npz file as a sparse matrix
    X_test = load_npz(test_features_path)

    # Load the target values
    y_test = joblib.load(test_target_path)
    
    return X_test, y_test

def save_report(report, path):
    report.to_csv(path, index=False)

def main():
    # Current file path
    current_path = Path(__file__).resolve()
    # Root directory path
    root_path = current_path.parent.parent.parent
    # Read the test feature and target paths from command line arguments
    test_features_path = root_path / sys.argv[1]
    test_target_path = root_path / sys.argv[2]

    # Load the data
    try:
        X_test, y_test = load_data(test_features_path, test_target_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load the model
    model_path = root_path / 'models' / 'model.joblib'
    model = joblib.load(model_path)

    # Check the shape of the test data to ensure it matches the model's expected input
    if X_test.shape[1] != model.n_features_in_:
        print(f"Error: X_test has {X_test.shape[1]} features, but the model is expecting {model.n_features_in_} features as input.")
        return

    # Make predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return

    # Evaluate predictions
    try:
        accuracy, f1, precision, recall, roc_auc = evaluate_clf(y_test, y_pred)
    except Exception as e:
        print(f"Error evaluating predictions: {e}")
        return

    # Save predictions
    predictions = pd.DataFrame({
        'y_true': y_test.flatten(),
        'y_pred': y_pred.flatten()
    })
    predictions_path = root_path / 'predictions' / 'predictions.csv'
    predictions.to_csv(predictions_path, index=False)

    # Save evaluation report
    report = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC'],
        'Score': [accuracy, f1, precision, recall, roc_auc]
    })
    report_path = root_path / 'predictions' / 'evaluation_report.csv'
    save_report(report, report_path)

    print("Prediction complete and results saved.")

if __name__ == '__main__':
    main()
