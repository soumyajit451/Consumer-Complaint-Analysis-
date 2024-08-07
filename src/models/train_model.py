import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
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

# Build neural network model
def build_neural_network(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data(train_features_path, train_target_path):
    # Load the .npz file as a sparse matrix
    X_train = load_npz(train_features_path)

    # Load the target values
    y_train = joblib.load(train_target_path)
    
    return X_train, y_train

def evaluate_models(X_train, y_train, models, batch_size=512, epochs=5):
    models_list = []
    accuracy_list = []
    auc = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        model_train_accuracy, model_train_f1, model_train_precision, model_train_recall, model_train_rocauc_score = evaluate_clf(y_train, y_train_pred)

        models_list.append(name)
        accuracy_list.append(model_train_accuracy)
        auc.append(model_train_rocauc_score)

        print(name)
        print('Model performance for Training set')
        print(f"- Accuracy: {model_train_accuracy:.4f}")
        print(f'- F1 score: {model_train_f1:.4f}')
        print(f'- Precision: {model_train_precision:.4f}')
        print(f'- Recall: {model_train_recall:.4f}')
        print(f'- Roc Auc Score: {model_train_rocauc_score:.4f}')
        print('=' * 35)
        print('\n')

    neural_net_model = build_neural_network(input_shape=(X_train.shape[1],))
    history = neural_net_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    y_train_pred_nn = (neural_net_model.predict(X_train) > 0.5).astype("int32")
    model_train_accuracy, model_train_f1, model_train_precision, model_train_recall, model_train_rocauc_score = evaluate_clf(y_train, y_train_pred_nn)

    models_list.append("Neural Network")
    accuracy_list.append(model_train_accuracy)
    auc.append(model_train_rocauc_score)

    print("Neural Network")
    print('Model performance for Training set')
    print(f'- Accuracy: {model_train_accuracy:.4f}')
    print(f'- F1 score: {model_train_f1:.4f}')
    print(f'- Precision: {model_train_precision:.4f}')
    print(f'- Recall: {model_train_recall:.4f}')
    print(f'- Roc Auc Score: {model_train_rocauc_score:.4f}')
    print('=' * 35)
    print('\n')

    report = pd.DataFrame(list(zip(models_list, accuracy_list, auc)), columns=['Model Name', 'Accuracy', 'ROC AUC']).sort_values(by=['Accuracy'], ascending=False)
    return report

def save_model(model, path):
    with open(path, 'wb') as f:
        joblib.dump(model, f)

def save_report(report, path):
    report.to_csv(path, index=False)

def main():
    # Current file path
    current_path = Path(__file__).resolve()
    # Root directory path
    root_path = current_path.parent.parent.parent
    # Read the train feature and target paths from command line arguments
    train_features_path = root_path / 'data' / 'feature' / sys.argv[1]
    train_target_path = root_path / 'data' / 'feature' / sys.argv[2]

    # Load the data
    try:
        X_train, y_train = load_data(train_features_path, train_target_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBClassifier": XGBClassifier(),
        "Support Vector Classifier": SVC(),
        "AdaBoost Classifier": AdaBoostClassifier()
    }

    # Evaluate models
    try:
        base_model_report = evaluate_models(X_train, y_train, models)
    except Exception as e:
        print(f"Error evaluating models: {e}")
        return
    
    # Determine the best model
    best_model_name = base_model_report.iloc[0]['Model Name']
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    # Save the best model
    model_save_path = root_path / 'models' / 'model.joblib'
    save_model(best_model, model_save_path)

    # Save the evaluation report
    report_save_path = root_path / 'models' / 'base_model_report.csv'
    save_report(base_model_report, report_save_path)

    print("Training complete and model saved.")

if __name__ == '__main__':
    main()
