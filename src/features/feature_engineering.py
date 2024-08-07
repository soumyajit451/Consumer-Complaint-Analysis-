import joblib
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from src.logger import create_log_path, CustomLogger

# Path to save the log files
log_file_path = create_log_path('feature_engineering')
# Create the custom logger object
feature_engineering_logger = CustomLogger(logger_name='feature_engineering', log_filename=log_file_path)
# Set the level of logging to INFO
feature_engineering_logger.set_log_level(level=logging.INFO)

def load_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    feature_engineering_logger.save_logs(msg=f'Data loaded from {file_path}', log_level='info')
    return data

def save_sparse_data(data, file_path: Path):
    save_npz(file_path, data)
    feature_engineering_logger.save_logs(msg=f'Sparse data saved to {file_path}', log_level='info')

def save_dense_data(data, file_path: Path):
    joblib.dump(data, file_path)
    feature_engineering_logger.save_logs(msg=f'Dense data saved to {file_path}', log_level='info')

def feature_engineering(preprocessed_data: pd.DataFrame, preprocessor=None):
    X = preprocessed_data.drop('Consumer Disputed?', axis=1)
    y = preprocessed_data['Consumer Disputed?']

    # Assign Columns for one hot encoding and for vectorization
    ohe_columns = ['Product', 'Sub-Product', 'Issue', 'Sub-Issue', 'Company Public Response', 'State', 'Tags', 'Consumer Consent Provided?', 'Submitted Via', 'Timely Response?', 'Company Response To Consumer']
    string_columns = ['Consumer Complaint Narrative', 'Company']

    if preprocessor is None:
        # Define the transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_columns),
                ('tfidf', TfidfVectorizer(), 'Consumer Complaint Narrative'),
                ('company_tfidf', TfidfVectorizer(), 'Company')
            ], n_jobs=-1)
        # Fit and transform the data
        X_transformed = preprocessor.fit_transform(X)
    else:
        # Transform the data using the pre-loaded preprocessor
        X_transformed = preprocessor.transform(X)

    feature_engineering_logger.save_logs('Data transformed using ColumnTransformer', 'info')

    # Encoding Target Columns
    lb = LabelBinarizer()
    y_transformed = lb.fit_transform(y)
    feature_engineering_logger.save_logs('Target column encoded', 'info')

    return X_transformed, y_transformed, preprocessor, lb

def complaint():
    # Current file path
    current_path = Path(__file__).resolve()
    root_path = current_path.parent.parent.parent
    data_path = root_path / 'data' / 'interim'
    feature_path = root_path / 'data' / 'feature'

    # Load datasets
    train_data = load_data(data_path / 'train.csv')
    test_data = load_data(data_path / 'test.csv')

    # Apply feature engineering
    X_train, y_train, preprocessor, lb = feature_engineering(train_data)
    X_test, y_test, _, _ = feature_engineering(test_data, preprocessor)

    # Save processed datasets
    feature_path.mkdir(parents=True, exist_ok=True)
    save_sparse_data(X_train, feature_path / 'train_features.npz')
    save_dense_data(y_train, feature_path / 'train_target.joblib')
    save_sparse_data(X_test, feature_path / 'test_features.npz')
    save_dense_data(y_test, feature_path / 'test_target.joblib')

    # Save the transformers
    joblib.dump(preprocessor, feature_path / 'preprocessor.joblib')
    joblib.dump(lb, feature_path / 'label_binarizer.joblib')

    feature_engineering_logger.save_logs('Feature engineering complete and datasets saved', 'info')
    print("Feature engineering complete and datasets saved.")

if __name__ == '__main__':
    complaint()
