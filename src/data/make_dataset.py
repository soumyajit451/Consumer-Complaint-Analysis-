import sys
import time
import logging
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import create_log_path, CustomLogger

log_file_path = create_log_path('make_dataset')
# Create the custom logger object
dataset_logger = CustomLogger(logger_name='make_dataset', log_filename=log_file_path)
# Set the level of logging to INFO
dataset_logger.set_log_level(level=logging.INFO)

def load_preprocessed_data(input_path: Path) -> pd.DataFrame:
    preprocessed_data = pd.read_csv(input_path)
    # Randomly sample 50000 rows
    sampled_preprocessed_data = preprocessed_data.sample(n=100000)
    rows, columns = sampled_preprocessed_data.shape
    dataset_logger.save_logs(msg=f'{input_path.stem} data read having {rows} rows and {columns} columns', log_level='info')
    return sampled_preprocessed_data

def train_test_split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    dataset_logger.save_logs(msg=f'Data is split into train split with shape {train_data.shape} and test split with shape {test_data.shape}', log_level='info')
    dataset_logger.save_logs(msg=f'The parameter values are test_size={test_size} and random_state={random_state}', log_level='info')
    return train_data, test_data

def save_data(data: pd.DataFrame, output_path: Path):
    data.to_csv(output_path, index=False)
    dataset_logger.save_logs(msg=f'{output_path.stem + output_path.suffix} data saved successfully to the output folder', log_level='info')

def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
    except FileNotFoundError as e:
        dataset_logger.save_logs(msg='Parameters file not found, switching to default values for train test split', log_level='error')
        default_dict = {'test_size': 0.2, 'random_state': 30}
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']
        return test_size, random_state
    else:
        dataset_logger.save_logs(msg='Parameters file read successfully', log_level='info')
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']
        return test_size, random_state

def complaint():
    time.sleep(10)
    # Read the input file name from command line
    input_file_name = sys.argv[1]
    # Current file path
    current_path = Path(__file__).resolve()
    # Root directory path
    root_path = current_path.parent.parent.parent
    # Interim data directory path
    interim_data_path = root_path / 'data' / 'interim'
    # Make directory for the interim path
    interim_data_path.mkdir(exist_ok=True)
    # Raw train file path
    preprocessed_data_path = root_path / 'data' / 'processed' / input_file_name
    # Load the training file
    preprocessed_df = load_preprocessed_data(input_path=preprocessed_data_path)
    # Parameters from params file
    test_size, random_state = read_params('params.yaml')
    # Split the file to train and test data
    train_df, test_df = train_test_split_data(data=preprocessed_df, test_size=test_size, random_state=random_state)
    # Save the train data to the output path
    save_data(data=train_df, output_path=interim_data_path / 'train.csv')
    # Save the test data to the output path
    save_data(data=test_df, output_path=interim_data_path / 'test.csv')

if __name__ == '__main__':
    complaint()
    print("Divide dataset into train and test splits")

