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
dataset_logger = CustomLogger(logger_name='make_dataset',
                              log_filename=log_file_path)
# Set the level of logging to INFO
dataset_logger.set_log_level(level=logging.INFO)

def load_raw_data(input_path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows, columns = raw_data.shape
    dataset_logger.save_logs(msg=f'{input_path.stem} data read having {rows} rows and {columns} columns',
                             log_level='info')
    return raw_data

def train_val_test_split(data: pd.DataFrame,
                         val_size: float,
                         test_size: float,
                         random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, temp_data = train_test_split(data, 
                                             test_size=(val_size + test_size),
                                             random_state=random_state)
    val_data, test_data = train_test_split(temp_data, 
                                           test_size=test_size/(val_size + test_size),
                                           random_state=random_state)
    dataset_logger.save_logs(msg=f'Data is split into train split with shape {train_data.shape}, val split with shape {val_data.shape}, and test split with shape {test_data.shape}',
                             log_level='info')
    dataset_logger.save_logs(msg=f'The parameter values are val_size={val_size}, test_size={test_size}, and random_state={random_state}',
                             log_level='info')
    return train_data, val_data, test_data

def save_data(data: pd.DataFrame, output_path: Path):
    data.to_csv(output_path, index=False)
    dataset_logger.save_logs(msg=f'{output_path.stem + output_path.suffix} data saved successfully to the output folder',
                             log_level='info')

def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
    except FileNotFoundError as e:
        dataset_logger.save_logs(msg='Parameters file not found, switching to default values for train test split',
                                 log_level='error')
        default_dict = {'val_size': 0.2, 'test_size': 0.2, 'random_state': 30}
        val_size = default_dict['val_size']
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']
        return val_size, test_size, random_state
    else:
        dataset_logger.save_logs(msg='Parameters file read successfully',
                                 log_level='info')
        val_size = params_file['make_dataset']['val_size']
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']
        return val_size, test_size, random_state

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
    raw_df_path = root_path / 'data' / 'extracted_data' / input_file_name
    # Load the training file
    raw_df = load_raw_data(input_path=raw_df_path)
    # Parameters from params file
    val_size, test_size, random_state = read_params('params.yaml')
    # Split the file to train, validation, and test data
    train_df, val_df, test_df = train_val_test_split(data=raw_df,
                                                     val_size=val_size,
                                                     test_size=test_size,
                                                     random_state=random_state)
    # Remove the target column from the test data
    target_column = 'Consumer disputed?'
    if target_column in test_df.columns:
        test_df = test_df.drop(columns=[target_column])
    # Save the train data to the output path
    save_data(data=train_df, output_path=interim_data_path / 'train.csv')
    # Save the validation data to the output path
    save_data(data=val_df, output_path=interim_data_path / 'val.csv')
    # Save the test data to the output path
    save_data(data=test_df, output_path=interim_data_path / 'test.csv')

if __name__ == '__main__':
    complaint()
