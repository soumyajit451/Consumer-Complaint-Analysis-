import pandas as pd
import sys
import logging
from pathlib import Path
from src.logger import create_log_path, CustomLogger

log_file_path = create_log_path('data_preprocessing')
# Create the custom logger object
preprocessing_logger = CustomLogger(logger_name='data_preprocessing',
                                    log_filename=log_file_path)
# Set the level of logging to INFO
preprocessing_logger.set_log_level(level=logging.INFO)

def preprocess_data(consumer_complaint, logger):
    logger.save_logs('Starting preprocessing', 'info')

    # Removing Duplicates
    consumer_complaint = consumer_complaint.drop_duplicates(inplace=False)
    logger.save_logs('Duplicates removed', 'info')

    # make columns name as title
    consumer_complaint.columns = consumer_complaint.columns.str.title()
    # Remove NaN values in specific columns
    consumer_complaint = consumer_complaint.dropna(subset=['Consumer Disputed?'])
    consumer_complaint = consumer_complaint.dropna(subset=['Company Response To Consumer'])
    consumer_complaint = consumer_complaint.dropna(subset=['Issue'])
    logger.save_logs('NaN values in specific columns removed', 'info')

    # Fill NaN values in other columns
    consumer_complaint['Sub-Product'] = consumer_complaint['Sub-Product'].fillna("Sub Product not listed")
    consumer_complaint['Sub-Issue'] = consumer_complaint['Sub-Issue'].fillna("Sub issue not listed")
    consumer_complaint['Consumer Complaint Narrative'] = consumer_complaint['Consumer Complaint Narrative'].fillna("Complaint not listed")
    consumer_complaint['Company Public Response'] = consumer_complaint['Company Public Response'].fillna("Company Public Response not listed")
    consumer_complaint['State'] = consumer_complaint['State'].fillna("State not listed")
    consumer_complaint['Tags'] = consumer_complaint['Tags'].fillna("Normal citizen")
    consumer_complaint['Consumer Consent Provided?'] = consumer_complaint['Consumer Consent Provided?'].fillna("Data not listed")

    logger.save_logs('NaN values filled', 'info')

    # Drop date columns
    consumer_complaint = consumer_complaint.drop(['Date Received', 'Date Sent To Company'], axis=1)
    logger.save_logs('Date columns dropped', 'info')

    # Drop columns that are not useful for classification
    consumer_complaint = consumer_complaint.drop(['Complaint Id', 'Zip Code'], axis=1)
    logger.save_logs('Unnecessary columns dropped', 'info')

    return consumer_complaint

def save_data(df, output_path, logger):
    df.to_csv(output_path, index=False)
    logger.save_logs(f'Data saved to {output_path}', 'info')

def complaint():
    # Current file path
    current_path = Path(__file__).resolve()
     # Root directory path
    root_path = current_path.parent.parent.parent
    # Read the  preprocessing_path name from command line
    preprocessing_data_read = sys.argv[1]

    preprocessing_logger.save_logs('Paths read from command line arguments', 'info')

    # specify the path of input data
    preprocessing_path = root_path / 'data' / 'extracted_data' / preprocessing_data_read
    preprocessing_logger.save_logs('Data loaded', 'info')

    # Load the data
    extracted_data = pd.read_csv(preprocessing_path)
    preprocessing_logger.save_logs('Data loaded', 'info')
    # Preprocess the data independently
    preprocessed_data_cleaned = preprocess_data(extracted_data, preprocessing_logger)

    # preprocessed data directory path
    preprocessed_data_path = root_path / 'data' / 'processed'
    # Make directory for the preprocessed_data if not exist
    preprocessed_data_path.mkdir(exist_ok=True)
    # Save the preprocessed data
    save_data(df=preprocessed_data_cleaned, output_path=preprocessed_data_path / 'preprocessed.csv', logger=preprocessing_logger)

    preprocessing_logger.save_logs('Data preprocessing complete and files saved', 'info')
    print("Data preprocessing complete and files saved.")

if __name__ == '__main__':
    complaint()


