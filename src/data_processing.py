import pandas as pd
import numpy as np

# File paths
DATA_PATH = "data/raw/data.csv"
PROCESSED_PATH = "data/processed/data_cleaned.csv"

def load_data(path=DATA_PATH):
    """Load the raw data"""
    return pd.read_csv(path)

def process_data(input_path=DATA_PATH, output_path=PROCESSED_PATH):
    """Process the raw data by cleaning and parsing necessary columns"""
    df = load_data(input_path)

    # Parsing 'TransactionStartTime' into datetime format
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)  # Remove timezone if present

    # Handle missing values (optional based on data cleaning)
    df.fillna(0, inplace=True)

    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"Data cleaned and saved to {output_path}")

if __name__ == "__main__":
    process_data()
