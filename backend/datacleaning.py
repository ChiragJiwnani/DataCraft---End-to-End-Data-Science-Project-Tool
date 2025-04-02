#datacleaning.py
import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

# Initialize logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to clean the data
def clean_data(data_path, target_column=None):
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Dataset loaded successfully from {data_path}")
        print(df.head())
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return None, str(e)
    
    # Preserve the target column if it exists
    if target_column and target_column in df.columns:
        target_col_copy = df[target_column].copy()  # Preserve the target column
        df = df.drop(columns=[target_column])  # Drop the target column for cleaning
    else:
        target_col_copy = None
    
    # 1. Remove duplicate or irrelevant observations
    initial_row_count = df.shape[0]
    df.drop_duplicates(inplace=True)
    # target_column = target_column_copy[df.index]  # Align the target column with the dataset after removing duplicates
    duplicate_removed_row_count = df.shape[0]
    logging.info(f"Removed duplicates: {initial_row_count - duplicate_removed_row_count} rows")

    # 2. Fix structural errors (e.g., casing in categorical columns)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].str.strip().str.lower()
    logging.info(f"Fixed structural errors in categorical columns: {list(categorical_cols)}")

    # 3. Convert numeric columns, handle errors by coercing to NaN
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
    logging.info(f"Converted columns to numeric where applicable: {list(numeric_cols)}")

    # 4. Remove rows with missing values
    initial_row_count = df.shape[0]
    df.dropna(inplace=True)
    # target_column = target_column[df.index]  # Align the target column with the dataset after removing rows
    cleaned_row_count = df.shape[0]
    logging.info(f"Removed rows with missing values: {initial_row_count - cleaned_row_count} rows removed")

    # 5. Filter unwanted outliers (using 3 standard deviations)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        initial_row_count = df.shape[0]
        df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]
        # target_column = target_column[df.index]  # Align the target column after filtering outliers
        filtered_row_count = df.shape[0]
        logging.info(f"Filtered outliers in column {col}: {initial_row_count - filtered_row_count} rows removed")

    if target_col_copy is not None:
        target_col_copy = target_col_copy[df.index]
    
    # 6. Normalize numeric columns and return scaled data
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[numeric_cols])  # Scale only numeric columns
        print("Type of scaled_data", type(scaled_data))
        print("scaled_data: \n", scaled_data)
        
        # Convert scaled data back to a DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
        print("hello")
        # Re-add the categorical columns
        for col in categorical_cols:
            scaled_df[col] = df[col].values
        print("hello1")  

        # Re-add the target column
        if target_col_copy is not None:
            scaled_df[target_column] = target_col_copy.values
        print("hello2")  
        # print(scaled_df[target_column])
        logging.info(f"Normalized the following numeric columns: {list(numeric_cols)}")
        
    except Exception as e:
        logging.error(f"Error during normalization: {str(e)}")
        return None, str(e)

    # Save cleaned dataset
    cleaned_filepath = os.path.join(os.path.dirname(data_path), 'cleaned_data.csv')
    try:
        scaled_df.to_csv(cleaned_filepath, index=False)
        logging.info(f"Cleaned dataset saved successfully at {cleaned_filepath}")
        return cleaned_filepath, scaled_df  # Return cleaned file path and scaled data
    except Exception as e:
        logging.error(f"Error saving cleaned dataset: {str(e)}")
        return None, str(e)
