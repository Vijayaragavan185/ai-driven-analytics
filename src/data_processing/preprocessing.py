# src/data_processing/preprocessing.py
import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Clean the dataframe by handling missing values and outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    df_clean = df_clean.dropna()
    
    # Handle outliers using IQR method for numerical columns
    for column in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                           (df_clean[column] <= upper_bound)]
    
    return df_clean

def feature_engineering(df):
    """
    Create new features from existing data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new features
    """
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Convert date to datetime if it's not already
    if 'date' in df_features.columns:
        df_features['date'] = pd.to_datetime(df_features['date'])
        
        # Extract date components
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
        df_features['month'] = df_features['date'].dt.month
        df_features['year'] = df_features['date'].dt.year
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate efficiency metrics
    if all(col in df_features.columns for col in ['sales', 'marketing_spend']):
        df_features['marketing_efficiency'] = df_features['sales'] / df_features['marketing_spend']
    
    if all(col in df_features.columns for col in ['sales', 'customer_count']):
        df_features['sales_per_customer'] = df_features['sales'] / df_features['customer_count']
    
    return df_features

def process_data(input_path, output_path):
    """
    Complete data processing pipeline.
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to save processed CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Processed dataframe
    """
    # Load data
    df = load_data(input_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Feature engineering
    df_processed = feature_engineering(df_clean)
    
    # Save processed data
    df_processed.to_csv(output_path, index=False)
    
    return df_processed

if __name__ == "__main__":
    # Example usage
    process_data("../data/raw/sample_business_data.csv", 
                "../data/processed/processed_business_data.csv")
