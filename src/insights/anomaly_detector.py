import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class AnomalyDetector:
    """Detects anomalies in business metrics using various techniques."""
    
    def __init__(self, df):
        """
        Initialize anomaly detector with dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing business metrics
        """
        self.df = df.copy()
        self.anomalies = {}
    
    def detect_statistical_anomalies(self, column, method='zscore', threshold=3):
        """
        Detect anomalies using statistical methods.
        
        Parameters:
        -----------
        column : str
            Column to analyze for anomalies
        method : str
            Method to use: 'zscore' or 'iqr'
        threshold : float
            Threshold for anomaly detection
            
        Returns:
        --------
        dict
            Dictionary containing anomaly information
        """
        if column not in self.df.columns:
            return {"error": f"Column {column} not found in dataframe"}
        
        df = self.df.copy()
        
        if method == 'zscore':
            # Calculate z-scores
            mean = df[column].mean()
            std = df[column].std()
            df['zscore'] = abs((df[column] - mean) / std)
            
            # Identify anomalies
            anomalies = df[df['zscore'] > threshold].copy()
            anomalies['deviation'] = (anomalies[column] - mean) / mean * 100
            
            # Prepare result
            result = {
                "column": column,
                "method": "zscore",
                "threshold": threshold,
                "num_anomalies": len(anomalies),
                "anomalies_pct": len(anomalies) / len(df) * 100,
                "anomaly_rows": anomalies,
                "high_anomalies": anomalies[anomalies[column] > mean].shape[0],
                "low_anomalies": anomalies[anomalies[column] < mean].shape[0]
            }
            
            # If date column exists, include dates of anomalies
            if 'date' in df.columns:
                result["anomaly_dates"] = anomalies['date'].tolist()
            
        elif method == 'iqr':
            # Calculate IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify anomalies
            anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
            anomalies['deviation'] = (anomalies[column] - df[column].median()) / df[column].median() * 100
            
            # Prepare result
            result = {
                "column": column,
                "method": "iqr",
                "threshold": threshold,
                "num_anomalies": len(anomalies),
                "anomalies_pct": len(anomalies) / len(df) * 100,
                "anomaly_rows": anomalies,
                "high_anomalies": anomalies[anomalies[column] > upper_bound].shape[0],
                "low_anomalies": anomalies[anomalies[column] < lower_bound].shape[0],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            
            # If date column exists, include dates of anomalies
            if 'date' in df.columns:
                result["anomaly_dates"] = anomalies['date'].tolist()
        else:
            return {"error": f"Method {method} not supported. Use 'zscore' or 'iqr'."}
        
        self.anomalies[f"{column}_{method}"] = result
        return result
    
    def detect_isolation_forest_anomalies(self, columns, contamination=0.05):
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Parameters:
        -----------
        columns : list of str
            Columns to use for anomaly detection
        contamination : float
            Expected proportion of anomalies
            
        Returns:
        --------
        dict
            Dictionary containing anomaly information
        """
        # Check if all columns exist
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            return {"error": f"Columns {missing_cols} not found in dataframe"}
        
        df = self.df.copy()
        
        # Prepare the data
        X = df[columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Initialize and fit the Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly'] = model.fit_predict(X)
        
        # In Isolation Forest, -1 means anomaly, 1 means normal
        anomalies = df[df['anomaly'] == -1].copy()
        
        # Prepare result
        result = {
            "columns": columns,
            "method": "isolation_forest",
            "contamination": contamination,
            "num_anomalies": len(anomalies),
            "anomalies_pct": len(anomalies) / len(df) * 100,
            "anomaly_rows": anomalies
        }
        
        # If date column exists, include dates of anomalies
        if 'date' in df.columns:
            result["anomaly_dates"] = anomalies['date'].tolist()
        
        key = f"isolation_forest_{'_'.join(columns)}"
        self.anomalies[key] = result
        return result
    
    def detect_time_series_anomalies(self, column, date_col='date', window=7, threshold=2.5):
        """
        Detect anomalies in time series data using rolling statistics.
        
        Parameters:
        -----------
        column : str
            Column to analyze for anomalies
        date_col : str
            Date column name
        window : int
            Window size for rolling statistics
        threshold : float
            Z-score threshold for anomaly detection
            
        Returns:
        --------
        dict
            Dictionary containing anomaly information
        """
        if column not in self.df.columns:
            return {"error": f"Column {column} not found in dataframe"}
        
        if date_col not in self.df.columns:
            return {"error": f"Date column {date_col} not found in dataframe"}
        
        df = self.df.copy()
        
        # Ensure the date column is datetime and sort
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Calculate rolling mean and std
        df[f'{column}_roll_mean'] = df[column].rolling(window=window, min_periods=1).mean()
        df[f'{column}_roll_std'] = df[column].rolling(window=window, min_periods=1).std()
        
        # Fill NaN in std (first value) with a small number to avoid division by zero
        df[f'{column}_roll_std'] = df[f'{column}_roll_std'].fillna(df[column].std() * 0.1)
        
        # Calculate z-scores
        df['zscore'] = abs((df[column] - df[f'{column}_roll_mean']) / df[f'{column}_roll_std'])
        
        # Identify anomalies
        anomalies = df[df['zscore'] > threshold].copy()
        
        # Calculate deviation percentage from rolling mean
        anomalies['deviation'] = (anomalies[column] - anomalies[f'{column}_roll_mean']) / anomalies[f'{column}_roll_mean'] * 100
        
        # Prepare result
        result = {
            "column": column,
            "method": "time_series",
            "window": window,
            "threshold": threshold,
            "num_anomalies": len(anomalies),
            "anomalies_pct": len(anomalies) / len(df) * 100,
            "anomaly_rows": anomalies,
            "high_anomalies": anomalies[anomalies[column] > anomalies[f'{column}_roll_mean']].shape[0],
            "low_anomalies": anomalies[anomalies[column] < anomalies[f'{column}_roll_mean']].shape[0]
        }
        
        # Include dates of anomalies
        result["anomaly_dates"] = anomalies[date_col].tolist()
        
        key = f"{column}_time_series"
        self.anomalies[key] = result
        return result
    
    def find_correlation_anomalies(self, col1, col2, threshold=2.5):
        """
        Detect anomalies in the relationship between two metrics.
        
        Parameters:
        -----------
        col1 : str
            First column name
        col2 : str
            Second column name
        threshold : float
            Threshold for determining anomalies
            
        Returns:
        --------
        dict
            Dictionary containing correlation anomaly information
        """
        if col1 not in self.df.columns or col2 not in self.df.columns:
            missing = []
            if col1 not in self.df.columns:
                missing.append(col1)
            if col2 not in self.df.columns:
                missing.append(col2)
            return {"error": f"Columns {missing} not found in dataframe"}
        
        df = self.df.copy()
        
        # Calculate correlation
        correlation = df[col1].corr(df[col2])
        
        # Perform linear regression
        x = df[col1]
        y = df[col2]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate expected y values based on regression
        df['expected'] = slope * df[col1] + intercept
        
        # Calculate residuals and standardize them
        df['residuals'] = df[col2] - df['expected']
        residual_std = df['residuals'].std()
        df['std_residuals'] = df['residuals'] / residual_std
        
        # Identify correlation anomalies (points that deviate significantly from the trend)
        anomalies = df[abs(df['std_residuals']) > threshold].copy()
        
        # Calculate deviation percentage
        anomalies['deviation'] = (anomalies[col2] - anomalies['expected']) / anomalies['expected'] * 100
        
        # Prepare result
        result = {
            "columns": [col1, col2],
            "method": "correlation_anomaly",
            "correlation": correlation,
            "r_squared": r_value**2,
            "slope": slope,
            "num_anomalies": len(anomalies),
            "anomalies_pct": len(anomalies) / len(df) * 100,
            "anomaly_rows": anomalies,
            "positive_residuals": anomalies[anomalies['residuals'] > 0].shape[0],
            "negative_residuals": anomalies[anomalies['residuals'] < 0].shape[0]
        }
        
        # If date column exists, include dates of anomalies
        if 'date' in df.columns:
            result["anomaly_dates"] = anomalies['date'].tolist()
        
        key = f"correlation_{col1}_{col2}"
        self.anomalies[key] = result
        return result

def analyze_anomalies(df, metrics=None, date_col='date'):
    """
    Detect anomalies in business metrics using multiple methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing business data
    metrics : list of str, optional
        List of metrics to analyze. If None, common business metrics will be used.
    date_col : str
        Name of date column
        
    Returns:
    --------
    dict
        Dictionary containing anomaly analysis results
    """
    # If metrics is None, try to identify common business metrics
    if metrics is None:
        metrics = []
        common_metrics = ['sales', 'revenue', 'marketing_spend', 'customer_count', 
                         'marketing_efficiency', 'sales_per_customer', 'profit']
        for metric in common_metrics:
            if metric in df.columns:
                metrics.append(metric)
    
    detector = AnomalyDetector(df)
    results = {}
    
    # Analyze each metric individually
    for metric in metrics:
        # Statistical anomalies
        results[f"{metric}_statistical"] = detector.detect_statistical_anomalies(metric, method='zscore')
        
        # Time series anomalies
        if date_col in df.columns:
            results[f"{metric}_time_series"] = detector.detect_time_series_anomalies(metric, date_col=date_col)
    
    # Analyze relationships between metrics
    if len(metrics) >= 2:
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                col1, col2 = metrics[i], metrics[j]
                results[f"correlation_{col1}_{col2}"] = detector.find_correlation_anomalies(col1, col2)
    
    # Multi-dimensional anomalies using Isolation Forest
    if len(metrics) >= 2:
        results["multidimensional"] = detector.detect_isolation_forest_anomalies(metrics)
    
    return results

if __name__ == "__main__":
    # Example of how to use the anomaly detector
    import os
    
    def get_project_root():
        script_path = os.path.abspath(__file__)
        insights_dir = os.path.dirname(script_path)
        src_dir = os.path.dirname(insights_dir)
        project_root = os.path.dirname(src_dir)
        return project_root
    
    # Load data
    root = get_project_root()
    try:
        data_path = os.path.join(root, "data", "processed", "processed_business_data.csv")
        df = pd.read_csv(data_path)
        
        # Analyze anomalies in business metrics
        metrics = ['sales', 'marketing_spend', 'customer_count']
        results = analyze_anomalies(df, metrics)
        
        # Print sample results
        for key, result in results.items():
            if 'error' not in result:
                print(f"\nAnalysis for {key}:")
                print(f"Method: {result['method']}")
                print(f"Found {result['num_anomalies']} anomalies ({result['anomalies_pct']:.2f}% of data)")
                
                if 'anomaly_dates' in result and result['anomaly_dates']:
                    print(f"Anomaly dates: {result['anomaly_dates'][:3]}...")
                    
                if 'correlation' in result:
                    print(f"Correlation: {result['correlation']:.3f}, R²: {result['r_squared']:.3f}")
    
    except Exception as e:
        print(f"Error running anomaly detector: {str(e)}")
