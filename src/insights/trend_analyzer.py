import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class TrendAnalyzer:
    """Detects and analyzes trends in time series data."""
    
    def __init__(self, df, date_col='date'):
        """
        Initialize trend analyzer with dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing time series data
        date_col : str
            Name of date column
        """
        self.original_df = df.copy()
        self.date_col = date_col
        
        # Ensure date column is datetime type
        if date_col in self.original_df.columns:
            self.original_df[date_col] = pd.to_datetime(self.original_df[date_col])
            # Sort by date
            self.original_df = self.original_df.sort_values(date_col)
        
        # Initialize empty results
        self.trends = {}
        self.seasonality = {}
        self.change_points = {}
    
    def detect_trend(self, column, window=7):
        """
        Detect trend in a specific column using rolling averages and linear regression.
        
        Parameters:
        -----------
        column : str
            Column to analyze for trend
        window : int
            Window size for rolling average
            
        Returns:
        --------
        dict
            Dictionary containing trend information
        """
        if column not in self.original_df.columns:
            return {"error": f"Column {column} not found in dataframe"}
        
        df = self.original_df.copy()
        
        # Calculate rolling average
        df[f'{column}_rolling'] = df[column].rolling(window=window, min_periods=1).mean()
        
        # Calculate linear trend
        x = np.arange(len(df))
        y = df[column].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate start and end values from the trendline
        trend_start = intercept
        trend_end = intercept + slope * (len(df) - 1)
        
        # Calculate overall percentage change
        overall_change_pct = ((df[column].iloc[-1] - df[column].iloc[0]) / df[column].iloc[0]) * 100 if df[column].iloc[0] != 0 else np.inf
        
        # Calculate recent trend direction (last quarter of data)
        recent_start_idx = max(0, int(len(df) * 0.75))
        recent_x = np.arange(recent_start_idx, len(df))
        if len(recent_x) > 1:
            recent_y = df[column].iloc[recent_start_idx:].values
            recent_slope, _, _, recent_p_value, _ = stats.linregress(recent_x, recent_y)
            recent_significant = recent_p_value < 0.05
        else:
            recent_slope = 0
            recent_significant = False
        
        # Determine trend strength based on R-squared value
        trend_strength = abs(r_value)
        
        # Interpret trend
        if abs(r_value) < 0.3:
            trend_interpretation = "No clear trend"
        elif r_value > 0 and p_value < 0.05:
            trend_interpretation = "Significant upward trend"
        elif r_value < 0 and p_value < 0.05:
            trend_interpretation = "Significant downward trend"
        elif r_value > 0:
            trend_interpretation = "Slight upward trend"
        else:
            trend_interpretation = "Slight downward trend"
        
        # Different trend language for different metrics
        if "efficiency" in column.lower() or "per_customer" in column.lower():
            trending_term = "improving" if slope > 0 else "declining"
        elif "cost" in column.lower() or "spend" in column.lower():
            trending_term = "increasing" if slope > 0 else "decreasing"
        else:
            trending_term = "growing" if slope > 0 else "declining"
        
        # Different recent trend language
        if recent_slope > 0 and recent_significant:
            recent_trend = f"accelerating {trending_term}"
        elif recent_slope < 0 and slope > 0 and recent_significant:
            recent_trend = "growth is slowing"
        elif recent_slope > 0 and slope < 0 and recent_significant:
            recent_trend = "decline is slowing"
        elif recent_slope < 0 and recent_significant:
            recent_trend = f"accelerating {trending_term}"
        else:
            recent_trend = f"steady {trending_term}"
        
        trend_info = {
            "column": column,
            "slope": slope,
            "p_value": p_value,
            "r_squared": r_value**2,
            "trend_strength": trend_strength,
            "interpretation": trend_interpretation,
            "trending_term": trending_term,
            "recent_trend": recent_trend,
            "overall_change_pct": overall_change_pct,
            "trend_start": trend_start,
            "trend_end": trend_end,
            "significant": p_value < 0.05,
            "direction": "up" if slope > 0 else "down"
        }
        
        self.trends[column] = trend_info
        return trend_info
    
    def detect_seasonality(self, column, period=7):
        """
        Detect seasonality in time series data.
        
        Parameters:
        -----------
        column : str
            Column to analyze for seasonality
        period : int
            Expected periodicity (7 for weekly, 12 for monthly, etc.)
            
        Returns:
        --------
        dict
            Dictionary containing seasonality information
        """
        if column not in self.original_df.columns:
            return {"error": f"Column {column} not found in dataframe"}
        
        df = self.original_df.copy()
        
        # Check if we have enough data points
        if len(df) < period * 2:
            self.seasonality[column] = {
                "has_seasonality": False,
                "message": "Not enough data points to detect seasonality"
            }
            return self.seasonality[column]
        
        try:
            # Perform seasonal decomposition
            # Create a regular time series with the date as index
            ts_df = df.set_index(self.date_col)[[column]].asfreq('D', method='ffill')
            
            # Decompose the series
            result = seasonal_decompose(ts_df, model='additive', period=period)
            
            # Extract seasonality strength
            # Compute variance of seasonality component and original series
            detrended = result.observed - result.trend
            strength = 1 - np.var(result.resid) / np.var(detrended)
            
            seasonality_info = {
                "column": column,
                "has_seasonality": strength > 0.3,
                "seasonality_strength": strength,
                "period": period
            }
            
            # Find peak days if there's notable seasonality
            if strength > 0.3:
                # Get day of week if period is 7
                if period == 7 and self.date_col in df.columns:
                    df['day_of_week'] = df[self.date_col].dt.dayofweek
                    day_means = df.groupby('day_of_week')[column].mean()
                    peak_day_idx = day_means.idxmax()
                    low_day_idx = day_means.idxmin()
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    seasonality_info["peak_day"] = days[peak_day_idx]
                    seasonality_info["low_day"] = days[low_day_idx]
                    seasonality_info["day_variance_pct"] = ((day_means.max() - day_means.min()) / day_means.mean()) * 100
                    
                    # Detect weekend effect
                    weekday_mean = df[df['day_of_week'] < 5][column].mean()
                    weekend_mean = df[df['day_of_week'] >= 5][column].mean()
                    weekend_effect = (weekend_mean - weekday_mean) / weekday_mean * 100
                    seasonality_info["weekend_effect"] = weekend_effect
                    seasonality_info["weekend_effect_significant"] = abs(weekend_effect) > 10
            
            self.seasonality[column] = seasonality_info
            return seasonality_info
            
        except Exception as e:
            self.seasonality[column] = {
                "has_seasonality": False,
                "error": str(e)
            }
            return self.seasonality[column]
    
    def detect_change_points(self, column, window=10):
        """
        Detect significant changes or breakpoints in the time series.
        
        Parameters:
        -----------
        column : str
            Column to analyze for change points
        window : int
            Window size for change detection
            
        Returns:
        --------
        dict
            Dictionary containing change point information
        """
        if column not in self.original_df.columns:
            return {"error": f"Column {column} not found in dataframe"}
        
        df = self.original_df.copy()
        
        # Calculate rolling mean and standard deviation
        df[f'{column}_roll_mean'] = df[column].rolling(window=window).mean()
        df[f'{column}_roll_std'] = df[column].rolling(window=window).std()
        
        # Drop rows with NaN (first window-1 rows)
        df = df.dropna(subset=[f'{column}_roll_mean', f'{column}_roll_std'])
        
        if len(df) < 2:
            self.change_points[column] = {
                "has_change_points": False,
                "message": "Not enough data points after creating rolling statistics"
            }
            return self.change_points[column]
        
        # Calculate z-scores for each point
        df[f'{column}_zscore'] = abs((df[column] - df[f'{column}_roll_mean']) / df[f'{column}_roll_std'])
        
        # Identify significant change points (z-score > 2)
        significant_changes = df[df[f'{column}_zscore'] > 2].copy()
        
        # Get dates of significant changes
        if len(significant_changes) > 0 and self.date_col in significant_changes.columns:
            change_dates = significant_changes[self.date_col].tolist()
            # Calculate the percentage change for each point
            significant_changes['pct_change'] = significant_changes[column].pct_change() * 100
            
            # Get most significant change
            most_sig_idx = significant_changes[f'{column}_zscore'].idxmax()
            most_sig_date = significant_changes.loc[most_sig_idx, self.date_col]
            most_sig_value = significant_changes.loc[most_sig_idx, column]
            most_sig_zscore = significant_changes.loc[most_sig_idx, f'{column}_zscore']
            
            # Find the value before the change
            most_sig_date_idx = df[df[self.date_col] == most_sig_date].index[0]
            if most_sig_date_idx > 0:
                value_before = df.iloc[most_sig_date_idx - 1][column]
                pct_change = ((most_sig_value - value_before) / value_before) * 100 if value_before != 0 else float('inf')
            else:
                value_before = None
                pct_change = None
            
            change_info = {
                "column": column,
                "has_change_points": True,
                "num_change_points": len(significant_changes),
                "change_dates": change_dates,
                "most_significant_date": most_sig_date,
                "most_significant_zscore": most_sig_zscore,
                "most_significant_value": most_sig_value,
                "value_before_change": value_before,
                "most_significant_pct_change": pct_change,
                "direction": "up" if pct_change > 0 else "down" if pct_change < 0 else "unchanged"
            }
        else:
            change_info = {
                "column": column,
                "has_change_points": False,
                "num_change_points": 0
            }
        
        self.change_points[column] = change_info
        return change_info
    
    def analyze_metric(self, column, window=7, seasonality_period=7):
        """
        Perform complete analysis of a metric including trend, seasonality, and change points.
        
        Parameters:
        -----------
        column : str
            Column to analyze
        window : int
            Window size for rolling calculations
        seasonality_period : int
            Period for seasonality detection
            
        Returns:
        --------
        dict
            Dictionary containing complete analysis
        """
        if column not in self.original_df.columns:
            return {"error": f"Column {column} not found in dataframe"}
        
        # Detect trends
        trend_info = self.detect_trend(column, window)
        
        # Detect seasonality
        seasonality_info = self.detect_seasonality(column, seasonality_period)
        
        # Detect change points
        change_info = self.detect_change_points(column, window)
        
        # Combine all analyses
        analysis = {
            "column": column,
            "trend": trend_info,
            "seasonality": seasonality_info,
            "change_points": change_info
        }
        
        return analysis

def analyze_business_metrics(df, metrics=None, date_col='date'):
    """
    Analyze key business metrics for trends, seasonality, and change points.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing business metrics
    metrics : list of str, optional
        List of column names to analyze. If None, will select common business metrics
    date_col : str
        Name of date column
        
    Returns:
    --------
    dict
        Dictionary containing complete analysis for all metrics
    """
    # If metrics is None, try to identify common business metrics
    if metrics is None:
        metrics = []
        common_metrics = ['sales', 'revenue', 'marketing_spend', 'customer_count', 
                         'marketing_efficiency', 'sales_per_customer', 'profit']
        for metric in common_metrics:
            if metric in df.columns:
                metrics.append(metric)
    
    analyzer = TrendAnalyzer(df, date_col=date_col)
    
    results = {}
    for metric in metrics:
        results[metric] = analyzer.analyze_metric(metric)
    
    return results

if __name__ == "__main__":
    # Example of how to use the trend analyzer
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
        
        # Analyze business metrics
        metrics = ['sales', 'marketing_spend', 'customer_count', 'marketing_efficiency']
        results = analyze_business_metrics(df, metrics)
        
        # Print sample results
        for metric, analysis in results.items():
            print(f"\nAnalysis for {metric}:")
            print(f"Trend: {analysis['trend']['interpretation']}")
            print(f"R² value: {analysis['trend']['r_squared']:.3f}")
            
            if analysis['seasonality']['has_seasonality']:
                print(f"Has seasonality with strength: {analysis['seasonality']['seasonality_strength']:.3f}")
                if 'peak_day' in analysis['seasonality']:
                    print(f"Peak day: {analysis['seasonality']['peak_day']}")
            else:
                print("No significant seasonality detected")
            
            if analysis['change_points']['has_change_points']:
                print(f"Found {analysis['change_points']['num_change_points']} significant change points")
                print(f"Most significant change: {analysis['change_points']['most_significant_pct_change']:.2f}% on {analysis['change_points']['most_significant_date']}")
            else:
                print("No significant change points detected")
    
    except Exception as e:
        print(f"Error running trend analyzer: {str(e)}")
