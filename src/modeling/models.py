# src/modeling/models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_features_and_target(df, target_col, feature_cols=None, test_size=0.2, random_state=42):
    """
    Prepare features and target variables, split into train/test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    feature_cols : list of str, optional
        List of feature columns to use. If None, use all columns except target
    test_size : float, optional
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_columns)
    """
    # If no feature columns provided, use all except target
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Select only numeric features (avoid date, categorical without encoding)
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare X and y
    X = df[numeric_features]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, numeric_features

def train_models(X_train, y_train):
    """
    Train multiple regression models.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
        
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained {name}")
        
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate trained models on training and test data.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Test features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    pandas.DataFrame
        Evaluation metrics for each model
    """
    results = []
    
    for name, model in models.items():
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Store results
        results.append({
            'Model': name,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R²': train_r2,
            'Test R²': test_r2
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def plot_model_comparison(evaluation_df, metric='Test R²', ascending=False):
    """
    Plot model comparison based on a selected metric.
    
    Parameters:
    -----------
    evaluation_df : pandas.DataFrame
        DataFrame with model evaluation metrics
    metric : str, optional
        Metric to use for comparison
    ascending : bool, optional
        Whether to sort in ascending order
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by the specified metric
    sorted_df = evaluation_df.sort_values(metric, ascending=ascending)
    
    # Create bar chart
    sns.barplot(x='Model', y=metric, data=sorted_df)
    plt.title(f'Model Comparison by {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_best_model(models, evaluation_df, metric='Test R²', ascending=False, save_path=None):
    """
    Save the best model based on a selected metric.
    """
    # Import required modules
    import joblib
    import os
    
    # Find best model name based on metric
    if ascending:
        best_model_name = evaluation_df.loc[evaluation_df[metric].idxmin(), 'Model']
    else:
        best_model_name = evaluation_df.loc[evaluation_df[metric].idxmax(), 'Model']
    
    # Get the best model
    best_model = models[best_model_name]
    
    # Save the model if path provided
    if save_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        joblib.dump(best_model, save_path)
        print(f"Best model ({best_model_name}) saved to {save_path}")
    
    return best_model_name, best_model


def predict_with_model(model, X):
    """
    Make predictions using a trained model.
    
    Parameters:
    -----------
    model : trained model
        Model to use for prediction
    X : pandas.DataFrame
        Features to predict on
        
    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    return model.predict(X)

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/processed/processed_business_data.csv")
    
    # Prepare features and target (predict sales)
    X_train, X_test, y_train, y_test, features = prepare_features_and_target(
        df, target_col='sales'
    )
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    eval_df = evaluate_models(models, X_train, X_test, y_train, y_test)
    print("\nModel Evaluation:")
    print(eval_df)
    
    # Plot comparison
    plot_model_comparison(eval_df)
    
    # Save best model
    best_name, best_model = save_best_model(
        models, eval_df, save_path="../models/best_sales_model.pkl"
    )
    print(f"\nBest model: {best_name}")
