# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_plotting_style():
    """Set the style for all visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_time_series(df, date_col, value_cols, title=None, save_path=None):
    """
    Plot time series data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    date_col : str
        Name of date column
    value_cols : list of str
        Names of value columns to plot
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    set_plotting_style()
    
    fig, ax = plt.subplots()
    
    for col in value_cols:
        ax.plot(df[date_col], df[col], label=col)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Time Series Plot')
    
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def plot_correlation_matrix(df, title=None, save_path=None):
    """
    Plot correlation matrix of numerical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    set_plotting_style()
    
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, 
                fmt=".2f", vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5)
    
    if title:
        plt.title(title)
    else:
        plt.title('Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def create_dashboard(df, save_path=None):
    """
    Create a comprehensive dashboard of visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str, optional
        Path to save the dashboard
    """
    set_plotting_style()
    
    # Create subplot grid
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # Plot 1: Time Series of Sales
    ax1 = fig.add_subplot(gs[0, 0])
    if 'date' in df.columns and 'sales' in df.columns:
        ax1.plot(df['date'], df['sales'], color='blue')
        ax1.set_title('Sales Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales ($)')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Marketing Spend vs Sales Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    if 'marketing_spend' in df.columns and 'sales' in df.columns:
        ax2.scatter(df['marketing_spend'], df['sales'], alpha=0.6)
        ax2.set_title('Marketing Spend vs. Sales')
        ax2.set_xlabel('Marketing Spend ($)')
        ax2.set_ylabel('Sales ($)')
    
    # Plot 3: Sales by Day of Week
    ax3 = fig.add_subplot(gs[1, 0])
    if 'day_of_week' in df.columns and 'sales' in df.columns:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        sales_by_day = df.groupby('day_of_week')['sales'].mean().reindex(range(7))
        ax3.bar(day_names, sales_by_day)
        ax3.set_title('Average Sales by Day of Week')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Average Sales ($)')
    
    # Plot 4: Marketing Efficiency Over Time
    ax4 = fig.add_subplot(gs[1, 1])
    if 'date' in df.columns and 'marketing_efficiency' in df.columns:
        ax4.plot(df['date'], df['marketing_efficiency'], color='green')
        ax4.set_title('Marketing Efficiency Over Time')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sales per $ Spent')
        ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Distribution of Sales
    ax5 = fig.add_subplot(gs[2, 0])
    if 'sales' in df.columns:
        sns.histplot(df['sales'], kde=True, ax=ax5)
        ax5.set_title('Distribution of Sales')
        ax5.set_xlabel('Sales ($)')
        ax5.set_ylabel('Frequency')
    
    # Plot 6: Correlation Heatmap of Key Metrics
    ax6 = fig.add_subplot(gs[2, 1])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    key_metrics = [col for col in ['sales', 'customer_count', 'marketing_spend', 
                                  'marketing_efficiency', 'sales_per_customer'] 
                   if col in numeric_cols]
    if len(key_metrics) > 1:
        sns.heatmap(df[key_metrics].corr(), annot=True, cmap='coolwarm', ax=ax6)
        ax6.set_title('Correlation Between Key Metrics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/processed/processed_business_data.csv")
    create_dashboard(df, save_path="../data/visualizations/dashboard.png")
