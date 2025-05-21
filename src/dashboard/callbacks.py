from dash import Input, Output, html, dcc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from dateutil import parser
import dash_bootstrap_components as dbc

def get_project_root():
    script_path = os.path.abspath(__file__)
    dashboard_dir = os.path.dirname(script_path)
    src_dir = os.path.dirname(dashboard_dir)
    project_root = os.path.dirname(src_dir)
    return project_root

def load_model():
    """Load the best model if available"""
    root = get_project_root()
    model_path = os.path.join(root, "models", "best_sales_model.pkl")
    try:
        return joblib.load(model_path)
    except:
        return None

def register_callbacks(app):
    """Register all callbacks for the dashboard"""
    
    @app.callback(
        Output('time-series-chart', 'figure'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('metrics-checklist', 'value')]
    )
    def update_time_series(start_date, end_date, metrics):
        # Load data
        root = get_project_root()
        df = pd.read_csv(os.path.join(root, "data", "processed", "processed_business_data.csv"))
        
        # Convert date column to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if provided
            if start_date and end_date:
                start_date = parser.parse(start_date)
                end_date = parser.parse(end_date)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Create figure
        fig = go.Figure()
        
        # Add selected metrics
        for metric in metrics:
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title()
                ))
                
        fig.update_layout(
            title='Metrics Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Metrics',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('summary-stats', 'children'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_summary_stats(start_date, end_date):
        # Load data
        root = get_project_root()
        df = pd.read_csv(os.path.join(root, "data", "processed", "processed_business_data.csv"))
        
        # Convert date column to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if provided
            if start_date and end_date:
                start_date = parser.parse(start_date)
                end_date = parser.parse(end_date)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Calculate summary statistics
        stats = []
        key_metrics = ['sales', 'marketing_spend', 'customer_count', 'marketing_efficiency']
        
        for metric in key_metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                change = df[metric].iloc[-1] - df[metric].iloc[0] if len(df) > 1 else 0
                pct_change = (change / df[metric].iloc[0] * 100) if len(df) > 1 and df[metric].iloc[0] != 0 else 0
                
                # Format values appropriately
                if metric == 'marketing_efficiency':
                    formatted_mean = f"{mean_val:.2f}"
                elif 'count' in metric:
                    formatted_mean = f"{mean_val:.0f}"
                else:
                    formatted_mean = f"${mean_val:.2f}"
                
                # Determine if change is positive or negative for styling
                color = "success" if pct_change >= 0 else "danger"
                arrow = "↑" if pct_change >= 0 else "↓"
                
                stats.append(
                    html.Div([
                        html.H5(metric.replace('_', ' ').title()),
                        html.H4(formatted_mean),
                        html.P([
                            arrow,
                            f" {abs(pct_change):.1f}%"
                        ], className=f"text-{color}")
                    ], className="mb-4")
                )
        
        return stats
    
    @app.callback(
        Output('correlation-heatmap', 'figure'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_correlation(start_date, end_date):
        # Load data
        root = get_project_root()
        df = pd.read_csv(os.path.join(root, "data", "processed", "processed_business_data.csv"))
        
        # Convert date column to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if provided
            if start_date and end_date:
                start_date = parser.parse(start_date)
                end_date = parser.parse(end_date)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Keep only business metrics for cleaner visualization
        business_metrics = ['sales', 'marketing_spend', 'customer_count', 
                           'marketing_efficiency', 'sales_per_customer']
        
        # Filter to metrics that exist in the dataframe
        metrics = [col for col in business_metrics if col in numeric_cols]
        
        # Calculate correlation
        corr = df[metrics].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr,
            x=metrics,
            y=metrics,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto=True
        )
        
        fig.update_layout(
            title='Correlation Between Business Metrics',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('actual-vs-predicted', 'figure'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_prediction_chart(start_date, end_date):
        # Load data
        root = get_project_root()
        try:
            df = pd.read_csv(os.path.join(root, "data", "processed", "results_with_predictions.csv"))
            has_predictions = 'predicted_sales' in df.columns
        except:
            df = pd.read_csv(os.path.join(root, "data", "processed", "processed_business_data.csv"))
            has_predictions = False
        
        # Convert date column to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if provided
            if start_date and end_date:
                start_date = parser.parse(start_date)
                end_date = parser.parse(end_date)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Create figure
        fig = go.Figure()
        
        # Add actual sales
        if 'sales' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['sales'],
                mode='lines',
                name='Actual Sales'
            ))
        
        # Add predicted sales if available
        if has_predictions:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['predicted_sales'],
                mode='lines',
                name='Predicted Sales',
                line=dict(dash='dash')
            ))
        
        fig.update_layout(
            title='Actual vs Predicted Sales',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            legend_title='Type',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('feature-importance', 'figure'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_feature_importance(start_date, end_date):
        # Load model
        model = load_model()
        
        # If no model is available, return empty figure
        if model is None:
            return go.Figure()
        
        # Get feature importance if available in model
        if hasattr(model, 'feature_importances_'):
            # Load data to get feature names
            root = get_project_root()
            df = pd.read_csv(os.path.join(root, "data", "processed", "processed_business_data.csv"))
            
            # Convert date column to datetime if it's not
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Select numeric columns as potential features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column
            if 'sales' in numeric_cols:
                numeric_cols.remove('sales')
            
            # Ensure we have the right number of features
            if len(numeric_cols) == len(model.feature_importances_):
                importance_df = pd.DataFrame({
                    'feature': numeric_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                # Show top 10 features
                importance_df = importance_df.tail(10)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance for Sales Prediction'
                )
                
                fig.update_layout(
                    yaxis_title='',
                    xaxis_title='Importance',
                    template='plotly_white'
                )
                
                return fig
        
        # Return empty figure if feature importance not available
        return go.Figure()
    
    @app.callback(
        Output('prediction-insights', 'children'),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_prediction_insights(start_date, end_date):
        # Load data
        root = get_project_root()
        try:
            df = pd.read_csv(os.path.join(root, "data", "processed", "results_with_predictions.csv"))
            has_predictions = 'predicted_sales' in df.columns
        except:
            has_predictions = False
            return html.P("No prediction data available. Run the full analytics pipeline to generate predictions.")
        
        if not has_predictions:
            return html.P("No prediction data available. Run the full analytics pipeline to generate predictions.")
            
        # Convert date column to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if provided
            if start_date and end_date:
                start_date = parser.parse(start_date)
                end_date = parser.parse(end_date)
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Calculate prediction accuracy metrics
        if 'sales' in df.columns and 'predicted_sales' in df.columns:
            mae = np.mean(np.abs(df['sales'] - df['predicted_sales']))
            mape = np.mean(np.abs((df['sales'] - df['predicted_sales']) / df['sales'])) * 100
            r2 = 1 - (np.sum((df['sales'] - df['predicted_sales'])**2) / 
                      np.sum((df['sales'] - df['sales'].mean())**2))
            
            # Find days with largest discrepancy
            df['abs_error'] = np.abs(df['sales'] - df['predicted_sales'])
            top_errors = df.nlargest(3, 'abs_error')
            
            insights = [
                html.H5("Model Accuracy"),
                html.P([
                    "Mean Absolute Error: ",
                    html.Strong(f"${mae:.2f}")
                ]),
                html.P([
                    "Mean Percentage Error: ",
                    html.Strong(f"{mape:.1f}%")
                ]),
                html.P([
                    "R² Score: ",
                    html.Strong(f"{r2:.3f}")
                ]),
                
                html.H5("Unusual Days", className="mt-3"),
                html.P("Days with largest prediction errors:")
            ]
            
            # Add list of unusual days
            error_list = []
            for _, row in top_errors.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
                error_list.append(
                    html.Li([
                        html.Strong(f"{date_str}: "),
                        f"Predicted ${row['predicted_sales']:.2f}, Actual ${row['sales']:.2f}",
                        html.Span(f" (${row['abs_error']:.2f} off)", 
                                className="text-danger")
                    ])
                )
            
            insights.append(html.Ul(error_list))
            
            return insights
        
        return html.P("Prediction data incomplete. Run the full analytics pipeline to generate complete predictions.")
