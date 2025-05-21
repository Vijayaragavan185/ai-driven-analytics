import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
import os

def get_project_root():
    script_path = os.path.abspath(__file__)
    dashboard_dir = os.path.dirname(script_path)
    src_dir = os.path.dirname(dashboard_dir)
    project_root = os.path.dirname(src_dir)
    return project_root

def load_data():
    """Load the processed data"""
    root = get_project_root()
    try:
        return pd.read_csv(os.path.join(root, "data", "processed", "processed_business_data.csv"))
    except FileNotFoundError:
        return pd.read_csv(os.path.join(root, "data", "raw", "sample_business_data.csv"))

# Load data for initial layouts
df = load_data()

def create_layout():
    """Creates the main dashboard layout"""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("AI-Driven Business Analytics Dashboard", 
                        className="text-center my-4"),
                html.P("Interactive visualization of business metrics and predictions",
                      className="text-center text-muted mb-5")
            ])
        ]),
        
        # Filters Row
        dbc.Row([
            dbc.Col([
                html.H4("Filters"),
                dbc.Card([
                    dbc.CardBody([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=df['date'].min() if 'date' in df else None,
                            end_date=df['date'].max() if 'date' in df else None,
                            display_format='YYYY-MM-DD'
                        ),
                        
                        html.Div(className="mt-3"),
                        
                        html.Label("Metrics to Display:"),
                        dcc.Checklist(
                            id='metrics-checklist',
                            options=[
                                {'label': ' Sales', 'value': 'sales'},
                                {'label': ' Marketing Spend', 'value': 'marketing_spend'},
                                {'label': ' Customer Count', 'value': 'customer_count'},
                                {'label': ' Marketing Efficiency', 'value': 'marketing_efficiency'}
                            ],
                            value=['sales', 'marketing_spend'],
                            inline=True
                        )
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Main Charts Row
        dbc.Row([
            # Time Series Chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Metrics Over Time"),
                    dbc.CardBody([
                        dcc.Graph(id='time-series-chart')
                    ])
                ])
            ], md=8),
            
            # Summary Statistics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Key Metrics"),
                    dbc.CardBody(id='summary-stats')
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Predictive Insights"),
                    dbc.CardBody([
                        html.Div(id='prediction-insights')
                    ])
                ])
            ], md=4)
        ], className="mb-4"),
        
        # Analytics Row
        dbc.Row([
            # Correlation Heatmap
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Correlation Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='correlation-heatmap')
                    ])
                ])
            ], md=6),
            
            # Feature Importance
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Feature Importance"),
                    dbc.CardBody([
                        dcc.Graph(id='feature-importance')
                    ])
                ])
            ], md=6)
        ], className="mb-4"),
        
        # Sales Prediction Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sales Prediction"),
                    dbc.CardBody([
                        dcc.Graph(id='actual-vs-predicted')
                    ])
                ])
            ])
        ])
    ], fluid=True)
