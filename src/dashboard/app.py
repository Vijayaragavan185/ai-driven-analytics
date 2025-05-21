import dash
from dash import html
import dash_bootstrap_components as dbc
from .layout import create_layout
from .callbacks import register_callbacks

def create_dashboard():
    """Create and configure the Dash application"""
    # Initialize the Dash app with Bootstrap theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="AI-Driven Analytics Dashboard"
    )
    
    # Set up the layout
    app.layout = create_layout()
    
    # Register callbacks
    register_callbacks(app)
    
    return app

def run_dashboard(debug=True, port=8050):
    """Run the dashboard application"""
    app = create_dashboard()
    app.run(debug=debug, port=port)
    
if __name__ == "__main__":
    run_dashboard()
