from src.insights.report_generator import generate_report
import pandas as pd
import os
import argparse

def main():
    """Generate business insights report from data."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate business insights report.')
    parser.add_argument('--data', help='Path to data CSV file')
    parser.add_argument('--metrics', help='Comma-separated list of metrics to analyze')
    parser.add_argument('--date-col', default='date', help='Name of date column')
    parser.add_argument('--report-type', default='detailed', choices=['detailed', 'executive'],
                       help='Type of report to generate')
    args = parser.parse_args()
    
    # Load data
    if args.data:
        df = pd.read_csv(args.data)
    else:
        # Try to load from standard location
        try:
            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(project_root, "data", "processed", "processed_business_data.csv")
            df = pd.read_csv(data_path)
        except:
            raise ValueError("Could not load data. Please provide a data file path.")
    
    # Parse metrics
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Generate report
    report_path = generate_report(
        df=df,
        metrics=metrics,
        date_col=args.date_col,
        report_type=args.report_type
    )
    
    print(f"Report generated successfully at: {report_path}")
    
    # Create reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # If the report wasn't saved to the reports directory, copy it there
    if 'reports' not in report_path:
        import shutil
        filename = os.path.basename(report_path)
        dest_path = os.path.join('reports', filename)
        shutil.copy2(report_path, dest_path)
        print(f"Report also copied to: {dest_path}")
    
    return report_path

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
