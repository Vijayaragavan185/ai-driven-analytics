import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
from .trend_analyzer import analyze_business_metrics
from .anomaly_detector import analyze_anomalies

class ReportGenerator:
    """Generates natural language reports from data analysis."""
    
    def __init__(self, df, date_col='date'):
        """
        Initialize report generator with dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing business data
        date_col : str
            Name of date column
        """
        self.df = df.copy()
        self.date_col = date_col
        
        # Convert date column to datetime if it exists
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Sort by date to ensure chronological analysis
            self.df = self.df.sort_values(date_col)
            
            # Set date range attributes
            self.start_date = self.df[date_col].min()
            self.end_date = self.df[date_col].max()
            self.date_range_str = f"{self.start_date.strftime('%B %d, %Y')} to {self.end_date.strftime('%B %d, %Y')}"
        else:
            self.start_date = None
            self.end_date = None
            self.date_range_str = "entire period"
        
        # Initialize placeholder for analyses
        self.trend_analysis = None
        self.anomaly_analysis = None
    
    def analyze_data(self, metrics=None):
        """
        Perform trend and anomaly analysis on the data.
        
        Parameters:
        -----------
        metrics : list of str, optional
            List of metrics to analyze. If None, common business metrics will be used.
        """
        # If metrics is None, try to identify common business metrics
        if metrics is None:
            metrics = []
            common_metrics = ['sales', 'revenue', 'marketing_spend', 'customer_count', 
                             'marketing_efficiency', 'sales_per_customer', 'profit']
            for metric in common_metrics:
                if metric in self.df.columns:
                    metrics.append(metric)
        
        # Perform trend analysis
        self.trend_analysis = analyze_business_metrics(self.df, metrics, date_col=self.date_col)
        
        # Perform anomaly analysis
        self.anomaly_analysis = analyze_anomalies(self.df, metrics, date_col=self.date_col)
        
        return self
    
    def get_executive_summary(self):
        """
        Generate an executive summary of the key findings.
        
        Returns:
        --------
        str
            Executive summary text
        """
        if self.trend_analysis is None:
            return "No analysis has been performed. Call analyze_data() first."
        
        # Initialize summary sections
        intro = self._generate_intro()
        trend_highlights = self._generate_trend_highlights()
        anomaly_highlights = self._generate_anomaly_highlights()
        recommendations = self._generate_recommendations()
        
        # Combine all sections
        summary = f"""# Executive Summary: Business Performance Analysis

{intro}

## Key Trends

{trend_highlights}

## Notable Anomalies

{anomaly_highlights}

## Recommendations

{recommendations}
"""
        return summary
    
    def get_detailed_report(self):
        """
        Generate a detailed report with all findings.
        
        Returns:
        --------
        str
            Detailed report text
        """
        if self.trend_analysis is None:
            return "No analysis has been performed. Call analyze_data() first."
        
        # Generate executive summary
        executive_summary = self.get_executive_summary()
        
        # Generate detailed sections for each metric
        detailed_sections = []
        
        for metric, analysis in self.trend_analysis.items():
            section = self._generate_metric_section(metric, analysis)
            detailed_sections.append(section)
        
        # Combine all sections
        report = f"""{executive_summary}

# Detailed Analysis

{chr(10).join(detailed_sections)}

## Analysis Methodology

This report was automatically generated by analyzing historical data using:
1. Trend analysis with linear regression and time series decomposition
2. Anomaly detection using statistical methods and machine learning
3. Correlation analysis between business metrics

For more information, please contact the data science team.
"""
        return report
    
    def _generate_intro(self):
        """Generate introduction paragraph."""
        # Count metrics analyzed
        num_metrics = len(self.trend_analysis)
        metrics_list = list(self.trend_analysis.keys())
        
        # Format metrics for readability
        if num_metrics == 1:
            metrics_text = metrics_list[0].replace('_', ' ')
        elif num_metrics == 2:
            metrics_text = f"{metrics_list[0].replace('_', ' ')} and {metrics_list[1].replace('_', ' ')}"
        else:
            metrics_text = ", ".join([m.replace('_', ' ') for m in metrics_list[:-1]])
            metrics_text += f", and {metrics_list[-1].replace('_', ' ')}"
        
        # Find most significant metrics (with highest R²)
        r2_values = {}
        for metric, analysis in self.trend_analysis.items():
            r2_values[metric] = analysis['trend']['r_squared']
        
        top_metric = max(r2_values, key=r2_values.get)
        top_trend = self.trend_analysis[top_metric]['trend']
        
        # Generate intro text
        intro = f"""This report analyzes {metrics_text} from {self.date_range_str}. **The most significant trend observed was in {top_metric.replace('_', ' ')}**, which showed a {top_trend['interpretation'].lower()} (R² = {top_trend['r_squared']:.2f}). """
        
        # Add information about overall business performance
        if 'sales' in self.trend_analysis or 'revenue' in self.trend_analysis:
            performance_metric = 'sales' if 'sales' in self.trend_analysis else 'revenue'
            direction = self.trend_analysis[performance_metric]['trend']['direction']
            change_pct = self.trend_analysis[performance_metric]['trend']['overall_change_pct']
            
            if direction == 'up':
                intro += f"Overall business performance has been positive with {performance_metric.replace('_', ' ')} {self.trend_analysis[performance_metric]['trend']['trending_term']} by {abs(change_pct):.1f}% over this period."
            else:
                intro += f"Overall business performance has been challenging with {performance_metric.replace('_', ' ')} {self.trend_analysis[performance_metric]['trend']['trending_term']} by {abs(change_pct):.1f}% over this period."
        
        return intro
    
    def _generate_trend_highlights(self):
        """Generate highlights of the most important trends."""
        highlights = []
        
        # Prioritize metrics
        priority_order = ['sales', 'revenue', 'profit', 'marketing_efficiency', 
                          'sales_per_customer', 'customer_count', 'marketing_spend']
        
        # Sort metrics by priority and significance
        metrics_by_significance = {}
        for metric, analysis in self.trend_analysis.items():
            # Calculate significance score (combination of R² and priority)
            priority_score = 10 - priority_order.index(metric) if metric in priority_order else 0
            significance_score = analysis['trend']['r_squared'] * 10 + priority_score
            metrics_by_significance[metric] = significance_score
        
        # Sort metrics by significance score
        sorted_metrics = sorted(metrics_by_significance.keys(), 
                               key=lambda k: metrics_by_significance[k], 
                               reverse=True)
        
        # Generate highlights for top 3 most significant metrics
        for i, metric in enumerate(sorted_metrics[:3]):
            analysis = self.trend_analysis[metric]
            trend = analysis['trend']
            
            # Format the metric name
            metric_name = metric.replace('_', ' ').title()
            
            # Determine the trend language
            if trend['significant']:
                if trend['direction'] == 'up':
                    trend_text = f"**{metric_name} has shown a significant increase** of {abs(trend['overall_change_pct']):.1f}% over the period"
                else:
                    trend_text = f"**{metric_name} has shown a significant decrease** of {abs(trend['overall_change_pct']):.1f}% over the period"
            else:
                if trend['direction'] == 'up':
                    trend_text = f"**{metric_name} has increased slightly** by {abs(trend['overall_change_pct']):.1f}% over the period"
                else:
                    trend_text = f"**{metric_name} has decreased slightly** by {abs(trend['overall_change_pct']):.1f}% over the period"
            
            # Add seasonality information if available
            if analysis['seasonality']['has_seasonality'] and 'peak_day' in analysis['seasonality']:
                peak_day = analysis['seasonality']['peak_day']
                low_day = analysis['seasonality']['low_day']
                trend_text += f", with peaks typically occurring on {peak_day}s and lows on {low_day}s"
            
            # Add recent trend information
            recent_trend = trend['recent_trend']
            trend_text += f". The metric is currently {recent_trend}."
            
            highlights.append(f"{i+1}. {trend_text}")
        
        # Check for strong correlations between metrics
        correlations = []
        for i, metric1 in enumerate(sorted_metrics):
            for metric2 in sorted_metrics[i+1:]:
                if f"correlation_{metric1}_{metric2}" in self.anomaly_analysis:
                    corr_analysis = self.anomaly_analysis[f"correlation_{metric1}_{metric2}"]
                    if abs(corr_analysis['correlation']) > 0.7:
                        corr_type = "strong positive" if corr_analysis['correlation'] > 0 else "strong negative"
                        correlations.append(f"There is a {corr_type} correlation (r = {corr_analysis['correlation']:.2f}) between {metric1.replace('_', ' ')} and {metric2.replace('_', ' ')}.")
        
        # Add a correlation highlight if found
        if correlations:
            highlights.append(f"{len(highlights)+1}. {correlations[0]}")
        
        return "\n".join(highlights)
    
    def _generate_anomaly_highlights(self):
        """Generate highlights of the most notable anomalies."""
        highlights = []
        
        # Look for time series anomalies first
        time_series_anomalies = {}
        for key, analysis in self.anomaly_analysis.items():
            if 'time_series' in key and analysis['num_anomalies'] > 0:
                metric = key.split('_time_series')[0]
                time_series_anomalies[metric] = analysis
        
        # Sort metrics by number of anomalies
        sorted_anomalies = sorted(time_series_anomalies.keys(), 
                                 key=lambda k: time_series_anomalies[k]['num_anomalies'],
                                 reverse=True)
        
        # Generate highlights for top 2 metrics with most anomalies
        for i, metric in enumerate(sorted_anomalies[:2]):
            analysis = time_series_anomalies[metric]
            
            # Format the metric name
            metric_name = metric.replace('_', ' ').title()
            
            # Get the most recent anomaly
            recent_anomalies = sorted(analysis['anomaly_dates'], reverse=True)
            if recent_anomalies:
                # Convert to datetime if it's a string
                recent_date = recent_anomalies[0]
                if isinstance(recent_date, str):
                    recent_date = pd.to_datetime(recent_date)
                
                # Get the corresponding row
                row = analysis['anomaly_rows'][analysis['anomaly_rows'][self.date_col] == recent_date].iloc[0]
                
                # Determine if it's high or low
                direction = "spike" if row['deviation'] > 0 else "drop"
                
                anomaly_text = f"**Unusual {direction} in {metric_name}** detected on {recent_date.strftime('%B %d, %Y') if isinstance(recent_date, pd.Timestamp) else recent_date}, with a {abs(row['deviation']):.1f}% deviation from expected values"
                highlights.append(f"{i+1}. {anomaly_text}")
        
        # Look for correlation anomalies
        correlation_anomalies = {}
        for key, analysis in self.anomaly_analysis.items():
            if key.startswith('correlation_') and analysis['num_anomalies'] > 0:
                correlation_anomalies[key] = analysis
        
        # If we have correlation anomalies, add one highlight
        if correlation_anomalies:
            # Pick the one with highest R²
            top_corr = max(correlation_anomalies.keys(), 
                          key=lambda k: correlation_anomalies[k]['r_squared'])
            
            analysis = correlation_anomalies[top_corr]
            metrics = [m.replace('_', ' ').title() for m in analysis['columns']]
            
            # Get the most recent anomaly
            if 'anomaly_dates' in analysis and analysis['anomaly_dates']:
                recent_anomalies = sorted(analysis['anomaly_dates'], reverse=True)
                recent_date = recent_anomalies[0]
                if isinstance(recent_date, str):
                    recent_date = pd.to_datetime(recent_date)
                
                date_str = recent_date.strftime('%B %d, %Y') if isinstance(recent_date, pd.Timestamp) else recent_date
                corr_text = f"**Unusual relationship** between {metrics[0]} and {metrics[1]} detected on {date_str}, deviating from the typical pattern"
                highlights.append(f"{len(highlights)+1}. {corr_text}")
        
        # If no highlights were generated, add a default message
        if not highlights:
            highlights.append("No significant anomalies were detected in the data during this period.")
        
        return "\n".join(highlights)
    
    def _generate_recommendations(self):
        """Generate business recommendations based on the analysis."""
        recommendations = []
        
        # Check for sales trend
        if 'sales' in self.trend_analysis:
            sales_trend = self.trend_analysis['sales']['trend']
            
            if sales_trend['direction'] == 'down' and sales_trend['significant']:
                # Sales are significantly decreasing
                recommendations.append("**Develop a sales recovery plan** to address the significant downward trend in sales. Consider promotions, pricing strategies, or new market segments.")
            elif sales_trend['direction'] == 'up' and sales_trend['recent_trend'] == 'growth is slowing':
                # Sales growth is slowing
                recommendations.append("**Investigate why sales growth is slowing** and develop strategies to reinvigorate growth. Consider new product offerings or expanding into new customer segments.")
        
        # Check for marketing efficiency
        if 'marketing_spend' in self.trend_analysis and 'sales' in self.trend_analysis:
            marketing_trend = self.trend_analysis['marketing_spend']['trend']
            sales_trend = self.trend_analysis['sales']['trend']
            
            # Check correlation
            corr_key = 'correlation_marketing_spend_sales'
            if corr_key not in self.anomaly_analysis and 'correlation_sales_marketing_spend' in self.anomaly_analysis:
                corr_key = 'correlation_sales_marketing_spend'
            
            if corr_key in self.anomaly_analysis:
                corr = self.anomaly_analysis[corr_key]['correlation']
                
                if marketing_trend['direction'] == 'up' and sales_trend['direction'] == 'down':
                    recommendations.append("**Review marketing strategy urgently**. Marketing spend is increasing while sales are decreasing, suggesting ineffective marketing campaigns or channels.")
                elif marketing_trend['direction'] == 'up' and corr < 0.5:
                    recommendations.append("**Optimize marketing allocation** across channels. Current marketing spend shows weak correlation with sales performance, indicating potential inefficiencies.")
        
        # Check for customer metrics
        if 'customer_count' in self.trend_analysis:
            customer_trend = self.trend_analysis['customer_count']['trend']
            
            if customer_trend['direction'] == 'down' and customer_trend['significant']:
                recommendations.append("**Implement customer retention initiatives** to address the significant decline in customer count. Consider loyalty programs or improved customer service.")
            
            if 'sales_per_customer' in self.trend_analysis:
                spc_trend = self.trend_analysis['sales_per_customer']['trend']
                
                if spc_trend['direction'] == 'down' and customer_trend['direction'] == 'up':
                    recommendations.append("**Focus on increasing customer value**. While customer count is growing, the average value per customer is declining, suggesting upselling opportunities.")
        
        # Check for seasonality
        seasonality_insights = []
        for metric, analysis in self.trend_analysis.items():
            if analysis['seasonality']['has_seasonality'] and 'peak_day' in analysis['seasonality']:
                metric_name = metric.replace('_', ' ')
                peak_day = analysis['seasonality']['peak_day']
                seasonality_insights.append((metric_name, peak_day))
        
        if seasonality_insights:
            metric_name, peak_day = seasonality_insights[0]
            recommendations.append(f"**Adjust resource allocation** to account for {metric_name} seasonality, with particular focus on {peak_day}s when activity peaks.")
        
        # Add general recommendations if we don't have enough specific ones
        if len(recommendations) < 3:
            general_recommendations = [
                "**Implement regular data review meetings** to discuss metrics and trends with key stakeholders, ensuring data-driven decision making across the organization.",
                "**Expand data collection** to include more granular metrics like customer segmentation, product-level performance, and marketing channel attribution for deeper insights.",
                "**Develop predictive models** for key business metrics to anticipate changes before they occur and take proactive measures."
            ]
            
            # Add general recommendations until we have at least 3
            for rec in general_recommendations:
                if len(recommendations) >= 3:
                    break
                recommendations.append(rec)
        
        return "\n".join(recommendations)
    
    def _generate_metric_section(self, metric, analysis):
        """Generate a detailed section for a specific metric."""
        # Format the metric name for display
        metric_display = metric.replace('_', ' ').title()
        
        # Get trend information
        trend = analysis['trend']
        trend_interpretation = trend['interpretation']
        r_squared = trend['r_squared']
        overall_change_pct = trend['overall_change_pct']
        
        # Get seasonality information
        seasonality = analysis['seasonality']
        has_seasonality = seasonality['has_seasonality']
        
        # Get change point information
        change_points = analysis['change_points']
        has_change_points = change_points['has_change_points']
        
        # Build the section
        section = f"""## {metric_display} Analysis

### Trend Analysis
{metric_display} shows a {trend_interpretation.lower()} (R² = {r_squared:.3f}) with an overall change of {overall_change_pct:.1f}% over the analyzed period. The metric is currently {trend['recent_trend']}.
"""

        # Add seasonality information if available
        if has_seasonality:
            section += "\n### Seasonality\n"
            if 'peak_day' in seasonality:
                peak_day = seasonality['peak_day']
                low_day = seasonality['low_day']
                day_variance = seasonality['day_variance_pct']
                section += f"Strong weekly seasonality detected with peaks on {peak_day}s and lows on {low_day}s. The day-to-day variance is approximately {day_variance:.1f}%.\n"
                
                if 'weekend_effect' in seasonality:
                    weekend_effect = seasonality['weekend_effect']
                    if abs(weekend_effect) > 10:
                        weekend_text = "significantly higher" if weekend_effect > 0 else "significantly lower"
                        section += f"Weekend performance is {weekend_text} than weekdays (by {abs(weekend_effect):.1f}%).\n"
            else:
                section += f"Seasonality detected with strength of {seasonality['seasonality_strength']:.2f}.\n"
        else:
            section += "\n### Seasonality\nNo significant seasonality detected in the data.\n"
        
        # Add change point information if available
        if has_change_points:
            section += "\n### Significant Changes\n"
            num_changes = change_points['num_change_points']
            
            if num_changes == 1:
                section += f"1 significant change point detected.\n"
            else:
                section += f"{num_changes} significant change points detected.\n"
            
            # Add details about the most significant change
            if 'most_significant_date' in change_points:
                date = change_points['most_significant_date']
                if isinstance(date, pd.Timestamp):
                    date_str = date.strftime('%B %d, %Y')
                else:
                    date_str = date
                
                pct_change = change_points['most_significant_pct_change']
                direction = "increase" if pct_change > 0 else "decrease"
                
                section += f"Most notable change: {abs(pct_change):.1f}% {direction} on {date_str}\n"
        else:
            section += "\n### Significant Changes\nNo abrupt changes detected in the time series.\n"
        
        # Add anomaly information
        anomaly_key = f"{metric}_time_series"
        if anomaly_key in self.anomaly_analysis:
            anomalies = self.anomaly_analysis[anomaly_key]
            section += "\n### Anomalies\n"
            
            if anomalies['num_anomalies'] > 0:
                section += f"{anomalies['num_anomalies']} anomalies detected ({anomalies['anomalies_pct']:.1f}% of data points).\n"
                
                # List top 3 anomalies
                if 'anomaly_rows' in anomalies and len(anomalies['anomaly_rows']) > 0:
                    rows = anomalies['anomaly_rows'].sort_values('zscore', ascending=False).head(3)
                    section += "Top anomalies:\n"
                    
                    for _, row in rows.iterrows():
                        date = row[self.date_col]
                        if isinstance(date, pd.Timestamp):
                            date_str = date.strftime('%B %d, %Y')
                        else:
                            date_str = date
                        
                        deviation = row['deviation'] if 'deviation' in row else 0
                        direction = "above" if deviation > 0 else "below"
                        
                        section += f"- {date_str}: {abs(deviation):.1f}% {direction} expected value\n"
            else:
                section += "No significant anomalies detected.\n"
        
        return section
    
    def save_report(self, output_path=None, report_type='detailed'):
        """
        Save the generated report to a file.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the report. If None, will use a default path.
        report_type : str
            Type of report to generate: 'executive' or 'detailed'
            
        Returns:
        --------
        str
            Path where the report was saved
        """
        if self.trend_analysis is None:
            raise ValueError("No analysis has been performed. Call analyze_data() first.")
        
        # Generate the report
        if report_type == 'executive':
            report = self.get_executive_summary()
            report_name = "executive_summary"
        else:
            report = self.get_detailed_report()
            report_name = "detailed_report"
        
        # Create default output path if none provided
        if output_path is None:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get project root
            script_path = os.path.abspath(__file__)
            insights_dir = os.path.dirname(script_path)
            src_dir = os.path.dirname(insights_dir)
            project_root = os.path.dirname(src_dir)
            
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(project_root, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            output_path = os.path.join(reports_dir, f"{report_name}_{timestamp}.md")
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")
        return output_path

def generate_report(df=None, metrics=None, date_col='date', report_type='detailed'):
    """
    Generate a business insights report from data.
    
    Parameters:
    -----------
    df : pandas.DataFrame, optional
        DataFrame containing business data. If None, will try to load from standard location.
    metrics : list of str, optional
        List of metrics to analyze. If None, common business metrics will be used.
    date_col : str
        Name of date column
    report_type : str
        Type of report to generate: 'executive' or 'detailed'
        
    Returns:
    --------
    str
        Path to the saved report
    """
    # If no dataframe provided, try to load from standard location
    if df is None:
        script_path = os.path.abspath(__file__)
        insights_dir = os.path.dirname(script_path)
        src_dir = os.path.dirname(insights_dir)
        project_root = os.path.dirname(src_dir)
        
        try:
            data_path = os.path.join(project_root, "data", "processed", "processed_business_data.csv")
            df = pd.read_csv(data_path)
        except:
            raise ValueError("Could not load data automatically. Please provide a dataframe.")
    
    # Create report generator
    generator = ReportGenerator(df, date_col=date_col)
    
    # Analyze data
    generator.analyze_data(metrics=metrics)
    
    # Generate and save report
    return generator.save_report(report_type=report_type)

if __name__ == "__main__":
    # Example of how to use the report generator
    try:
        report_path = generate_report()
        print(f"Report generated successfully at {report_path}")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
