import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class DataVisualizer:
    """
    Professional data visualization suite with industry-grade styling
    Focused on: Numerical Distributions, Categorical Analysis, Correlation Maps
    """
    
    def __init__(self):
        self.theme = self._get_professional_theme()
        self.charts = []
    
    def _get_professional_theme(self):
        """Professional color theme and styling"""
        return {
            'colors': {
                'primary': '#2E4057',      # Dark blue-gray
                'secondary': '#048A81',     # Teal
                'accent': '#54C6EB',       # Light blue
                'warning': '#F39C12',      # Orange
                'danger': '#E74C3C',       # Red
                'success': '#27AE60',      # Green
                'neutral': '#7F8C8D',      # Gray
                'background': '#FFFFFF',    # White
                'grid': '#E8E8E8'          # Light gray
            },
            'fonts': {
                'title': 'Segoe UI, Arial, sans-serif',
                'body': 'Segoe UI, Arial, sans-serif'
            }
        }
    
    def generate_professional_visualizations(self, df):
        """
        Generate focused set of professional visualizations
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            list: Professional chart collection
        """
        self.charts = []
        
        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Numerical Distributions
        if len(numerical_cols) > 0:
            self._create_numerical_distributions(df, numerical_cols)
        
        # 2. Categorical Analysis
        if len(categorical_cols) > 0:
            self._create_categorical_analysis(df, categorical_cols)
        
        # 3. Correlation Analysis
        if len(numerical_cols) > 1:
            self._create_correlation_analysis(df, numerical_cols)
        
        return self.charts
    
    def _create_numerical_distributions(self, df, numerical_cols):
        """Create professional numerical distribution plots"""
        # Limit to top 6 columns for readability
        cols_to_plot = numerical_cols[:6]
        
        if len(cols_to_plot) == 1:
            # Single distribution with enhanced styling
            col = cols_to_plot[0]
            
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=df[col],
                nbinsx=30,
                name='Distribution',
                marker_color=self.theme['colors']['primary'],
                marker_line_color=self.theme['colors']['background'],
                marker_line_width=1,
                opacity=0.8
            ))
            
            # Add mean line
            mean_val = df[col].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color=self.theme['colors']['danger'],
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top"
            )
            
            # Professional styling
            fig.update_layout(
                title={
                    'text': f"Distribution Analysis: {col}",
                    'x': 0.5,
                    'font': {'size': 18, 'family': self.theme['fonts']['title'], 'color': self.theme['colors']['primary']}
                },
                plot_bgcolor=self.theme['colors']['background'],
                paper_bgcolor=self.theme['colors']['background'],
                font={'family': self.theme['fonts']['body'], 'color': self.theme['colors']['primary']},
                height=500,
                showlegend=False
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['colors']['grid'])
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['colors']['grid'])
            
            self.charts.append({
                'title': f'Distribution Analysis: {col}',
                'figure': fig,
                'type': 'numerical_distribution'
            })
        
        else:
            # Multiple distributions in professional grid
            rows = (len(cols_to_plot) + 1) // 2
            cols = 2
            
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=cols_to_plot,
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            colors = [self.theme['colors']['primary'], self.theme['colors']['secondary'], 
                     self.theme['colors']['accent'], self.theme['colors']['success'],
                     self.theme['colors']['warning'], self.theme['colors']['neutral']]
            
            for i, col in enumerate(cols_to_plot):
                row = i // 2 + 1
                col_pos = i % 2 + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        nbinsx=20,
                        name=col,
                        marker_color=colors[i % len(colors)],
                        marker_line_color=self.theme['colors']['background'],
                        marker_line_width=1,
                        opacity=0.8,
                        showlegend=False
                    ),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title={
                    'text': "Numerical Variables Distribution Analysis",
                    'x': 0.5,
                    'font': {'size': 20, 'family': self.theme['fonts']['title'], 'color': self.theme['colors']['primary']}
                },
                plot_bgcolor=self.theme['colors']['background'],
                paper_bgcolor=self.theme['colors']['background'],
                font={'family': self.theme['fonts']['body'], 'color': self.theme['colors']['primary']},
                height=300 * rows
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['colors']['grid'])
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['colors']['grid'])
            
            self.charts.append({
                'title': 'Numerical Variables Distribution Analysis',
                'figure': fig,
                'type': 'numerical_distributions'
            })
    
    def _create_categorical_analysis(self, df, categorical_cols):
        """Create professional categorical analysis"""
        # Focus on top 3 categorical columns
        cols_to_plot = categorical_cols[:3]
        
        for col in cols_to_plot:
            # Get top 10 categories
            value_counts = df[col].value_counts().head(10)
            
            fig = go.Figure()
            
            # Add bar chart with professional styling
            fig.add_trace(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=col,
                marker_color=self.theme['colors']['secondary'],
                marker_line_color=self.theme['colors']['background'],
                marker_line_width=2,
                text=value_counts.values,
                textposition='outside',
                textfont=dict(size=10, color=self.theme['colors']['primary'])
            ))
            
            # Add percentage line
            percentages = (value_counts.values / len(df)) * 100
            fig.add_trace(go.Scatter(
                x=value_counts.index,
                y=percentages,
                mode='lines+markers',
                name='Percentage',
                yaxis='y2',
                line=dict(color=self.theme['colors']['danger'], width=3),
                marker=dict(size=8, color=self.theme['colors']['danger'])
            ))
            
            # Professional layout
            fig.update_layout(
                title={
                    'text': f"Categorical Analysis: {col}",
                    'x': 0.5,
                    'font': {'size': 18, 'family': self.theme['fonts']['title'], 'color': self.theme['colors']['primary']}
                },
                xaxis_title=col,
                yaxis_title='Count',
                yaxis2=dict(
                    title='Percentage (%)',
                    overlaying='y',
                    side='right',
                    range=[0, max(percentages) * 1.1]
                ),
                plot_bgcolor=self.theme['colors']['background'],
                paper_bgcolor=self.theme['colors']['background'],
                font={'family': self.theme['fonts']['body'], 'color': self.theme['colors']['primary']},
                height=500,
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['colors']['grid'])
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['colors']['grid'])
            
            self.charts.append({
                'title': f'Categorical Analysis: {col}',
                'figure': fig,
                'type': 'categorical_analysis'
            })
    
    def _create_correlation_analysis(self, df, numerical_cols):
        """Create professional correlation heatmap"""
        if len(numerical_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create professional heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[
                [0, self.theme['colors']['danger']],
                [0.5, self.theme['colors']['background']],
                [1, self.theme['colors']['success']]
            ],
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12, "color": self.theme['colors']['primary']},
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="Correlation Coefficient", side="right"),
                tickmode="linear",
                tick0=-1,
                dtick=0.5
            )
        ))
        
        fig.update_layout(
            title={
                'text': "Correlation Matrix Analysis",
                'x': 0.5,
                'font': {'size': 20, 'family': self.theme['fonts']['title'], 'color': self.theme['colors']['primary']}
            },
            plot_bgcolor=self.theme['colors']['background'],
            paper_bgcolor=self.theme['colors']['background'],
            font={'family': self.theme['fonts']['body'], 'color': self.theme['colors']['primary']},
            height=max(500, len(numerical_cols) * 50),
            width=max(600, len(numerical_cols) * 50)
        )
        
        self.charts.append({
            'title': 'Correlation Matrix Analysis',
            'figure': fig,
            'type': 'correlation_analysis'
        })
    
    def get_visualization_summary(self):
        """Get summary of generated visualizations"""
        return {
            'total_charts': len(self.charts),
            'chart_types': list(set([chart['type'] for chart in self.charts])),
            'chart_titles': [chart['title'] for chart in self.charts]
        }