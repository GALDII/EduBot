import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Any, Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_interactive_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = ""
) -> go.Figure:
    """Create interactive Plotly chart."""
    try:
        fig = None
        
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title,
                        labels={x_col: xlabel or x_col, y_col: ylabel or y_col},
                        color=y_col, color_continuous_scale="Viridis")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title,
                         labels={x_col: xlabel or x_col, y_col: ylabel or y_col},
                         markers=True)
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title,
                           labels={x_col: xlabel or x_col, y_col: ylabel or y_col})
        elif chart_type == "barh":
            fig = px.bar(df, x=y_col, y=x_col, orientation='h', title=title,
                        labels={x_col: xlabel or x_col, y_col: ylabel or y_col})
        else:
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        
        # Update layout for dark theme
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1f2e",
            font=dict(color="#e2e8f0", size=12),
            title_font=dict(size=16, color="#e2e8f0"),
            hovermode='closest'
        )
        
        return fig
    
    except Exception as e:
        # Fallback to simple chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df[x_col].astype(str), y=df[y_col]))
        fig.update_layout(template="plotly_dark", title=title or "Chart")
        return fig


def create_performance_heatmap(df: pd.DataFrame, semester_col: str = None) -> go.Figure:
    """Create semester-wise subject performance heatmap."""
    try:
        # Find semester column
        if not semester_col:
            for col in df.columns:
                if 'semester' in col.lower() or 'sem' in col.lower():
                    semester_col = col
                    break
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found")
        
        # Prepare data for heatmap
        if semester_col and semester_col in df.columns:
            pivot_data = df.groupby(semester_col)[numeric_cols].mean()
        else:
            pivot_data = df[numeric_cols].T
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns if hasattr(pivot_data, 'columns') else list(range(len(pivot_data.columns))),
            y=pivot_data.index.astype(str) if hasattr(pivot_data, 'index') else list(range(len(pivot_data))),
            colorscale='Viridis',
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title="Performance Heatmap",
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1f2e",
            font=dict(color="#e2e8f0")
        )
        
        return fig
    
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Heatmap (Data Error)")
        return fig


def create_comparison_chart(df: pd.DataFrame, semester_col: str, value_col: str) -> go.Figure:
    """Create side-by-side comparison chart for multiple semesters."""
    try:
        if semester_col not in df.columns or value_col not in df.columns:
            raise ValueError("Columns not found")
        
        grouped = df.groupby(semester_col)[value_col].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grouped[semester_col].astype(str),
            y=grouped[value_col],
            marker_color='#60a5fa',
            text=grouped[value_col].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Comparison: {value_col} across Semesters",
            xaxis_title="Semester",
            yaxis_title=value_col,
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1f2e",
            font=dict(color="#e2e8f0")
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Comparison Chart")
        return fig


def create_timeline_chart(events: List[Dict[str, Any]]) -> go.Figure:
    """Create career progression timeline."""
    try:
        fig = go.Figure()
        
        for i, event in enumerate(events):
            fig.add_trace(go.Scatter(
                x=[event.get("date", "")],
                y=[i],
                mode='markers+text',
                marker=dict(size=15, color=event.get("color", "#60a5fa")),
                text=event.get("title", ""),
                textposition="middle right",
                name=event.get("title", "")
            ))
        
        fig.update_layout(
            title="Career Progression Timeline",
            xaxis_title="Date",
            yaxis=dict(showticklabels=False),
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1f2e",
            font=dict(color="#e2e8f0"),
            height=300
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Timeline")
        return fig


def create_skill_gap_chart(user_skills: List[str], required_skills: List[str]) -> go.Figure:
    """Create visual skill gap analysis chart."""
    try:
        matched = set(user_skills) & set(required_skills)
        missing = set(required_skills) - set(user_skills)
        extra = set(user_skills) - set(required_skills)
        
        categories = ['Matched Skills', 'Missing Skills', 'Extra Skills']
        counts = [len(matched), len(missing), len(extra)]
        colors = ['#4ade80', '#f97316', '#60a5fa']
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=counts, marker_color=colors, text=counts, textposition='outside')
        ])
        
        fig.update_layout(
            title="Skill Gap Analysis",
            yaxis_title="Number of Skills",
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1f2e",
            font=dict(color="#e2e8f0")
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Skill Gap Chart")
        return fig


def create_readiness_radar(score_breakdown: Dict[str, float]) -> go.Figure:
    """Create radar chart for career readiness score breakdown."""
    try:
        categories = list(score_breakdown.keys())
        values = list(score_breakdown.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Profile',
            line_color='#60a5fa'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Career Readiness Breakdown",
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            font=dict(color="#e2e8f0")
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Readiness Radar")
        return fig


def export_chart(fig: go.Figure, format: str = "png") -> bytes:
    """Export Plotly figure to bytes."""
    try:
        if format == "png":
            return fig.to_image(format="png", width=1200, height=600)
        elif format == "pdf":
            return fig.to_image(format="pdf", width=1200, height=600)
        elif format == "svg":
            return fig.to_image(format="svg", width=1200, height=600)
        else:
            return fig.to_image(format="png")
    except Exception as e:
        return b""

