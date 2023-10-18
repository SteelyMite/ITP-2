"""
File:           utils.py
Description:    Utility functions for data parsing and generating column summaries for Dash visualizations.
Authors:        Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Date Updated:   2023-10-18
"""
# Essential libraries for data processing and base64 encoding
import base64
import io
import pandas as pd
import plotly.express as px
import dash

# Dash libraries for creating web-based components and handling callbacks
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL

# Bootstrap components for styling and structuring Dash apps
import dash_bootstrap_components as dbc

def parse_contents(contents, file_type):
    """
    Parse the uploaded file content and convert it into a DataFrame.

    Args:
    - contents (str): Base64 encoded content of the uploaded file.
    - file_type (str): The type of the uploaded file, e.g. 'csv', 'excel', 'json'.

    Returns:
    - pd.DataFrame or None: DataFrame containing the parsed data or None if parsing fails.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if file_type == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif file_type == 'excel':
            df = pd.read_excel(io.BytesIO(decoded))
        elif file_type == 'json':
            df = pd.read_json(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

def generate_column_summary_box(df, column_name):
    """
    Generate a Dash card component to visualize the summary of a given column in a DataFrame.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The name of the column to summarize.

    Returns:
    - dbc.Card: A Dash Bootstrap Card component visualizing the column summary.
    """
    # Calculate the number of NaN values in the column
    nan_count = df[column_name].isna().sum()

    if pd.api.types.is_numeric_dtype(df[column_name]):
        # Handle numeric columns
        data_type = "Numeric"
        fig = px.histogram(df, x=column_name, nbins=5)  # Here, 5 bins are used for simplicity; adjust as needed.
        # Extract basic statistics
        min_val = "{:.4f}".format(df[column_name].min())
        max_val = "{:.4f}".format(df[column_name].max())
        mean_val = "{:.4f}".format(df[column_name].mean())
        # Options for cleaning numeric data
        data_cleaning_options = [
            {"label": "Replace NaN with Min", "value": "min"},
            {"label": "Replace NaN with Max", "value": "max"},
            {"label": "Replace NaN with Mean", "value": "mean"},
            {"label": "Replace NaN with Zero", "value": "zero"}
        ]
        # Options for normalization
        normalization_options = [
            {"label": "Normalize (New Column)", "value": "normalize_new"},
            {"label": "Normalize (Replace)", "value": "normalize_replace"},
        ]
        # Dropdown menu for column operations (e.g., data cleaning, normalization)
        dropdown_menu = dbc.DropdownMenu(
            label="Options",
            children=[
                dbc.DropdownMenuItem("Change data type", header=True),
                dbc.DropdownMenuItem("Numeric", id={"type": "convert", "index": column_name, "to": "Numeric"}),
                dbc.DropdownMenuItem("String", id={"type": "convert", "index": column_name, "to": "String"}),
                
                dbc.DropdownMenuItem(divider=True),

                dbc.DropdownMenuItem("Data Cleaning", header=True),
                *[dbc.DropdownMenuItem(item["label"], id={"type": "clean", "index": column_name, "action": item["value"]}) for item in data_cleaning_options],

                dbc.DropdownMenuItem(divider=True),

                dbc.DropdownMenuItem("Normalization", header=True),
                *[dbc.DropdownMenuItem(item["label"], id={"type": "clean", "index": column_name, "action": item["value"]}) for item in normalization_options],
            ],
            className="m-1",
            right=True
        )
        # Return the visualization card for numeric columns
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div(column_name),
                    html.Div(f"Data type: {data_type}", style={'fontSize': '12px', 'color': 'grey'})
                ]),
                dropdown_menu
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '250px'}),
                html.P(f"NaN values: {nan_count}"),
                html.P(f"Min: {min_val}"),
                html.P(f"Max: {max_val}"),
                html.P(f"Mean: {mean_val}")
            ])
        ])
    else:
        # Handle non-numeric (string) columns
        data_type = "String"
        
        # Get the top 10 most frequent string values
        top_values = df[column_name].value_counts().head(10)
        fig = px.bar(top_values, x=top_values.index, y=top_values.values, labels={'x': column_name, 'y': 'Count'})
        
        # Extract relevant statistics for string columns
        unique_count = df[column_name].nunique()
        value_counts = df[column_name].value_counts()
        most_frequent_string = value_counts.index[0] if not value_counts.empty else "N/A"
        least_frequent_string = value_counts.index[-1] if not value_counts.empty else "N/A"
        # Options for cleaning string data
        data_cleaning_options = [
            {"label": "Replace NaN with N/A", "value": "na_string"},
            {"label": f"Replace NaN with Most Frequent: {most_frequent_string}", "value": "most_frequent"},
        ]
        # Dropdown menu for column operations
        dropdown_menu = dbc.DropdownMenu(
            label="Options",
            children=[
                dbc.DropdownMenuItem("Change data type", header=True),
                dbc.DropdownMenuItem("Numeric", id={"type": "convert", "index": column_name, "to": "Numeric"}),
                dbc.DropdownMenuItem("String", id={"type": "convert", "index": column_name, "to": "String"}),
                
                dbc.DropdownMenuItem(divider=True),

                dbc.DropdownMenuItem("Data Cleaning", header=True),
                *[dbc.DropdownMenuItem(item["label"], id={"type": "clean", "index": column_name, "action": item["value"]}) for item in data_cleaning_options],
            ],
            className="m-1",
            right=True
        )
        # Return the visualization card for string columns
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div(column_name),
                    html.Div(f"Data type: {data_type}", style={'fontSize': '12px', 'color': 'grey'})
                ]),
                dropdown_menu  # Adding dropdown menu here for consistency
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '250px'}),
                html.P(f"NaN values: {nan_count}"),
                html.P(f"Most Frequent: {most_frequent_string}"),
                html.P(f"Least Frequent: {least_frequent_string}"),
                html.P(f"Unique Strings: {unique_count}")  # Display the number of unique strings
            ])
        ])
