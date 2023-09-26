from dash import Input, Output, html, dash_table, dcc
import pandas as pd
import base64
import io
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import dash_bootstrap_components as dbc
import json


from app_instance import app

@app.callback(
    Output('dynamic-content', 'children'),
    Input('data-analysis-dropdown', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def populate_dynamic_content(analysis_choice, stored_data):
    if not stored_data:
        return html.Div()

    df = pd.DataFrame(stored_data)
    numeric_columns = [{'label': col, 'value': col} for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    all_columns = [{'label': col, 'value': col} for col in df.columns]

    if analysis_choice == 'classification':
        return [
            dbc.Row([
                dbc.Col([
                    html.Label("Select Numeric Columns for Classification:"),
                    dcc.Dropdown(
                        id='classification-column-dropdown',
                        options=numeric_columns,
                        multi=True,
                        className='custom-dropdown'
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Select Label Column:"),
                    dcc.Dropdown(
                        id='label-column-dropdown',
                        options=all_columns,
                        className='custom-dropdown'
                    )
                ], width=4)
            ], className='mb-4')
        ]

    elif analysis_choice == 'clustering':
        return [
            dbc.Row([
                dbc.Col([
                    html.Label("Select Numeric Columns for Clustering:"),
                    dcc.Dropdown(
                        id='clustering-columns-dropdown',
                        options=numeric_columns,
                        multi=True,
                        className='custom-dropdown'
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Number of Clusters:"),
                    dcc.Input(id='num-clusters-input', type='number', min=2, step=1, value=2)
                ], width=4)
            ], className='mb-4')
        ]

    return html.Div()
