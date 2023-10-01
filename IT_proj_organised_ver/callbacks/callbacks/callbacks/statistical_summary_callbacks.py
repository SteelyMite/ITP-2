import dash
from dash import Input, Output, html, dash_table, dcc
import pandas as pd
import base64
import io
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import dash_bootstrap_components as dbc
import json

from app_instance import app
from utils import generate_column_summary_box


@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True)],
    [Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def handle_dropdown_actions(n_clicks_convert, n_clicks_clean, stored_data):
    # Get the triggering input (i.e., which dropdown item was clicked)
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Information about the clicked dropdown item
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    prop_info = json.loads(prop_id)

    df = pd.DataFrame(stored_data)
    column_name = prop_info['index']

    if prop_info['type'] == "convert":
        if prop_info['to'] == "Numeric":
            # Convert the column to numeric
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            
            # Check if there are any NaN values in the resulting series
            if temp_series.isna().any():
                # If there are NaN values, then the conversion is not practical
                raise dash.exceptions.PreventUpdate
            else:
                df[column_name] = temp_series
        else:
            df[column_name] = df[column_name].astype(str)
    
    elif prop_info['type'] == "clean":
        if prop_info['action'] == "min":
            df[column_name].fillna(df[column_name].min(), inplace=True)
        elif prop_info['action'] == "max":
            df[column_name].fillna(df[column_name].max(), inplace=True)
        elif prop_info['action'] == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
        elif prop_info['action'] == "zero":
            df[column_name].fillna(0, inplace=True)
        elif prop_info['action'] == "na_string":
            df[column_name].fillna('N/A', inplace=True)
        elif prop_info['action'] == "most_frequent":
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].empty else "N/A"
            df[column_name].fillna(most_frequent, inplace=True)
        # Handle normalization actions
        elif prop_info['action'] == "normalize_new":
            normalized_col_name = "normalized_" + column_name
            df[normalized_col_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
        elif prop_info['action'] == "normalize_replace":
            df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())

    return [df.to_dict('records')]

@app.callback(
    Output('output-data-upload', 'children', allow_duplicate=True),
    [Input('stored-data', 'modified_timestamp')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def update_table_from_store(ts, stored_data):
    if ts is None or not stored_data:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(stored_data)
    data_table = dash_table.DataTable(
        id='table',
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
        style_table={'overflowX': 'auto'},
        editable=True,
        page_size=20
    )
    return data_table