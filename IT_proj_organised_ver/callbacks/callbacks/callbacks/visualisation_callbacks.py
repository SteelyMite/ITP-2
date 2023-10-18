"""
File:           visualization_callbacks.py
Description:    This module manages the visualization aspects of the Dash application.
                It handles tasks such as updating dropdown options based on available 
                data columns, generating visualizations based on user-selected types,
                and saving visualizations for future reference.

Authors:        Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Last Updated:   2023-10-18
"""

# Library imports
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
from utils import parse_contents
from state_saving_func import *

# Callback to update dropdown options based on the uploaded data columns
@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    # Check if data is provided
    if not stored_data:
        return [], [], [], []
    # Convert data to DataFrame
    df = pd.DataFrame(stored_data)
    # Define the columns and options for the dropdowns
    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options


# Callback to generate and display visualizations based on user's choice of visualization type, X-axis, and Y-axis
@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('stored-data', 'data')
)
def update_graph(x_column, y_column, vis_type, data):
    add_callback_source_code(update_graph)
    # Check if required inputs are provided
    if not data or not x_column or not y_column or not vis_type:
        return {}
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    # Generate the plotly graph based on visualization type
    if vis_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif vis_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif vis_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif vis_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif vis_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif vis_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif vis_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif vis_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)
    else:
        return {}
    # Log user's visualization actions
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig

# Callback to save the currently displayed graph for future reference
@app.callback(
    Output('saved-visgraphs-container', 'children'),
    Input('save-graph-button', 'n_clicks'),
    State('visualisation-graph', 'figure'),
    State('saved-visgraphs-container', 'children')
)
def save_current_graph(n_clicks, current_figure, current_saved_graphs):
    add_callback_source_code(save_current_graph)
    # If no figure is present, do not update
    if not current_figure:
        raise dash.exceptions.PreventUpdate
    # Create a Dash graph component from the figure
    current_graph = dcc.Graph(figure=current_figure)
    # Log the save action
    log_user_action(f"Save Graph: current_graph = dcc.Graph(figure=current_figure)")
    # Append the current graph to the saved graphs container
    if not current_saved_graphs:
        current_saved_graphs = []
    current_saved_graphs.append(current_graph)
    log_user_action(f"Save Graph: current_saved_graphs.append(current_graph)")
    
    return current_saved_graphs
