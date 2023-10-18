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
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options



@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('stored-data', 'data')  # Fetch data from dcc.Store
)
def update_graph(x_column, y_column, vis_type, data):
    add_callback_source_code(update_graph)
    if not data or not x_column or not y_column or not vis_type:
        return {}

    df = pd.DataFrame(data)

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
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig

@app.callback(
    Output('saved-visgraphs-container', 'children'),
    Input('save-graph-button', 'n_clicks'),
    State('visualisation-graph', 'figure'),
    State('saved-visgraphs-container', 'children')
)
def save_current_graph(n_clicks, current_figure, current_saved_graphs):
    add_callback_source_code(save_current_graph)
    if not current_figure:
        raise dash.exceptions.PreventUpdate
    current_graph = dcc.Graph(figure=current_figure)
    log_user_action(f"Save Graph: current_graph = dcc.Graph(figure=current_figure)")
    if not current_saved_graphs:
        current_saved_graphs = []
    current_saved_graphs.append(current_graph)
    log_user_action(f"Save Graph: current_saved_graphs.append(current_graph)")
    return current_saved_graphs
