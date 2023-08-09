import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import base64
import io
from app import app

layout = html.Div([
    dcc.Store(id='storage'),
    dcc.Store(id='graphs-storage', data={'graphs': []}), # To keep track of the existing graphs
    html.Button('Click to Upload', id='upload-button'),
    html.Div(id='upload-menu', style={'display': 'none'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    dcc.Dropdown(
        id='graph-selector',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Line Plot', 'value': 'line'}
        ],
        style={'width': '50%'}
    ),
    html.Button('Add', id='add-button'),
    html.Div(id='graphs-container')
])

@app.callback(
    Output('graphs-container', 'children'),
    Input('add-button', 'n_clicks'),
    State('graph-selector', 'value'),
    State('storage', 'data'),
    State('graphs-storage', 'data'),
    prevent_initial_call=True
)
def add_graph(n_clicks, graph_type, data, existing_graphs):
    if data is None or graph_type is None:
        return dash.no_update

    df = pd.read_json(data, orient='split')

    if graph_type == 'scatter':
        fig = px.scatter(df, x="YEAR", y="Cesarean Delivery Rate", size="Drug Overdose Mortality per 100,000", color="STATE", hover_name="STATE")
    elif graph_type == 'line':
        fig = px.line(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE")
    elif graph_type == 'bar':
        fig = px.bar(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE")
    elif graph_type == 'pie':
        fig = px.pie(df, names="STATE", values="Cesarean Delivery Rate")

    existing_graphs['graphs'].append(html.Div(dcc.Graph(figure=fig)))

    return existing_graphs['graphs']

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

@app.callback(
    Output('storage', 'data'),
    Input('upload-data', 'contents'),
)
def update_output(contents):
    if contents is None:
        return dash.no_update
    df = parse_contents(contents)
    return df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('button-menu', 'children'),
    Output('button-menu', 'style'),
    Input('upload-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_button_menu(n_clicks):
    if n_clicks is not None:
        return html.Div([
            html.Button('1', id='button-1'),
            html.Button('2', id='button-2'),
            html.Button('3', id='button-3'),
        ]), {'display': 'block'}
    return dash.no_update, dash.no_update

@app.callback(
    [Output('button-excel', 'style'),
     Output('button-csv', 'style'),
     Output('button-csv-non-delimited', 'style')],
    Input('upload-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_upload_menu(n_clicks):
    if n_clicks is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
