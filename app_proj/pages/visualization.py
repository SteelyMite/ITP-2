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
        multiple=False  # Allow multiple files to be uploaded
    ),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Line Chart', 'value': 'line'},
            # Add more options here
        ],
        value='scatter'
    ),
    dcc.Graph(id='graph')
])

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
    Output('graph', 'figure'),
    Input('dropdown', 'value'),
    Input('storage', 'data')
)
def update_graph(dropdown_value, data):
    if data is None:  
        return dash.no_update
    df = pd.read_json(data, orient='split')
    if dropdown_value == 'scatter':
        fig = px.scatter(df, x="YEAR", y="Cesarean Delivery Rate", size="Drug Overdose Mortality per 100,000", color="STATE", hover_name="STATE")
    elif dropdown_value == 'line':
        fig = px.line(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE")
    return fig


