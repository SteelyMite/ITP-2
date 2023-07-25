from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from app import app
import pandas as pd

# Define global variable to store the dataframe
df = pd.DataFrame()

layout = html.Div([
    dcc.Dropdown(
        id='dropdown',
        options=[]
    ),
    dcc.Graph(id='graph')
])

@app.callback(
    Output('dropdown', 'options'),
    Input('table', 'data'),
    Input('table', 'columns'))
def update_dropdown(data, columns):
    global df
    if data is not None and columns is not None:
        df = pd.DataFrame(data, columns=[c['name'] for c in columns])
        options = [{'label': i, 'value': i} for i in df.columns]
        return options
    return []

@app.callback(
    Output('graph', 'figure'),
    Input('dropdown', 'value'))
def update_graph(column):
    if column is not None:
        figure = px.histogram(df, column)
        return figure
    return {}
