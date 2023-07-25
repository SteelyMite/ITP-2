import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from app import app

layout = html.Div([
    dcc.Store(id='storage'),
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

@app.callback(
    Output('graph', 'figure'),
    Input('dropdown', 'value'),
    Input('storage', 'data')
)
def update_graph(dropdown_value, data):
    if data is None:  # Add this line to handle the case where data is None
        return dash.no_update
    # Load the DataFrame from the 'storage' component (convert the JSON string back into a DataFrame)
    df = pd.read_json(data, orient='split')
    if dropdown_value == 'scatter':
        fig = px.scatter(df, x="gdp per capita", y="life expectancy", size="pop", color="continent", hover_name="country", log_x=True, size_max=60)
    elif dropdown_value == 'line':
        fig = px.line(df, x="gdp per capita", y="life expectancy", color="continent")
    return fig

