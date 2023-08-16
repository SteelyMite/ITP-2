import json
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import base64
import io
from app import app
import dash_bootstrap_components as dbc

layout =html.Div([
    dcc.Store(id='storage'),
    dcc.Store(id='graphs-storage', data={'graphs': []}),  

    # upload
    html.Div(id='upload-menu', style={'display': 'none'}),
         dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='file-type-dropdown',
                    options=[
                        {'label': 'CSV', 'value': 'csv'},
                        {'label': 'Excel', 'value': 'excel'},
                        {'label': 'JSON', 'value': 'json'}
                    ],
                    placeholder="Select file type...",
                    value='csv',
                    style={
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'width': '100%',
                        'height': '60px',
                        'borderWidth': '1px',
                        'borderRadius': '5px',
                        'margin': '10px'
                    }
                ),
                html.Div(id='upload-container'), 
                dash_table.DataTable(id='datatable-upload-container')
            ], width=2),
            dbc.Col([
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
            ], width=4),
            dbc.Col([
                dcc.Dropdown(
                    id='export-format-dropdown',
                    options=[
                        {'label': 'CSV', 'value': 'csv'},
                        {'label': 'Excel', 'value': 'xlsx'},
                        {'label': 'JSON', 'value': 'json'}
                    ],
                    value='csv',
                    clearable=False,
                    style={
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'width': '100%',
                        'height': '60px',
                        'borderWidth': '1px',
                        'borderRadius': '5px',
                        'margin': '10px'
                    }
                ),
            ], width=2),
            dbc.Col([
                html.Button('Save', id='save-button', style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'background': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'cursor': 'pointer',
                }),
                dcc.Download(id="download")
            ], width=4)    
        ]),
    # selection
    dbc.Row([
        dbc.Col([
            html.Label('Select column for visualization:'),
            dcc.Dropdown(
                id='visualization-column-dropdown',
                options=[],
                placeholder="Select a column...",
                value=None
            ),
        ], width=6),
        dbc.Col([
            html.Label('Select visualization type:'),
            dcc.Dropdown(
                id='visualization-type-dropdown',
                options=[
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                ],
                placeholder="Select a type...",
                value=None
            ),
        ], width=6)
    ], className='mb-4'),  #spacing

    # Visualization
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='visualization-graph')  # display the selected visualization
        ])
    ], className='mb-4'),

    # graphs
    html.Label('Choose type of graph to add:', className='mb-2'),
    dcc.Dropdown(
        id='graph-selector',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Line Plot', 'value': 'line'},
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Pie Chart', 'value': 'pie'},
            {'label': 'Histogram', 'value': 'histogram'},
            {'label': 'Box Plot', 'value': 'box'},
            {'label': '3D Scatter Plot', 'value': '3dscatter'},
            {'label': 'Area Plot', 'value': 'area'},
            {'label': 'Violin Plot', 'value': 'violin'}
        ],
        style={'width': '50%'}
    ),
    html.Button('Add', id='add-button', className='mt-3 mb-4'),  #spacing
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
        fig = px.scatter(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE", hover_name="STATE")
    elif graph_type == 'line':
        fig = px.line(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE")
    elif graph_type == 'bar':
        fig = px.bar(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE")
    elif graph_type == 'pie':
        fig = px.pie(df, names="STATE", values="Cesarean Delivery Rate")
    elif graph_type == 'histogram':
        fig = px.histogram(df, x="Cesarean Delivery Rate", color="STATE") 
    elif graph_type == 'box':
        fig = px.box(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE") 
    elif graph_type == '3dscatter':
        fig = px.scatter_3d(df, x="YEAR", y="Cesarean Delivery Rate", z="Drug Overdose Mortality per 100,000", color="STATE") 
    elif graph_type == 'area':
        fig = px.area(df, x="YEAR", y="Cesarean Delivery Rate", color="STATE")
    elif graph_type == 'violin':
        fig = px.violin(df, y="Cesarean Delivery Rate", color="STATE")

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
    [Output('upload-menu', 'children'),
     Output('upload-menu', 'style')],
    Input('upload-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_upload_menu(n_clicks):
 if n_clicks is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}

@app.callback(
    Output('visualization-column-dropdown', 'options'),
    Input('storage', 'data'),
    prevent_initial_call=True
)
def set_column_options(data):
    if data:
        df = pd.read_json(data, orient='split')
        options = [{'label': col, 'value': col} for col in df.columns]
        return options
    return []

@app.callback(
    Output('visualization-graph', 'figure'),
    Input('visualization-column-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('storage', 'data'),
    prevent_initial_call=True
)
def update_visualization(selected_column, visualization_type, data):
    if data is None or selected_column is None or visualization_type is None:
        return dash.no_update

    df = pd.read_json(data, orient='split')

    if visualization_type == 'bar':
        value_counts = df[selected_column].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': selected_column, 'y': 'count'})

    elif visualization_type == 'pie':
        value_counts = df[selected_column].value_counts()
        fig = px.pie(names=value_counts.index, values=value_counts.values)

    return fig
