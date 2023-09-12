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

selected_graphs_list = []


df = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [5, 4, 3, 2, 1]
})



layout = html.Div([
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
            html.Label('Select X-axis column:'),
            dcc.Dropdown(
                id='xaxis-column-dropdown',
                options=[],
                placeholder="Select a column for X-axis...",
                value=None
            ),
        ], width=4),
        dbc.Col([
            html.Label('Select Y-axis column:'),
            dcc.Dropdown(
                id='yaxis-column-dropdown',
                options=[],
                placeholder="Select a column for Y-axis...",
                value=None
            ),
        ], width=4),
        dbc.Col([
            html.Label('Select visualization type:'),
            dcc.Dropdown(
                id='visualization-type-dropdown',
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
                placeholder="Select a type...",
                value=None
            ),
        ], width=6)
    ], className='mb-4'),  # spacing
    # Visualization
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='visualization-graph')  # display the selected visualization
        ])
    ], className='mb-4'),



    dbc.Row([
        dbc.Col([
        ])
    ], className='mb-4'),
    html.Button('Save Graph', id='save-graph-button', className='mt-3 mb-4'), 
    html.Div(id='saved-graphs-container')
        
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
    [Output('upload-menu', 'children'),
     Output('upload-menu', 'style')],
    Input('upload-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_upload_menu(n_clicks):
 if n_clicks is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}

@app.callback(
    [Output('xaxis-column-dropdown', 'options'),
     Output('yaxis-column-dropdown', 'options')],
    Input('storage', 'data'),
    prevent_initial_call=True
)
def set_column_options(data):
    if data:
        df = pd.read_json(data, orient='split')
        options = [{'label': col, 'value': col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    Output('visualization-graph', 'figure'),
    [Input('xaxis-column-dropdown', 'value'),
     Input('yaxis-column-dropdown', 'value'),
     Input('visualization-type-dropdown', 'value')],
    State('storage', 'data'),
    prevent_initial_call=True
)
def update_visualization(x_column, y_column, visualization_type, data):
    if data is None or x_column is None or y_column is None or visualization_type is None:
        return dash.no_update

    df = pd.read_json(data, orient='split')

    if visualization_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif visualization_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif visualization_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif visualization_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif visualization_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif visualization_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif visualization_type == '3dscatter':
        fig = px.scatter_3d(df, x=x_column, y=y_column)
    elif visualization_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif visualization_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)

    return fig



@app.callback(
    Output('graph-type-table', 'children'),
    Input('add-to-list-button', 'n_clicks'),
    State('graph-selector', 'value'),
    prevent_initial_call=True
)
def update_selected_graph_table(n_clicks, selected_graph):
    global selected_graphs_list

    selected_graphs_list.append(selected_graph)

    df = pd.DataFrame(selected_graphs_list, columns=['Selected Graph Type'])

    table = dash_table.DataTable(
        columns=[{'name': i, 'id': i} for i in df.columns],
        data=df.to_dict('records')
    )

    return table



@app.callback(
    Output('saved-graphs-container', 'children'),
    [Input('save-graph-button', 'n_clicks')],
    [State('visualization-graph', 'figure'),
     State('saved-graphs-container', 'children')]
)
def save_current_graph(n_clicks, current_figure, current_saved_graphs):
    if not current_figure:
        raise dash.exceptions.PreventUpdate
    current_graph = dcc.Graph(figure=current_figure)

    if not current_saved_graphs:
        current_saved_graphs = []

    current_saved_graphs.append(current_graph)

    return current_saved_graphs
