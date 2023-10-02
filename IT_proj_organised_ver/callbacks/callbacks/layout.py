import base64
import io
import dash
import json
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
from app_instance import app

navbar = dbc.NavbarSimple(
    brand="PyExploratory",
    brand_href="/", 
    sticky="top",
)

layout = html.Div([
    navbar,
    dcc.Store(id='stored-data'), # store the uploaded data
    dbc.Container([
        dbc.Tabs([
            # Tab 1: Import, Export, DataFrame Display
            dbc.Tab(label='Data Management', children=[
                dbc.Row([
                    # Import File Type Dropdown
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
                            className='custom-dropdown'
                        ),
                    ], width=3),
                    # Upload Data Section
                    dbc.Col([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '40px',
                                'lineHeight': '40px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        html.Div(id='error-message', style={'color': 'red'}),
                    ], width=3),
                    # Export File Type Dropdown
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
                            className='custom-dropdown'
                        ),
                    ], width=3),
                    # Save Button and Download Component
                    dbc.Col([
                        html.Button('Save', id='save-button', className='custom-button'),
                        dcc.Download(id="download")
                    ], width=3)
                ]),
                # DataTable to Display Uploaded Data
                html.Div([
                    dash_table.DataTable(id='datatable-upload-container')
                ], style={'display': 'none'}),
                html.Div(id='output-data-upload'),
            ]),
            # Tab 2: Statistical Summary
            dbc.Tab(label='Statistical Summary', children=[
                html.Div(id='summary-output'),
            ]),
            # Tab 3: Visualisation Summary
            dbc.Tab(label='Visualisation', children=[
                # column selection
                dbc.Row([
                    # x axis column
                    dbc.Col([
                        html.Label('Select X-axis column:'),
                        dcc.Dropdown(
                            id='xaxis-viscolumn-dropdown',
                            options=[],
                            placeholder="Select a column for X-axis...",
                            value=None,
                            className='custom-dropdown'
                        )
                    ], width=4),
                    # y axis column
                    dbc.Col([
                        html.Label('Select Y-axis column:'),
                        dcc.Dropdown(
                            id='yaxis-viscolumn-dropdown',
                            options=[],
                            placeholder="Select a column for Y-axis...",
                            value=None,
                            className='custom-dropdown'
                        ),
                    ], width=4),
                    # graph type
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
                            value=None,
                            className='custom-dropdown'
                        ),
                    ], width=4)
                ], className='mb-4'),
                # Visualization
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='visualisation-graph')  # display the selected visualization
                    ])
                ], className='mb-4'),
                # Save Graph Button
                dbc.Row([
                    dbc.Col([
                        
                    ], width=3)
                ], className='mb-4'),
                html.Button('Save Graph', id='save-graph-button', className='custom-button'), 
                html.Div(id='saved-visgraphs-container')
                
            ]),
            # Tab 4: Analytics Summary
            dbc.Tab(label='Analytics', children=[
                # Data Analysis Choice
                dbc.Row([
                    dbc.Col([
                        html.Label("Choose Data Analysis Method:"),
                        dcc.Dropdown(
                            id='data-analysis-dropdown',
                            options=[
                                {'label': 'Clustering', 'value': 'clustering'},
                                {'label': 'Classification', 'value': 'classification'}
                            ],
                            placeholder="Select an analysis method...",
                            value=None,
                            className='custom-dropdown'
                        )
                    ], width=4)
                ], className='mb-4'),

                # Dynamic content placeholder
                html.Div(id='dynamic-content'),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='analysis-graph')
                    ])
                ], className='mb-4'),
                dbc.Row([
                    dbc.Col([
                        html.Button('Run Analysis', id='run-analysis-button', className='custom-button')
                    ], width=4)
                ], className='mb-4')
            ]),
            # Tab 5: State Summary
            dbc.Tab(label='State Summary', children=[
            
            ])
        ]),
    ])
])
