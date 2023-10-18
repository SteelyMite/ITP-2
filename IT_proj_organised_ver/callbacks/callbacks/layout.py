"""
File:           layout.py
Description:    Main layout and components for the PyExploratory application.
Authors:        Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Date Updated:   2023-10-18 
"""
# Import required modules
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

# Define the navbar component
navbar = dbc.NavbarSimple(
    brand="PyExploratory",
    brand_href="/",
    sticky="top",
)

# Define the main layout of the application
layout = html.Div([
    navbar,  # Navbar at the top
    dcc.Store(id='stored-data'),  # Used to store uploaded data
    
    # Main container for the tabs
    dbc.Container([
        dbc.Tabs([
            # Tab 1: Data Management
            dbc.Tab(label='Data Management', children=[
                dbc.Row([
                    # Dropdown for choosing import file type
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
                    
                    # Component to upload data
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
                    
                    # Dropdown for choosing export file type
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
                    
                    # Save button and download component
                    dbc.Col([
                        html.Button('Save', id='save-button', className='custom-button'),
                        dcc.Download(id="download")
                    ], width=3)
                ]),
                
                # DataTable to display the uploaded data
                html.Div([
                    dash_table.DataTable(id='datatable-upload-container')
                ], style={'display': 'none'}),
                
                html.Div(id='output-data-upload'),
            ]),
            
            # Tab 2: Statistical Summary
            dbc.Tab(label='Statistical Summary', children=[
                html.Div(id='summary-output'),
            ]),
            
            # Tab 3: Visualization
            dbc.Tab(label='Visualisation', children=[
                dbc.Row([
                    # Dropdown for X-axis column selection
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
                    
                    # Dropdown for Y-axis column selection
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
                    
                    # Dropdown for visualization type selection
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
                                {'label': 'Area Plot', 'value': 'area'},
                                {'label': 'Violin Plot', 'value': 'violin'}
                            ],
                            placeholder="Select a type...",
                            value=None,
                            className='custom-dropdown'
                        ),
                    ], width=4)
                ], className='mb-4'),
                
                # Visualization display area
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='visualisation-graph')
                    ])
                ], className='mb-4'),
                
                # Save graph button
                dbc.Row([
                    dbc.Col([
                    ], width=3)
                ], className='mb-4'),
                
                html.Button('Save Graph', id='save-graph-button', className='custom-button'),
                html.Div(id='saved-visgraphs-container')
            ]),
            
            # Tab 4: Analytics
            dbc.Tab(label='Analytics', children=[
                dbc.Row([
                    # Dropdown for data analysis method selection
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
                    ], width=4),                
                ], className='mb-4'),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Store(id='input-dict-store', data=[]),
                        html.Button('Start', id='start-button', className='custom-button'),
                        html.Div(id='dynamic-input-div'),
                        html.Button('Perform', id='perform-button', className='custom-button'),
                        html.Div(id='perform-result-div', style={'color': 'green', 'margin-top': '10px'}),
                    ],)
                ], className='mb-4'),
            ]),
            
            # Tab 5: State Summary
            dbc.Tab(label='State Summary', children=[
                html.Ul(id='action-list'),
                html.Button('Export Commands to .py File', id='export-commands-button', className='custom-button'),
                
                # Interval component to update the state summary
                dcc.Interval(
                    id='update-interval',
                    interval=10 * 1000,  # Update every 10 seconds
                    n_intervals=0
                )
            ]),
        ]),
    ]),
])
