from dash import Input, Output, html, dash_table, dcc
import pandas as pd
import base64
import io
from dash.dependencies import State, ALL
import plotly.express as px
import dash_bootstrap_components as dbc
import json
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import dash
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.subplots as sp
import plotly.figure_factory as ff
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

from data_analysis_func import clustering_KMeans, classification_SVM, generateConfusionMatrix, generateAUC
from app_instance import app
from state_saving_func import *

# from cluster_class_callbacks import classification_SVM


@app.callback(
    Output('dynamic-input-div', 'children'),
    [Input('start-button', 'n_clicks')],
    [State('input-dict-store', 'data'),
     State('stored-data', 'data')]  # Adjusted data reference
)
def generate_input(n_clicks, input_dicts, df_data):
    add_callback_source_code(generate_input)
    # Only generate inputs when the button is clicked
    if n_clicks is None or n_clicks == 0:
        return []
    if input_dicts is None:
        # Handle None case appropriately
        return html.P("No input configurations available")
    df = pd.DataFrame(df_data)  # Adjusted data conversion
    
    components = []
    
    for input_dict in input_dicts:
        text_display = input_dict.get("Text", "")
        input_type = input_dict.get("Input", "")
        acceptable_value = input_dict.get("AcceptableValues", [])
        data_type = input_dict.get("DataType", "")
        
        components.append(html.H5(text_display))
        
        if input_type == "dropdown":
            components.append(dcc.Dropdown(
                options=[{'label': val, 'value': val} for val in acceptable_value],
                value=acceptable_value[0] if acceptable_value else None,
                className='custom-dropdown'
            ))
            
        elif input_type == "number_selection":
            min_val = acceptable_value.get("min")
            max_val = acceptable_value.get("max")
            
            components.append(dcc.Input(
                type='number',
                min=min_val,
                max=max_val,
                placeholder=f"Enter a number{' ≥ '+str(min_val) if min_val is not None else ''}{' ≤ '+str(max_val) if max_val is not None else ''}"
            ))
            
        elif input_type == "column_selection":
            if data_type == "numeric":
                options = [{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns]
            elif data_type == "string":
                options = [{'label': col, 'value': col} for col in df.select_dtypes(include='object').columns]
            else:  # both
                options = [{'label': col, 'value': col} for col in df.columns]
            
            components.append(dcc.Dropdown(
                options=options,
                multi= True,
                className='custom-dropdown'
            ))
           
        else:
            components.append(html.P("Invalid input type"))
        
        
    return components

CLUSTERING_INPUT_MAPPING = {
    'selected_columns': 1,  # Index of the dcc.Dropdown component in 'dynamic-input-div'
    'num_clusters': 3       # Index of the dcc.Input component in 'dynamic-input-div'
}

CLASSIFICATION_INPUT_MAPPING = {
    'features_columns': 1,  # Index of the dcc.Dropdown for selecting feature columns
    'target_column': 3,     # Index of the dcc.Dropdown for selecting the target column
    'kernel_type': 5        # Index of the dcc.Dropdown for selecting kernel type
}
@app.callback(
    Output('perform-result-div', 'children'),
    Input('perform-button', 'n_clicks'),
    State('dynamic-input-div', 'children'),
    State('input-dict-store', 'data'),
    State('stored-data', 'data'),  
    State('data-analysis-dropdown', 'value')  
)
def perform_operation(n_clicks, dynamic_input_children, input_dicts, stored_data, method):
    add_callback_source_code(perform_operation)
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Validation
    for idx, input_dict in enumerate(input_dicts):
        input_component = dynamic_input_children[idx * 2 + 1]
        input_type = input_dict.get("Input", "")
        acceptable_value = input_dict.get("AcceptableValues", [])
        
        if 'value' not in input_component['props'] or input_component['props']['value'] is None:
            return "Please ensure all inputs are provided before proceeding."
        
        if input_type == "column_selection":
            num_selected_columns = len(input_component['props']['value'])
            if isinstance(acceptable_value, int) and num_selected_columns != acceptable_value:
                return f"Please select exactly {acceptable_value} columns."
            elif isinstance(acceptable_value, dict):
                min_columns = acceptable_value.get("min", 0)
                max_columns = acceptable_value.get("max", num_selected_columns)
                min_columns = num_selected_columns if min_columns is None else min_columns
                max_columns = num_selected_columns if max_columns is None else max_columns
                if not (min_columns <= num_selected_columns <= max_columns):
                    return f"Please ensure all input meets the criteria."
    
    # Analysis
    input_data = pd.DataFrame.from_dict(stored_data)
    # Perform Operations
    if method == 'clustering':
        # Extract relevant inputs
        selected_columns = dynamic_input_children[CLUSTERING_INPUT_MAPPING['selected_columns']]['props']['value']
        num_clusters = dynamic_input_children[CLUSTERING_INPUT_MAPPING['num_clusters']]['props']['value']
        # Call clustering function and get results
        fig, stats = clustering_KMeans(input_data, selected_columns, num_clusters)
        print('called clustering')
        log_user_action(f"Data Analysis: Clustering Performed")
        # Combine Plotly figures and HTML components for display
        children_components = [
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_item), width=6) for fig_item in fig
            ]),
            stats
        ]
        return children_components
    
    elif method == 'classification':
         # Extract relevant inputs
        features_columns = dynamic_input_children[CLASSIFICATION_INPUT_MAPPING['features_columns']]['props']['value']
        target_column = dynamic_input_children[CLASSIFICATION_INPUT_MAPPING['target_column']]['props']['value']
        kernel_type = dynamic_input_children[CLASSIFICATION_INPUT_MAPPING['kernel_type']]['props']['value']
        print(features_columns)
        print(target_column)
        print(kernel_type)
        # Call classification function and get results
        figs, stats = classification_SVM(input_data, features_columns, target_column, kernel_type)
        # Combine Plotly figures and HTML components for display
        print('called classification')
        log_user_action(f"Data Analysis: Classification Performed")
        children_components = [
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_item), width=6) for fig_item in figs
            ]),
            stats
        ]
        return children_components
    else:
        return "Invalid method selected!"