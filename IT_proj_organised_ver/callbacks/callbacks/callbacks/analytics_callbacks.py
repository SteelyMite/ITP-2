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

from app_instance import app
import Data_Analysis_Func_Parameters as func_params
import Data_Analysis_Methods as func_methods
from func_methods import clustering_KMeans, classification_SVM

# from cluster_class_callbacks import classification_SVM

@app.callback(
    Output('clustering-results', 'children'),
    Input('stored-data', 'data'),
    Input('selected-columns-dropdown', 'value'),
    Input('num-clusters-input', 'value'),
    prevent_initial_call=True
)
def randomMethod():
    func_methods.clustering_KMeans(inputData, selectedColumns,numClusters)
    return 


def clustering_KMeans(stored_data, selectedColumns, numClusters):
    if stored_data.empty:
        return []

    inputData = pd.DataFrame(stored_data)
    
    kmeans = KMeans(n_clusters=numClusters)
    cluster_assignments = kmeans.fit_predict(inputData[selectedColumns])
    cluster_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    statistics = html.Div([
        html.H4("Cluster Statistics:"),
        html.P(f"Number of Clusters (K): {numClusters}"),
        html.P(f"Cluster Centers:\n{cluster_centers}"),
        html.P(f"Inertia (Within-cluster Sum of Squares): {inertia}")
    ])
    # Create the cluster plot
    # fig = px.scatter(df, x=selectedColumns[0], y=selectedColumns[1], color="blue", title='K-means Clustering')
    fig = px.scatter(inputData[selectedColumns], x=selectedColumns[0], y=selectedColumns[1], color=cluster_assignments, title='K-means Clustering')
    print(fig)
    # Check fig data type
    print(type(fig))
    return [fig], statistics


@app.callback(
    Output('classification-results', 'children'),
    Input('stored-data', 'data'),
    Input('selected-columns-dropdown', 'value'),
    Input('target-column-dropdown', 'value'),
    prevent_initial_call=True
)
def classification_SVM(stored_data, selectedColumns, targetColumn):
    if stored_data.empty:
        return []

    inputData = pd.DataFrame(stored_data)
    fig = []
    #  split into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(inputData[selectedColumns], inputData[targetColumn], test_size=0.2, random_state=42)
    # create + train SVM Classifier 
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    # gen confusion matrix and class report
    cm = confusion_matrix(y_test, y_pred)
    
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # html report 
    statistics = html.Div([
        html.H4("Classification Statistics:"),
        html.P(f"Precision: {class_report['weighted avg']['precision']}"),
        html.P(f"Recall: {class_report['weighted avg']['recall']}"),
        html.P(f"F1-Score: {class_report['weighted avg']['f1-score']}"),
        html.P(f"Support: {class_report['weighted avg']['support']}")
      ])
    

    figure = generateConfusionMatrix(cm)

    fig.append(figure)

    return fig, statistics


def generateConfusionMatrix(cm):
    heatmap = go.Heatmap(
        z=cm,
        x=["Predicted 0", "Predicted 1"],
        y=["True 0", "True 1"],
        colorscale="Blues",
        colorbar=dict(title="Count"),
    )

    # matrix layout
    layout = go.Layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted labels"),
        yaxis=dict(title="True labels"),
    )
    return go.Figure(data=[heatmap], layout=layout)



@app.callback(
    Output('dynamic-content', 'children'),
    Input('data-analysis-dropdown', 'value'),
    prevent_initial_call=True
)
def update_dynamic_content(chosen_analysis):
    if chosen_analysis == 'clustering':
        return [

        ]
    elif chosen_analysis == 'classification':
        return [
        ]
    return []



@app.callback(
    [Output('analysis-graph', 'figure'), Output('analysis-statistics', 'children')],
    [Input('run-analysis-button', 'n_clicks')],
    [State('data-analysis-dropdown', 'value'),
     State('stored-data', 'data')],
    prevent_initial_call=True
)
def run_and_display_analysis(n_clicks, selected_analysis, stored_data):


    df = pd.DataFrame(stored_data)

    # cheeky hardcode, has columns named "Column One" and "Column Three"       NEEDS IMPROVEMENT NEEDS IMPROVEMENT NEEDS IMPROVEMENT 
    selectedColumns = ["Column One", "Column Three"]

    if not df.empty and selected_analysis == "clustering":
        plot_list, html_obj = clustering_KMeans(df, selectedColumns, numClusters=3)
        return plot_list[0], html_obj 

    elif not df.empty and selected_analysis == "classification":
        if "Label" in df.columns:
            plot_list, html_obj = classification_SVM(df, selectedColumns, "Label")
            return plot_list[0], html_obj

    return dash.no_update, "Please select a valid analysis type and ensure data is uploaded."


@app.callback(
    Output('dynamic-input-div', 'children'),
    [Input('start-button', 'n_clicks')],
    [State('input-dict-store', 'data'),
     State('stored-data', 'data')]  # Adjusted data reference
)
def generate_input(n_clicks, input_dicts, df_data):
    # Only generate inputs when the button is clicked
    if n_clicks is None or n_clicks == 0:
        return []
    
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


@app.callback(
    Output('perform-result-div', 'children'),
    Input('perform-button', 'n_clicks'),
    State('dynamic-input-div', 'children'),
    State('input-dict-store', 'data')
)
def perform_operation(n_clicks, dynamic_input_children, input_dicts):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # The following code block is mostly kept the same as yours with some comments for clarity
    for idx, input_dict in enumerate(input_dicts):
        input_component = dynamic_input_children[idx * 2 + 1]  # Note the adjusted index
        
        input_type = input_dict.get("Input", "")
        acceptable_value = input_dict.get("AcceptableValues", [])
        
        # Check for missing or None input
        if 'value' not in input_component['props'] or input_component['props']['value'] is None:
            return "Please ensure all inputs are provided before proceeding."
        
        if input_type == "column_selection":
            num_selected_columns = len(input_component['props']['value'])
            
            # Input validation based on the acceptable_value type and quantity
            if isinstance(acceptable_value, int) and num_selected_columns != acceptable_value:
                return f"Please select exactly {acceptable_value} columns."
            elif isinstance(acceptable_value, dict):
                min_columns = acceptable_value.get("min", 0)
                max_columns = acceptable_value.get("max", num_selected_columns)
                
                min_columns = num_selected_columns if min_columns is None else min_columns
                max_columns = num_selected_columns if max_columns is None else max_columns
                
                if not (min_columns <= num_selected_columns <= max_columns):
                    return f"Please select between {min_columns} and {max_columns} columns."
    
    # Desired operations and further application logic goes here
    # ...

    # Success message or further outputs are returned/generated here
    return "All requirements has been met, ready to perform!"
