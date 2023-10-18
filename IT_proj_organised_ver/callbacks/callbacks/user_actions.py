import base64
import io
import pandas as pd
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import inspect
import dash
import json
import plotly.subplots as sp
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from dash import dcc
from dash import html
from app_instance import app
from layout import layout
from callbacks import state_saving_callbacks, data_management_callbacks, statistical_summary_callbacks
from utils import parse_contents
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import inspect
app.layout = layout
if __name__ == '__main__':
    app.run_server(debug=True)app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.YETI])
server = app.server
navbar = dbc.NavbarSimple(
    brand='PyExploratory',
    brand_href='/',
    sticky='top',
)
layout = html.Div([
    navbar,
    dcc.Store(id='stored-data'),  # store the uploaded data
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
                            placeholder='Select file type...',
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
                        dcc.Download(id='download')
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
                            placeholder='Select a column for X-axis...',
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
                            placeholder='Select a column for Y-axis...',
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
                                {'label': 'Area Plot', 'value': 'area'},
                                {'label': 'Violin Plot', 'value': 'violin'}
                            ],
                            placeholder='Select a type...',
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
                        html.Label('Choose Data Analysis Method:'),
                        dcc.Dropdown(
                            id='data-analysis-dropdown',
                            options=[
                                {'label': 'Clustering', 'value': 'clustering'},
                                {'label': 'Classification', 'value': 'classification'}
                            ],
                            placeholder='Select an analysis method...',
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
                    ], width=10)
                ], className='mb-4'),
            ]),
            dbc.Tab(label='State Summary', children=[
                html.Ul(id='action-list'),
                html.Button('Export Commands to .py File', id='export-commands-button', className='mt-3 mb-4', style={
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'width': '100%',
                    'height': '40px',
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin': '10px'
                }),
                dcc.Interval(
                    id='update-interval',
                    interval=10 * 1000,  # Update every 10 seconds (adjust as needed)
                    n_intervals=0
                )
            ]),
        ]),
    ]),
])

def parse_contents(contents, file_type):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if file_type == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif file_type == 'excel':
            df = pd.read_excel(io.BytesIO(decoded))
        elif file_type == 'json':
            df = pd.read_json(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

def generate_column_summary_box(df, column_name):
    # Calculate statistics
    nan_count = df[column_name].isna().sum()
    if pd.api.types.is_numeric_dtype(df[column_name]):
        # Generate histogram for numeric columns
        data_type = 'Numeric'
        fig = px.histogram(df, x=column_name, nbins=5)  # Here, 5 bins are used for simplicity; adjust as needed.
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        mean_val = df[column_name].mean()
        data_cleaning_options = [
            {'label': 'Replace NaN with Min', 'value': 'min'},
            {'label': 'Replace NaN with Max', 'value': 'max'},
            {'label': 'Replace NaN with Mean', 'value': 'mean'},
            {'label': 'Replace NaN with Zero', 'value': 'zero'}
        ]
        normalization_options = [
            {'label': 'Normalize (New Column)', 'value': 'normalize_new'},
            {'label': 'Normalize (Replace)', 'value': 'normalize_replace'},
        ]
        dropdown_menu = dbc.DropdownMenu(
            label='Options',
            children=[
                dbc.DropdownMenuItem('Change data type', header=True),
                dbc.DropdownMenuItem('Numeric', id={'type': 'convert', 'index': column_name, 'to': 'Numeric'}),
                dbc.DropdownMenuItem('String', id={'type': 'convert', 'index': column_name, 'to': 'String'}),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Data Cleaning', header=True),
                *[dbc.DropdownMenuItem(item['label'], id={'type': 'clean', 'index': column_name, 'action': item['value']}) for item in data_cleaning_options],
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Normalization', header=True),
                *[dbc.DropdownMenuItem(item['label'], id={'type': 'clean', 'index': column_name, 'action': item['value']}) for item in normalization_options],
            ],
            className='m-1',
            right=True
        )
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div(column_name),
                    html.Div(f'Data type: {data_type}', style={'fontSize': '12px', 'color': 'grey'})
                ]),
                dropdown_menu
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '250px'}),
                html.P(f'NaN values: {nan_count}'),
                html.P(f'Min: {min_val}'),
                html.P(f'Max: {max_val}'),
                html.P(f'Mean: {mean_val}')
            ])
        ])
    else:
        # For non-numeric columns, show a bar chart of top X most frequent values
        data_type = 'String'
        unique_count = df[column_name].nunique()  # Calculate the number of unique strings
        top_values = df[column_name].value_counts().head(10)
        fig = px.bar(top_values, x=top_values.index, y=top_values.values, labels={'x': column_name, 'y': 'Count'})
        value_counts = df[column_name].value_counts()
        most_frequent_string = value_counts.index[0] if not value_counts.empty else 'N/A'
        least_frequent_string = value_counts.index[-1] if not value_counts.empty else 'N/A'
        data_cleaning_options = [
            {'label': 'Replace NaN with N/A', 'value': 'na_string'},
            {'label': f'Replace NaN with Most Frequent: {most_frequent_string}', 'value': 'most_frequent'},
        ]
        dropdown_menu = dbc.DropdownMenu(
            label='Options',
            children=[
                dbc.DropdownMenuItem('Change data type', header=True),
                dbc.DropdownMenuItem('Numeric', id={'type': 'convert', 'index': column_name, 'to': 'Numeric'}),
                dbc.DropdownMenuItem('String', id={'type': 'convert', 'index': column_name, 'to': 'String'}),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Data Cleaning', header=True),
                *[dbc.DropdownMenuItem(item['label'], id={'type': 'clean', 'index': column_name, 'action': item['value']}) for item in data_cleaning_options],
            ],
            className='m-1',
            right=True
        )
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div(column_name),
                    html.Div(f'Data type: {data_type}', style={'fontSize': '12px', 'color': 'grey'})
                ]),
                dropdown_menu  # Adding dropdown menu here for consistency
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '250px'}),
                html.P(f'NaN values: {nan_count}'),
                html.P(f'Most Frequent: {most_frequent_string}'),
                html.P(f'Least Frequent: {least_frequent_string}'),
                html.P(f'Unique Strings: {unique_count}')  # Display the number of unique strings
            ])
        ])

user_actions = []
script_list = []

def get_function_source_code(function):
    source_code = inspect.getsource(function)
    return source_code

def add_callback_source_code(callback_func):
    source_code = inspect.getsource(callback_func)
    script_list.append(source_code)

def log_user_action(action, filename=None):
    if filename:
        user_actions.append(f'{action} ({filename})')
    else:
        user_actions.append(action)
def clustering_KMeans(inputData, selectedColumns, numClusters):
    if inputData.empty:
        return []

    fig = []
    try:
        kmeans = KMeans(n_clusters=numClusters)
        selectedData = inputData[selectedColumns]
        cluster_assignments = kmeans.fit_predict(selectedData)
        #? Create data analysis HTML report
        cluster_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_

        statistics = html.Div([
            html.H4('Cluster Statistics:'),
            html.P(f'Number of Clusters (K): {numClusters}'),
            html.P(f'Cluster Centers:\n{cluster_centers}'),
            html.P(f'Inertia (Within-cluster Sum of Squares): {inertia}'),
        ])
        #? Generate Plot if 2D, Else return Scatter Matrix and Cluster Profile
        if(len(selectedColumns)==2):
            scatter_plot = px.scatter(selectedData, x=selectedColumns[0], y=selectedColumns[1], color=cluster_assignments, title='K-means Clustering')
            scatter_plot.update_layout(title='K-means Clustering - Scatter Plot')
            fig.append(scatter_plot)
        else:
            pair_plot = px.scatter_matrix(selectedData, dimensions=selectedColumns, color=cluster_assignments)
            pair_plot.update_layout(title='K-means Clustering - Pair Plots')
            cluster_profile = pd.DataFrame(cluster_centers, columns=selectedData.columns)
            bar_plot = px.bar(cluster_profile)
            bar_plot.update_layout(title='K-means Clustering - Cluster Profile')
            fig.append(bar_plot)
            fig.append(pair_plot)
        return fig, statistics
    except ValueError as e:
        if 'Input X contains NaN' in str(e):
            print('Data contains NaN values. Please clean the data before analyzing.')
        else:
            print(f'An error occurred: {str(e)}')
    # You can return an informative message instead of a figure,
    # so you could render this message in a dedicated Div in your Dash app.
    except_msg = 'An error occurred during clustering. Please ensure data is clean and try again.'
    return except_msg


def classification_SVM(inputData, selectedColumns, targetColumn, kernel):
    fig = []
    print('1')
    selectedData = inputData[selectedColumns]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(selectedData, inputData[targetColumn], test_size=0.2, random_state=42)
    # Create and train SVM Classifier 
    y_train = y_train.values.ravel()
    print('1.5')
    y_test = y_test.values.ravel()
    print('2')
    classifier = svm.SVC(kernel=kernel) #!Make sure to update kernel type
    print('3')
    classifier.fit(X_train, y_train)
    print('4')
    # Predict labels on the test data
    y_pred = classifier.predict(X_test)
    print('5')
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print('6')
    # fig.append(cm)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print('7')
    # Create data analysis HTML report 
    statistics = html.Div([
        html.H4('Classification Statistics:'),
        html.P(f'Precision: {class_report['weighted avg']['precision']}'),
        html.P(f'Recall: {class_report['weighted avg']['recall']}'),
        html.P(f'F1-Score: {class_report['weighted avg']['f1-score']}'),
        html.P(f'Support: {class_report['weighted avg']['support']}')
      ])
        # Create the confusion matrix plot
    print('8')
    figure = generateConfusionMatrix(cm, class_labels=classifier.classes_)
    print('9')
    figure.update_layout(title='SVM Classification - Confusion Matrix')
    print('10')
    fig.append(figure)
    print('11')
    print(len(classifier.classes_)
    #return fig, statistics
    if len(classifier.classes_) == 2:
        y_pred_bin = label_binarize(y_pred, classes=classifier.classes_)
        y_test_bin = label_binarize(y_test, classes=classifier.classes_)
        AUC_Plot =  generateAUC(y_test_bin, y_pred_bin)
        AUC_Plot.update_layout(title='SVM Classification - Area Under Curve Plot')
        print('11.5')
        fig.append(AUC_Plot)
    print('12')
    return fig, statistics


def generateAUC(y_test, y_pred):
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Create ROC curve plot using Plotly Express
    fig = px.line(x=fpr, y=tpr, title=f'ROC curve (AUC = {roc_auc:.2f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    return fig


def generateConfusionMatrix(cm, class_labels):
    # Create a trace for the heatmap
    heatmap = go.Heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        colorscale='Blues',
        colorbar=dict(title='Count'),
    )
    # Define the layout for the confusion matrix plot
    layout = go.Layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted labels'),
        yaxis=dict(title='True labels'),
    )
    return go.Figure(data=[heatmap], layout=layout)
CLUSTERING_INPUT_MAPPING = {
    'selected_columns': 1,  # Index of the dcc.Dropdown component in 'dynamic-input-div'
    'num_clusters': 3       # Index of the dcc.Input component in 'dynamic-input-div'
}

CLASSIFICATION_INPUT_MAPPING = {
    'features_columns': 1,  # Index of the dcc.Dropdown for selecting feature columns
    'target_column': 3,     # Index of the dcc.Dropdown for selecting the target column
    'kernel_type': 5        # Index of the dcc.Dropdown for selecting kernel type
}@app.callback(
    Output('input-dict-store', 'data'),
    [Input('data-analysis-dropdown', 'value')]
)
def update_input_dict_store(selected_analysis_method):
    add_callback_source_code(update_input_dict_store)
    if selected_analysis_method == 'clustering':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select columns:",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "number_selection",
                "DataType": "numeric",
                "Text": "Number of Clusters:",
                "AcceptableValues": {"min": 1, "max": None}
            }
        ]
    elif selected_analysis_method == 'classification':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            }, 
            {
                "Input": "column_selection",
                "DataType": "string",
                "Text": "Select target column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "dropdown",
                "DataType": "string",
                "Text": "Select Kernel Type:",
                "AcceptableValues": ["linear", "poly", "rbf", "sigmoid"]
            }
        ]
    else:
        # Handle other cases or invalid selections appropriately
        return []

@app.callback(
    Output('saved-visgraphs-container', 'children'),
    Input('save-graph-button', 'n_clicks'),
    State('visualisation-graph', 'figure'),
    State('saved-visgraphs-container', 'children')
)
def save_current_graph(n_clicks, current_figure, current_saved_graphs):
    add_callback_source_code(save_current_graph)
    if not current_figure:
        raise dash.exceptions.PreventUpdate
    current_graph = dcc.Graph(figure=current_figure)
    log_user_action(f"Save Graph: current_graph = dcc.Graph(figure=current_figure)")
    if not current_saved_graphs:
        current_saved_graphs = []
    current_saved_graphs.append(current_graph)
    log_user_action(f"Save Graph: current_saved_graphs.append(current_graph)")
    return current_saved_graphs

@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('stored-data', 'data')  # Fetch data from dcc.Store
)
def update_graph(x_column, y_column, vis_type, data):
    add_callback_source_code(update_graph)
    if not data or not x_column or not y_column or not vis_type:
        return {}

    df = pd.DataFrame(data)

    if vis_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif vis_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif vis_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif vis_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif vis_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif vis_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif vis_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif vis_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)
    else:
        return {}
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig

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

@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('stored-data', 'data')  # Fetch data from dcc.Store
)
def update_graph(x_column, y_column, vis_type, data):
    add_callback_source_code(update_graph)
    if not data or not x_column or not y_column or not vis_type:
        return {}

    df = pd.DataFrame(data)

    if vis_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif vis_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif vis_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif vis_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif vis_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif vis_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif vis_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif vis_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)
    else:
        return {}
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig

@app.callback(
    Output('input-dict-store', 'data'),
    [Input('data-analysis-dropdown', 'value')]
)
def update_input_dict_store(selected_analysis_method):
    add_callback_source_code(update_input_dict_store)
    if selected_analysis_method == 'clustering':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select columns:",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "number_selection",
                "DataType": "numeric",
                "Text": "Number of Clusters:",
                "AcceptableValues": {"min": 1, "max": None}
            }
        ]
    elif selected_analysis_method == 'classification':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            }, 
            {
                "Input": "column_selection",
                "DataType": "string",
                "Text": "Select target column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "dropdown",
                "DataType": "string",
                "Text": "Select Kernel Type:",
                "AcceptableValues": ["linear", "poly", "rbf", "sigmoid"]
            }
        ]
    else:
        # Handle other cases or invalid selections appropriately
        return []

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

@app.callback(
    Output('saved-visgraphs-container', 'children'),
    Input('save-graph-button', 'n_clicks'),
    State('visualisation-graph', 'figure'),
    State('saved-visgraphs-container', 'children')
)
def save_current_graph(n_clicks, current_figure, current_saved_graphs):
    add_callback_source_code(save_current_graph)
    if not current_figure:
        raise dash.exceptions.PreventUpdate
    current_graph = dcc.Graph(figure=current_figure)
    log_user_action(f"Save Graph: current_graph = dcc.Graph(figure=current_figure)")
    if not current_saved_graphs:
        current_saved_graphs = []
    current_saved_graphs.append(current_graph)
    log_user_action(f"Save Graph: current_saved_graphs.append(current_graph)")
    return current_saved_graphs

@app.callback(
    [Output('output-data-upload', 'children'), 
     Output('error-message', 'children'), 
     Output('stored-data', 'data', allow_duplicate=True)],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('file-type-dropdown', 'value'),
    prevent_initial_call=True
)
def update_output(contents, filename, file_type):
    add_callback_source_code(update_output)
    if contents is None:
        raise dash.exceptions.PreventUpdate

    try:
        df = parse_contents(contents, file_type)
        command = f"Uploaded file: {filename}"  # Include the filename in the command
        log_user_action(command)
        data_table = dash_table.DataTable(
            id='table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
            style_table={'overflowX': 'auto'},
            editable=True,
            page_size=20
        )
        return data_table, "", df.to_dict('records')
    except ValueError as e:
        return html.Div(), str(e), {}

@app.callback(
    Output('stored-data', 'data'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def update_stored_data(edited_data):
    add_callback_source_code(update_stored_data)
    return edited_data

@app.callback(
    Output('output-data-upload', 'children', allow_duplicate=True),
    [Input('stored-data', 'modified_timestamp')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def update_table_from_store(ts, stored_data):
    add_callback_source_code(update_table_from_store)
    if ts is None or not stored_data:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(stored_data)
    data_table = dash_table.DataTable(
        id='table',
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
        style_table={'overflowX': 'auto'},
        editable=True,
        page_size=20
    )
    return data_table

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    Output('stored-data', 'data'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def update_stored_data(edited_data):
    add_callback_source_code(update_stored_data)
    return edited_data

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True)],
    [Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def handle_dropdown_actions(n_clicks_convert, n_clicks_clean, stored_data):
    # Get the triggering input (i.e., which dropdown item was clicked)
    add_callback_source_code(handle_dropdown_actions)

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Information about the clicked dropdown item
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    prop_info = json.loads(prop_id)

    df = pd.DataFrame(stored_data)
    column_name = prop_info['index']

    if prop_info['type'] == "convert":
        if prop_info['to'] == "Numeric":
            # Convert the column to numeric
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            
            # Check if there are any NaN values in the resulting series
            if temp_series.isna().any():
                # If there are NaN values, then the conversion is not practical
                raise dash.exceptions.PreventUpdate
            else:
                df[column_name] = temp_series
        else:
            df[column_name] = df[column_name].astype(str)

    elif prop_info['type'] == "clean":
        if prop_info['action'] == "min":
            df[column_name].fillna(df[column_name].min(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].min(), inplace=True)")
        elif prop_info['action'] == "max":
            df[column_name].fillna(df[column_name].max(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].max(), inplace=True)")
        elif prop_info['action'] == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].mean(), inplace=True)")
        elif prop_info['action'] == "zero":
            df[column_name].fillna(0, inplace=True)
            log_user_action(f"df[{column_name}].fillna(0, inplace=True)")
        elif prop_info['action'] == "na_string":
            df[column_name].fillna('N/A', inplace=True)
            log_user_action(f" df[{column_name}].fillna('N/A', inplace=True)")
        elif prop_info['action'] == "most_frequent":
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].empty else "N/A"
            df[column_name].fillna(most_frequent, inplace=True)
            log_user_action(f"df[{column_name}].fillna({most_frequent}, inplace=True)")
        # Handle normalization actions
        elif prop_info['action'] == "normalize_new":
            normalized_col_name = "normalized_" + column_name
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[normalized_col_name] = normalized_values.round(3)
            log_user_action(f"df[{normalized_col_name}] = normalized_values.round(3)")
        elif prop_info['action'] == "normalize_replace":
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[column_name] = normalized_values.round(3)
            log_user_action(f"df[{column_name}] = normalized_values.round(3)")

    return [df.to_dict('records')]

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True)],
    [Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def handle_dropdown_actions(n_clicks_convert, n_clicks_clean, stored_data):
    # Get the triggering input (i.e., which dropdown item was clicked)
    add_callback_source_code(handle_dropdown_actions)

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Information about the clicked dropdown item
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    prop_info = json.loads(prop_id)

    df = pd.DataFrame(stored_data)
    column_name = prop_info['index']

    if prop_info['type'] == "convert":
        if prop_info['to'] == "Numeric":
            # Convert the column to numeric
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            
            # Check if there are any NaN values in the resulting series
            if temp_series.isna().any():
                # If there are NaN values, then the conversion is not practical
                raise dash.exceptions.PreventUpdate
            else:
                df[column_name] = temp_series
        else:
            df[column_name] = df[column_name].astype(str)

    elif prop_info['type'] == "clean":
        if prop_info['action'] == "min":
            df[column_name].fillna(df[column_name].min(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].min(), inplace=True)")
        elif prop_info['action'] == "max":
            df[column_name].fillna(df[column_name].max(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].max(), inplace=True)")
        elif prop_info['action'] == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].mean(), inplace=True)")
        elif prop_info['action'] == "zero":
            df[column_name].fillna(0, inplace=True)
            log_user_action(f"df[{column_name}].fillna(0, inplace=True)")
        elif prop_info['action'] == "na_string":
            df[column_name].fillna('N/A', inplace=True)
            log_user_action(f" df[{column_name}].fillna('N/A', inplace=True)")
        elif prop_info['action'] == "most_frequent":
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].empty else "N/A"
            df[column_name].fillna(most_frequent, inplace=True)
            log_user_action(f"df[{column_name}].fillna({most_frequent}, inplace=True)")
        # Handle normalization actions
        elif prop_info['action'] == "normalize_new":
            normalized_col_name = "normalized_" + column_name
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[normalized_col_name] = normalized_values.round(3)
            log_user_action(f"df[{normalized_col_name}] = normalized_values.round(3)")
        elif prop_info['action'] == "normalize_replace":
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[column_name] = normalized_values.round(3)
            log_user_action(f"df[{column_name}] = normalized_values.round(3)")

    return [df.to_dict('records')]

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True)],
    [Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def handle_dropdown_actions(n_clicks_convert, n_clicks_clean, stored_data):
    # Get the triggering input (i.e., which dropdown item was clicked)
    add_callback_source_code(handle_dropdown_actions)

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Information about the clicked dropdown item
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    prop_info = json.loads(prop_id)

    df = pd.DataFrame(stored_data)
    column_name = prop_info['index']

    if prop_info['type'] == "convert":
        if prop_info['to'] == "Numeric":
            # Convert the column to numeric
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            
            # Check if there are any NaN values in the resulting series
            if temp_series.isna().any():
                # If there are NaN values, then the conversion is not practical
                raise dash.exceptions.PreventUpdate
            else:
                df[column_name] = temp_series
        else:
            df[column_name] = df[column_name].astype(str)

    elif prop_info['type'] == "clean":
        if prop_info['action'] == "min":
            df[column_name].fillna(df[column_name].min(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].min(), inplace=True)")
        elif prop_info['action'] == "max":
            df[column_name].fillna(df[column_name].max(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].max(), inplace=True)")
        elif prop_info['action'] == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].mean(), inplace=True)")
        elif prop_info['action'] == "zero":
            df[column_name].fillna(0, inplace=True)
            log_user_action(f"df[{column_name}].fillna(0, inplace=True)")
        elif prop_info['action'] == "na_string":
            df[column_name].fillna('N/A', inplace=True)
            log_user_action(f" df[{column_name}].fillna('N/A', inplace=True)")
        elif prop_info['action'] == "most_frequent":
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].empty else "N/A"
            df[column_name].fillna(most_frequent, inplace=True)
            log_user_action(f"df[{column_name}].fillna({most_frequent}, inplace=True)")
        # Handle normalization actions
        elif prop_info['action'] == "normalize_new":
            normalized_col_name = "normalized_" + column_name
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[normalized_col_name] = normalized_values.round(3)
            log_user_action(f"df[{normalized_col_name}] = normalized_values.round(3)")
        elif prop_info['action'] == "normalize_replace":
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[column_name] = normalized_values.round(3)
            log_user_action(f"df[{column_name}] = normalized_values.round(3)")

    return [df.to_dict('records')]

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True)],
    [Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def handle_dropdown_actions(n_clicks_convert, n_clicks_clean, stored_data):
    # Get the triggering input (i.e., which dropdown item was clicked)
    add_callback_source_code(handle_dropdown_actions)

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Information about the clicked dropdown item
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    prop_info = json.loads(prop_id)

    df = pd.DataFrame(stored_data)
    column_name = prop_info['index']

    if prop_info['type'] == "convert":
        if prop_info['to'] == "Numeric":
            # Convert the column to numeric
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            
            # Check if there are any NaN values in the resulting series
            if temp_series.isna().any():
                # If there are NaN values, then the conversion is not practical
                raise dash.exceptions.PreventUpdate
            else:
                df[column_name] = temp_series
        else:
            df[column_name] = df[column_name].astype(str)

    elif prop_info['type'] == "clean":
        if prop_info['action'] == "min":
            df[column_name].fillna(df[column_name].min(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].min(), inplace=True)")
        elif prop_info['action'] == "max":
            df[column_name].fillna(df[column_name].max(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].max(), inplace=True)")
        elif prop_info['action'] == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].mean(), inplace=True)")
        elif prop_info['action'] == "zero":
            df[column_name].fillna(0, inplace=True)
            log_user_action(f"df[{column_name}].fillna(0, inplace=True)")
        elif prop_info['action'] == "na_string":
            df[column_name].fillna('N/A', inplace=True)
            log_user_action(f" df[{column_name}].fillna('N/A', inplace=True)")
        elif prop_info['action'] == "most_frequent":
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].empty else "N/A"
            df[column_name].fillna(most_frequent, inplace=True)
            log_user_action(f"df[{column_name}].fillna({most_frequent}, inplace=True)")
        # Handle normalization actions
        elif prop_info['action'] == "normalize_new":
            normalized_col_name = "normalized_" + column_name
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[normalized_col_name] = normalized_values.round(3)
            log_user_action(f"df[{normalized_col_name}] = normalized_values.round(3)")
        elif prop_info['action'] == "normalize_replace":
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[column_name] = normalized_values.round(3)
            log_user_action(f"df[{column_name}] = normalized_values.round(3)")

    return [df.to_dict('records')]

@app.callback(
    Output('output-data-upload', 'children', allow_duplicate=True),
    [Input('stored-data', 'modified_timestamp')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def update_table_from_store(ts, stored_data):
    add_callback_source_code(update_table_from_store)
    if ts is None or not stored_data:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(stored_data)
    data_table = dash_table.DataTable(
        id='table',
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
        style_table={'overflowX': 'auto'},
        editable=True,
        page_size=20
    )
    return data_table

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    Output('stored-data', 'data'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def update_stored_data(edited_data):
    add_callback_source_code(update_stored_data)
    return edited_data

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    add_callback_source_code(update_summary)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows

@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True)],
    [Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def handle_dropdown_actions(n_clicks_convert, n_clicks_clean, stored_data):
    # Get the triggering input (i.e., which dropdown item was clicked)
    add_callback_source_code(handle_dropdown_actions)

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Information about the clicked dropdown item
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    prop_info = json.loads(prop_id)

    df = pd.DataFrame(stored_data)
    column_name = prop_info['index']

    if prop_info['type'] == "convert":
        if prop_info['to'] == "Numeric":
            # Convert the column to numeric
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            
            # Check if there are any NaN values in the resulting series
            if temp_series.isna().any():
                # If there are NaN values, then the conversion is not practical
                raise dash.exceptions.PreventUpdate
            else:
                df[column_name] = temp_series
        else:
            df[column_name] = df[column_name].astype(str)

    elif prop_info['type'] == "clean":
        if prop_info['action'] == "min":
            df[column_name].fillna(df[column_name].min(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].min(), inplace=True)")
        elif prop_info['action'] == "max":
            df[column_name].fillna(df[column_name].max(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].max(), inplace=True)")
        elif prop_info['action'] == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].mean(), inplace=True)")
        elif prop_info['action'] == "zero":
            df[column_name].fillna(0, inplace=True)
            log_user_action(f"df[{column_name}].fillna(0, inplace=True)")
        elif prop_info['action'] == "na_string":
            df[column_name].fillna('N/A', inplace=True)
            log_user_action(f" df[{column_name}].fillna('N/A', inplace=True)")
        elif prop_info['action'] == "most_frequent":
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].empty else "N/A"
            df[column_name].fillna(most_frequent, inplace=True)
            log_user_action(f"df[{column_name}].fillna({most_frequent}, inplace=True)")
        # Handle normalization actions
        elif prop_info['action'] == "normalize_new":
            normalized_col_name = "normalized_" + column_name
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[normalized_col_name] = normalized_values.round(3)
            log_user_action(f"df[{normalized_col_name}] = normalized_values.round(3)")
        elif prop_info['action'] == "normalize_replace":
            normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
            df[column_name] = normalized_values.round(3)
            log_user_action(f"df[{column_name}] = normalized_values.round(3)")

    return [df.to_dict('records')]

@app.callback(
    [Output('datatable-upload-container', 'data'),
     Output('datatable-upload-container', 'columns'),
     Output('xaxis-viscolumn-dropdown', 'options'),
     Output('yaxis-viscolumn-dropdown', 'options')],
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_dropdown_output(stored_data):
    add_callback_source_code(update_dropdown_output)
    if not stored_data:
        return [], [], [], []

    df = pd.DataFrame(stored_data)

    columns = [{"name": i, "id": i} for i in df.columns]
    options = [{'label': col, 'value': col} for col in df.columns]

    return df.to_dict('records'), columns, options, options

@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('stored-data', 'data')  # Fetch data from dcc.Store
)
def update_graph(x_column, y_column, vis_type, data):
    add_callback_source_code(update_graph)
    if not data or not x_column or not y_column or not vis_type:
        return {}

    df = pd.DataFrame(data)

    if vis_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif vis_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif vis_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif vis_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif vis_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif vis_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif vis_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif vis_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)
    else:
        return {}
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig

@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('stored-data', 'data')  # Fetch data from dcc.Store
)
def update_graph(x_column, y_column, vis_type, data):
    add_callback_source_code(update_graph)
    if not data or not x_column or not y_column or not vis_type:
        return {}

    df = pd.DataFrame(data)

    if vis_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif vis_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif vis_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif vis_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif vis_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif vis_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif vis_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif vis_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)
    else:
        return {}
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig

@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('stored-data', 'data')  # Fetch data from dcc.Store
)
def update_graph(x_column, y_column, vis_type, data):
    add_callback_source_code(update_graph)
    if not data or not x_column or not y_column or not vis_type:
        return {}

    df = pd.DataFrame(data)

    if vis_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif vis_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif vis_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif vis_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif vis_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif vis_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif vis_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif vis_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)
    else:
        return {}
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig

@app.callback(
    Output('saved-visgraphs-container', 'children'),
    Input('save-graph-button', 'n_clicks'),
    State('visualisation-graph', 'figure'),
    State('saved-visgraphs-container', 'children')
)
def save_current_graph(n_clicks, current_figure, current_saved_graphs):
    add_callback_source_code(save_current_graph)
    if not current_figure:
        raise dash.exceptions.PreventUpdate
    current_graph = dcc.Graph(figure=current_figure)
    log_user_action(f"Save Graph: current_graph = dcc.Graph(figure=current_figure)")
    if not current_saved_graphs:
        current_saved_graphs = []
    current_saved_graphs.append(current_graph)
    log_user_action(f"Save Graph: current_saved_graphs.append(current_graph)")
    return current_saved_graphs

@app.callback(
    Output('input-dict-store', 'data'),
    [Input('data-analysis-dropdown', 'value')]
)
def update_input_dict_store(selected_analysis_method):
    add_callback_source_code(update_input_dict_store)
    if selected_analysis_method == 'clustering':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select columns:",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "number_selection",
                "DataType": "numeric",
                "Text": "Number of Clusters:",
                "AcceptableValues": {"min": 1, "max": None}
            }
        ]
    elif selected_analysis_method == 'classification':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            }, 
            {
                "Input": "column_selection",
                "DataType": "string",
                "Text": "Select target column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "dropdown",
                "DataType": "string",
                "Text": "Select Kernel Type:",
                "AcceptableValues": ["linear", "poly", "rbf", "sigmoid"]
            }
        ]
    else:
        # Handle other cases or invalid selections appropriately
        return []

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

@app.callback(
    Output('input-dict-store', 'data'),
    [Input('data-analysis-dropdown', 'value')]
)
def update_input_dict_store(selected_analysis_method):
    add_callback_source_code(update_input_dict_store)
    if selected_analysis_method == 'clustering':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select columns:",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "number_selection",
                "DataType": "numeric",
                "Text": "Number of Clusters:",
                "AcceptableValues": {"min": 1, "max": None}
            }
        ]
    elif selected_analysis_method == 'classification':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            }, 
            {
                "Input": "column_selection",
                "DataType": "string",
                "Text": "Select target column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "dropdown",
                "DataType": "string",
                "Text": "Select Kernel Type:",
                "AcceptableValues": ["linear", "poly", "rbf", "sigmoid"]
            }
        ]
    else:
        # Handle other cases or invalid selections appropriately
        return []

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

@app.callback(
    Output('download', 'data', allow_duplicate=True),
    [Input('save-button', 'n_clicks'), Input('export-commands-button', 'n_clicks')],
    [State('table', 'data'), State('export-format-dropdown', 'value')],
    prevent_initial_call=True
)
def export_or_save(n_clicks_save, n_clicks_export, rows, export_format):

    add_callback_source_code(export_or_save)

    if n_clicks_save is None and n_clicks_export is None:
        raise dash.exceptions.PreventUpdate

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'save-button':
        if n_clicks_save is None:
            raise dash.exceptions.PreventUpdate

        df_to_save = pd.DataFrame(rows)
        command = f"Saved data to {export_format} file"  # Define the command for saving the data

        if export_format == 'csv':
            log_user_action(command, "edited_data.csv")  # Log the user action with the command and filename
            csv_string = df_to_save.to_csv(index=False, encoding='utf-8')
            return dict(content=csv_string, filename="edited_data.csv")
        elif export_format == 'xlsx':
            log_user_action(command, "edited_data.xlsx")  # Log the user action with the command and filename
            xlsx_io = io.BytesIO()
            df_to_save.to_excel(xlsx_io, index=False, engine='openpyxl')
            xlsx_io.seek(0)
            # Encode the Excel data to base64
            xlsx_base64 = base64.b64encode(xlsx_io.getvalue()).decode('utf-8')
            return dict(content=xlsx_base64, filename="edited_data.xlsx",
                        type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base64=True)
        elif export_format == 'json':
            log_user_action(command, "edited_data.json")  # Log the user action with the command and filename
            json_string = df_to_save.to_json(orient='records')
            return dict(content=json_string, filename="edited_data.json")

    elif trigger_id == 'export-commands-button':
        if n_clicks_export is None:
            raise dash.exceptions.PreventUpdate

        # Define the filename for the .py file where user actions will be saved
        filename = "user_actions.py"

        # Call the function to save user actions to the .py file
        write_callback_functions_to_file(filename)

        # Return the .py file for download
        with open(filename, "r") as file:
            file_content = file.read()

        return dict(content=file_content, filename=filename, type="text/python")


    raise dash.exceptions.PreventUpdate  # Handle unexpected situations

if __name__ == '__main__':
    app.run_server(debug=True)
