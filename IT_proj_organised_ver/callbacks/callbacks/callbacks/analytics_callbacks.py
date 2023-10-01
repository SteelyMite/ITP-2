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


# from cluster_class_callbacks import classification_SVM

@app.callback(
    Output('clustering-results', 'children'),
    Input('stored-data', 'data'),
    Input('selected-columns-dropdown', 'value'),
    Input('num-clusters-input', 'value'),
    prevent_initial_call=True
)
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