import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

df = pd.read_csv('iris.csv')

def clustering_KMeans(inputData, selectedColumns,numClusters):
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
    fig = px.scatter(df, x=selectedColumns[0], y=selectedColumns[1], color='Cluster', title='K-means Clustering')
    return fig, statistics


def classification_SVM(inputData, selectedColumns, targetColumn):
    fig = []
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputData[selectedColumns], inputData[targetColumn], test_size=0.2, random_state=42)
    # Create and train SVM Classifier 
    classifier = svm.SVC(kernel='kernel')
    classifier.fit(X_train, y_train)
    # Predict labels on the test data
    y_pred = classifier.predict(X_test)
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    fig.append(cm)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Create data analysis HTML report 
    statistics = html.Div([
        html.H4("Classification Statistics:"),
        html.P(f"Precision: {class_report['weighted avg']['precision']}"),
        html.P(f"Recall: {class_report['weighted avg']['recall']}"),
        html.P(f"F1-Score: {class_report['weighted avg']['f1-score']}"),
        html.P(f"Support: {class_report['weighted avg']['support']}")
      ])
    

    # Create the confusion matrix plot
    labels = np.unique(cm)
    z_text = [[str(y) for y in x] for x in classification_SVM]

    figure = ff.create_annotated_heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        annotation_text=z_text,
        colorscale='Viridis'
    )

    figure.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
    yaxis_title='True'
    )

    fig.append(figure)

    return fig, statistics





