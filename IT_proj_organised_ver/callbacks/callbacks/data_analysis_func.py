"""
File:               data_analysis_func.py
Description:        This module provides functions for data analysis, including clustering, classification, 
                    and visualization. It utilizes popular data analysis and machine learning libraries 
                    such as Pandas, NumPy, Seaborn, Plotly, and Scikit-Learn.
Authors:            Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Date Updated:       2023-10-18
"""
# Import necessary libraries for data analysis and visualization
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.figure_factory as ff

# Import necessary libraries for machine learning and clustering
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# Import necessary libraries for Dash web application framework
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, ALL


# Define clustering function using KMeans algorithm
def clustering_KMeans(inputData, selectedColumns, numClusters):
    """Cluster data using KMeans algorithm and visualize the clusters."""
    if inputData.empty:
        return []

    fig = []

    try:
        # Define and fit the KMeans model
        kmeans = KMeans(n_clusters=numClusters)
        selectedData = inputData[selectedColumns]
        cluster_assignments = kmeans.fit_predict(selectedData)

        # Generate an HTML report for cluster statistics
        cluster_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        statistics = html.Div([
            html.H4("Cluster Statistics:"),
            html.P(f"Number of Clusters (K): {numClusters}"),
            html.P(f"Cluster Centers:\n{cluster_centers}"),
            html.P(f"Inertia (Within-cluster Sum of Squares): {inertia}")
        ])

        # Decide the plot type based on the number of selected columns
        if len(selectedColumns) == 2:
            scatter_plot = px.scatter(selectedData, x=selectedColumns[0], y=selectedColumns[1], color=cluster_assignments, title='K-means Clustering')
            fig.append(scatter_plot)
        else:
            pair_plot = px.scatter_matrix(selectedData, dimensions=selectedColumns, color=cluster_assignments)        
            cluster_profile = pd.DataFrame(cluster_centers, columns=selectedData.columns)
            bar_plot = px.bar(cluster_profile)
            fig.append(bar_plot)
            fig.append(pair_plot)
        return fig, statistics

    except ValueError as e:
        # Handle common errors related to data issues
        if "Input X contains NaN" in str(e):
            print("Data contains NaN values. Please clean the data before analyzing.")
        else:
            print(f"An error occurred: {str(e)}")
        except_msg = "An error occurred during clustering. Please ensure data is clean and try again."
        return except_msg


# Define a function for SVM-based classification
def classification_SVM(inputData, selectedColumns, targetColumn, kernel):
    """Classify data using SVM and visualize the results."""
    fig = []

    # Split data into training and test sets
    selectedData = inputData[selectedColumns]
    X_train, X_test, y_train, y_test = train_test_split(selectedData, inputData[targetColumn], test_size=0.2, random_state=42)

    # Prepare the labels for SVM
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Define and train the SVM classifier
    classifier = svm.SVC(kernel=kernel)
    classifier.fit(X_train, y_train)

    # Make predictions using the trained SVM model
    y_pred = classifier.predict(X_test)

    # Compute classification performance metrics
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Generate an HTML report for classification statistics
    statistics = html.Div([
        html.H4("Classification Statistics:"),
        html.P(f"Precision: {class_report['weighted avg']['precision']}"),
        html.P(f"Recall: {class_report['weighted avg']['recall']}"),
        html.P(f"F1-Score: {class_report['weighted avg']['f1-score']}"),
        html.P(f"Support: {class_report['weighted avg']['support']}")
    ])

    # Create a confusion matrix plot
    figure = generateConfusionMatrix(cm, class_labels=classifier.classes_)
    fig.append(figure)

    # If binary classification, generate an AUC curve plot
    if len(classifier.classes_) == 2:
        y_pred_bin = label_binarize(y_pred, classes=classifier.classes_)
        y_test_bin = label_binarize(y_test, classes=classifier.classes_)
        AUC_Plot = generateAUC(y_test_bin, y_pred_bin)
        fig.append(AUC_Plot)

    return fig, statistics


# Helper function to generate an AUC curve plot
def generateAUC(y_test, y_pred):
    """Generate an AUC curve plot given true labels and predictions."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    fig = px.line(x=fpr, y=tpr, title=f'ROC curve (AUC = {roc_auc:.2f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    return fig


# Helper function to generate a confusion matrix plot
def generateConfusionMatrix(cm, class_labels):
    """Generate a confusion matrix heatmap plot given a confusion matrix."""
    heatmap = go.Heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        colorscale="Blues",
        colorbar=dict(title="Count"),
    )
    layout = go.Layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted labels"),
        yaxis=dict(title="True labels"),
    )
    return go.Figure(data=[heatmap], layout=layout)
