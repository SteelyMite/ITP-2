from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# !Make sure the functions return a list of plots and a HTML object 
# !Make sure plots are of type plotly.graph_objs._figure.Figure


def clustering_KMeans(inputData, selectedColumns,numClusters):
    fig = []
    kmeans = KMeans(n_clusters=numClusters)
    cluster_assignments = kmeans.fit_predict(inputData[selectedColumns])

    #? Create data analysis HTML report
    cluster_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    cluster_center_x = cluster_centers[:, 0]  # X-coordinates of cluster centers
    cluster_center_y = cluster_centers[:, 1]  # Y-coordinates of cluster centers





    statistics = html.Div([
        html.H4("Cluster Statistics:"),
        html.P(f"Number of Clusters (K): {numClusters}"),
        html.P(f"Cluster Centers:\n{cluster_centers}"),
        html.P(f"Inertia (Within-cluster Sum of Squares): {inertia}")
    ])
    #? Generate Plots
    # fig = px.scatter(df, x=selectedColumns[0], y=selectedColumns[1], color="blue", title='K-means Clustering')
    # fig = px.scatter(inputData[selectedColumns], x=selectedColumns[0], y=selectedColumns[1], color=cluster_assignments, title='K-means Clustering')


    #plotting the results:
    for i in range(len(selectedColumns)):
        for j in range(len(selectedColumns)):   
            if(i!=j):
                scatter_plot = px.scatter(inputData[selectedColumns], x=selectedColumns[i], y=selectedColumns[j], color=cluster_assignments, title='K-means Clustering')
           
                fig.append(scatter_plot)   


    # fig.append(px.scatter(inputData[selectedColumns], x=selectedColumns[0], y=selectedColumns[1], color=cluster_assignments, title='K-means Clustering'))



    return fig, statistics


def classification_SVM(inputData, selectedColumns, targetColumn,kernel):
    fig = []
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputData[selectedColumns], inputData[targetColumn], test_size=0.2, random_state=42)
    # Create and train SVM Classifier 
    classifier = svm.SVC(kernel=kernel) #!Make sure to update kernel type
    classifier.fit(X_train, y_train)
    # Predict labels on the test data
    y_pred = classifier.predict(X_test)
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    # fig.append(cm)
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
    figure = generateConfusionMatrix(cm,class_labels=classifier.classes_)
    fig.append(figure)
    
    if(len(classifier.classes_ == 2)):
        y_pred_bin = label_binarize(y_pred, classes=classifier.classes_)
        y_test_bin = label_binarize(y_test, classes=classifier.classes_)
        AUC_Plot =  generateAUC(y_test_bin,y_pred_bin)
        fig.append(AUC_Plot)

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
        colorscale="Blues",
        colorbar=dict(title="Count"),
    )

    # Define the layout for the confusion matrix plot
    layout = go.Layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted labels"),
        yaxis=dict(title="True labels"),
    )

    return go.Figure(data=[heatmap], layout=layout)

