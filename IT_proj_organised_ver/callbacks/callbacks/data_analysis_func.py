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
from dash.dependencies import Input, Output, State, ALL
# !Make sure the functions return a list of plots and a HTML object 
# !Make sure plots are of type plotly.graph_objs._figure.Figure


def clustering_KMeans(inputData, selectedColumns,numClusters):
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
            html.H4("Cluster Statistics:"),
            html.P(f"Number of Clusters (K): {numClusters}"),
            html.P(f"Cluster Centers:\n{cluster_centers}"),
            html.P(f"Inertia (Within-cluster Sum of Squares): {inertia}")
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
        if "Input X contains NaN" in str(e):
            print("Data contains NaN values. Please clean the data before analyzing.")
        else:
            print(f"An error occurred: {str(e)}")
    
    # You can return an informative message instead of a figure, 
    # so you could render this message in a dedicated Div in your Dash app.
    except_msg = "An error occurred during clustering. Please ensure data is clean and try again."
    
    return except_msg


def classification_SVM(inputData, selectedColumns, targetColumn,kernel):
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
        html.H4("Classification Statistics:"),
        html.P(f"Precision: {class_report['weighted avg']['precision']}"),
        html.P(f"Recall: {class_report['weighted avg']['recall']}"),
        html.P(f"F1-Score: {class_report['weighted avg']['f1-score']}"),
        html.P(f"Support: {class_report['weighted avg']['support']}")
      ])
        # Create the confusion matrix plot
    print('8')
    figure = generateConfusionMatrix(cm,class_labels=classifier.classes_)
    print('9')
    figure.update_layout(title='SVM Classification - Confusion Matrix')
    print('10')
    fig.append(figure)
    print('11')
    print(len(classifier.classes_))
    #return fig, statistics
    if len(classifier.classes_) == 2:
        y_pred_bin = label_binarize(y_pred, classes=classifier.classes_)
        y_test_bin = label_binarize(y_test, classes=classifier.classes_)
        AUC_Plot =  generateAUC(y_test_bin,y_pred_bin)
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

