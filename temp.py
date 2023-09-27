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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

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
    # fig = px.scatter(df, x=selectedColumns[0], y=selectedColumns[1], color="blue", title='K-means Clustering')
    fig = px.scatter(inputData[selectedColumns], x=selectedColumns[0], y=selectedColumns[1], color=cluster_assignments, title='K-means Clustering')
    print(fig)
    # Check fig data type
    print(type(fig))
    return [fig], statistics


def classification_SVM(inputData, selectedColumns, targetColumn):
    fig = []
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputData[selectedColumns], inputData[targetColumn], test_size=0.2, random_state=42)
    # Create and train SVM Classifier 
    classifier = svm.SVC(kernel='linear') #!What kernel?
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
    figure = generateConfusionMatrix(cm)

    # # Create Decision Boundary plot
    # # Create scatter plot
    # plt.figure(figsize=(10, 6))

    # # Scatter plot for class 0
    # support_vectors = classifier.support_vectors_
    # plt.scatter(X_train[y_train == 0]['feature1'], X_train[y_train == 0]['feature2'], label='Class 0', c='b')

    # # Scatter plot for class 1
    # plt.scatter(X_train[y_train == 1]['feature1'], X_train[y_train == 1]['feature2'], label='Class 1', c='r')

    # # Scatter plot for support vectors
    # plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='g', marker='*', s=100, label='Support Vectors')

    # # Decision boundary
    # xx, yy = np.meshgrid(np.linspace(X_train['feature1'].min(), X_train['feature1'].max(), 100),
    #                     np.linspace(X_train['feature2'].min(), X_train['feature2'].max(), 100))
    # Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend()
    # plt.title('SVM Classification with Decision Boundary')

    # plt.show()





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

    # Define the layout for the confusion matrix plot
    layout = go.Layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted labels"),
        yaxis=dict(title="True labels"),
    )
    return go.Figure(data=[heatmap], layout=layout)



