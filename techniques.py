import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Read the CSV data into a DataFrame
df = pd.read_csv('iris.csv')

# Define the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Label("Select Columns for K-means Clustering:"),
    dcc.Dropdown(
        id='column-selector',
        options=[{'label': col, 'value': col} for col in df.columns],
        multi=True
    ),
    html.Label("Select the Number of Clusters (K):"),
    dcc.Input(id='k-input', type='number', value=3),
    html.Button(id='run-button', n_clicks=0, children='Run K-means Clustering'),
    dcc.Graph(id='cluster-plot'),
    html.Div(id='cluster-statistics')
])

# Define a callback to run K-means clustering and update the results
@app.callback(
    [Output('cluster-plot', 'figure'),
     Output('cluster-statistics', 'children')],
    [Input('run-button', 'n_clicks')],
    [dash.dependencies.State('column-selector', 'value'),
     dash.dependencies.State('k-input', 'value')]
)

# Perform K-means clustering and display the results
def run_kmeans(n_clicks, selected_columns, k):
    if n_clicks > 0 and selected_columns and k:
        # Select the specified columns from the DataFrame
        data = df[selected_columns]

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(data)
        df['Cluster'] = labels

        # Create the cluster plot
        fig = px.scatter(df, x=data.columns[0], y=data.columns[1], color='Cluster', title='K-means Clustering')

        # Calculate and display cluster statistics
        cluster_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        statistics = html.Div([
            html.H4("Cluster Statistics:"),
            html.P(f"Number of Clusters (K): {k}"),
            html.P(f"Cluster Centers:\n{cluster_centers}"),
            html.P(f"Inertia (Within-cluster Sum of Squares): {inertia}")
        ])

        return fig, statistics

    # Return empty outputs if the button hasn't been clicked or fields are empty
    return {}, ""

# Perform SVM Classification 
def perform_svm_classification(df, feature_columns, label_column):
    # Select the features and labels from the DataFrame
    X = df[feature_columns]
    y = df[label_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM classifier (you can change the kernel as needed)
    clf = svm.SVC(kernel='linear')

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Predict labels on the test data
    y_pred = clf.predict(X_test)

    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    return cm, class_report, clf
    

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
