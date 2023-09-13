import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1('Predictive Statistics Dashboard'),

    # Dropdown to select the method
    dcc.Dropdown(
        id='method-selector',
        options=[
            {'label': 'Classification', 'value': 'classification'},
            {'label': 'Clustering', 'value': 'clustering'},
        ],
        value='classification'  # Default method
    ),

    # Input components for method-specific parameters
    dcc.Input(
        id='target-column-input',
        type='text',
        placeholder='Enter target column for classification',
        style={'display': 'none'}  # Hide by default
    ),

    dcc.Input(
        id='num-clusters-input',
        type='number',
        placeholder='Enter number of clusters for clustering',
        style={'display': 'none'}  # Hide by default
    ),

    # Button to trigger method execution
    html.Button('Run', id='run-button'),

    html.H2(id='classification-accuracy', style={'display': 'none'}),
    html.H2(id='cluster-silhouette-score', style={'display': 'none'}),

    dcc.Graph(
        id='cluster-plot',
        style={'display': 'none'}  # Hide by default
    ),
])


@app.callback(
    Output('target-column-input', 'style'),
    Output('num-clusters-input', 'style'),
    Output('cluster-plot', 'style'),
    Output('classification-accuracy', 'style'),
    Output('cluster-silhouette-score', 'style'),
    Input('method-selector', 'value')
)
def show_hide_inputs(method):
    if method == 'classification':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}


@app.callback(
    Output('classification-accuracy', 'children'),
    Output('cluster-silhouette-score', 'children'),
    Output('cluster-plot', 'figure'),
    Input('run-button', 'n_clicks'),
    Input('method-selector', 'value'),
    Input('target-column-input', 'value'),
    Input('num-clusters-input', 'value'),
)
def perform_analysis(n_clicks, method, target_column, num_clusters):
    if n_clicks is None:
        return '', '', {'data': []}

    data = pd.read_csv('social_capital.csv')

    if method == 'classification':
        if not target_column:
            return 'Target column not specified', '', {'data': []}
        classification_accuracy = perform_classification(data, target_column)
        return 'Classification Accuracy: {:.2f}'.format(classification_accuracy), '', {'data': []}
    else:
        if not num_clusters:
            return '', 'Number of clusters not specified', {'data': []}
        cluster_silhouette_score, scatter_plot = perform_clustering(data, num_clusters)
        return '', 'Cluster Silhouette Score: {:.2f}'.format(cluster_silhouette_score), scatter_plot


def perform_classification(data, target_column):
    """
    Perform classification on a Pandas DataFrame.

    Args:
    - data (pd.DataFrame): The input DataFrame containing the features and target variable for classification.
    - target_column (str): The name of the column containing the target variable for classification.

    Returns:
    - classification_accuracy (float): Accuracy of the classification model.
    """
    # Split the data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

    # Apply one-hot encoding to categorical columns
    if categorical_columns:
        transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(), categorical_columns)],
            remainder='passthrough'
        )
        X_encoded = transformer.fit_transform(X)
    else:
        X_encoded = X  # No categorical columns to encode

    # Split the data into training and testing sets for classification
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Perform Classification using Random Forest
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Calculate classification accuracy
    classification_accuracy = accuracy_score(y_test, y_pred)

    return classification_accuracy

def perform_clustering(data, num_clusters=3):
    """
    Perform clustering on a Pandas DataFrame.

    Args:
    - data (pd.DataFrame): The input DataFrame containing the features for clustering.
    - num_clusters (int): The number of clusters for K-Means clustering (default is 3).

    Returns:
    - cluster_silhouette_score (float): Silhouette score of the clustering model.
    - scatter_plot (dcc.Graph): Dash graph displaying the clustering results.
    """
    # Remove or handle categorical columns as needed

    # Select only numeric columns (assuming other columns are categorical)
    numeric_data = data.select_dtypes(include='number')

    # Standardize the numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    # Perform Clustering using K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate cluster silhouette score (a measure of clustering quality)
    cluster_silhouette_score = silhouette_score(X_scaled, cluster_labels)

    # Create a Dash graph for clustering results
    scatter_plot = dcc.Graph(
        id='cluster-plot',
        figure={
            'data': [
                go.Scatter(
                    x=X_scaled[:, 0],
                    y=X_scaled[:, 1],
                    mode='markers',
                    marker=dict(color=cluster_labels, opacity=0.7, colorscale='Viridis'),
                    text=cluster_labels,
                )
            ],
            'layout': go.Layout(
                title='Cluster Plot',
                xaxis={'title': 'Feature 1'},
                yaxis={'title': 'Feature 2'},
                hovermode='closest',
            )
        }
    )

    return cluster_silhouette_score, scatter_plot



# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
