import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def perform_classification(data, target_column):
    """
    Perform classification on DataFrame using Random Forest.

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
    # set random_state=42 for reproducibility
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Calculate classification accuracy
    classification_accuracy = accuracy_score(y_test, y_pred)

    return classification_accuracy


from sklearn.preprocessing import StandardScaler

def perform_clustering(data, num_clusters=3):
    """
    Perform clustering on DataFrame.

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
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
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


# Create a Dash app
app = dash.Dash(__name__)

# Example usage:
if __name__ == "__main__":
    data = pd.read_csv('social_capital.csv')

    # Specify the target column for classification
    target_column = 'YEAR'

    classification_accuracy = perform_classification(data, target_column)

    cluster_silhouette_score, scatter_plot = perform_clustering(data)

    # Define the layout of the Dash app
    app.layout = html.Div([
        html.H1('Predictive Statistics Dashboard'),

        html.H2('Classification Accuracy: {:.2f}'.format(classification_accuracy)),

        html.H2('Cluster Silhouette Score: {:.2f}'.format(cluster_silhouette_score)),

        scatter_plot  # Display the Dash graph
    ])

    # Run the Dash app
    app.run_server(debug=True)