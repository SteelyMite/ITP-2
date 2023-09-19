# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans

# Load your data into a DataFrame (you can replace this with your data loading code)
data = pd.read_csv('social_capital.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Data Analysis Techniques"),
    
    # Dropdown to select the analysis technique
    dcc.Dropdown(
        id='analysis-technique',
        options=[
            {'label': 'Clustering', 'value': 'clustering'},
            {'label': 'Classification', 'value': 'classification'},
        ],
        value='clustering'  # Default selection
    ),

    # Dropdown to select columns for clustering
    dcc.Dropdown(
        id='clustering-columns',
        options=[{'label': col, 'value': col} for col in data.columns],
        multi=True,  # Allow multiple column selection
        value=['YEAR', 'Cesarean Delivery Rate'],  # Default selection
    ),
    
    # Output for displaying the summary report
    dcc.Graph(id='summary-report'),
    html.Div(id='clustering-report'),  # Add a div for the report
])

# Define the clustering function
def perform_clustering(df, columns, num_clusters):
    # Select the specified columns
    selected_data = df[columns]
    
    # Perform clustering (example: K-Means)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(selected_data)
    df['cluster'] = kmeans.labels_
    
    # Calculate cluster statistics
    cluster_stats = []
    for cluster_label in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster_label]
        cluster_center = kmeans.cluster_centers_[cluster_label]
        radius = np.max(np.linalg.norm(cluster_data[columns] - cluster_center, axis=1))
        density = len(cluster_data) / (np.pi * radius**2)
        cluster_stats.append({'Cluster': cluster_label, 'Radius': radius, 'Density': density})
    
    # Create a scatter plot with cluster boundaries
    fig = px.scatter(df, x=columns[0], y=columns[1], color='cluster')
    for cluster_label in range(num_clusters):
        cluster_center = kmeans.cluster_centers_[cluster_label]
        fig.add_shape(type='circle',
                      x0=cluster_center[0] - cluster_stats[cluster_label]['Radius'],
                      y0=cluster_center[1] - cluster_stats[cluster_label]['Radius'],
                      x1=cluster_center[0] + cluster_stats[cluster_label]['Radius'],
                      y1=cluster_center[1] + cluster_stats[cluster_label]['Radius'],
                      line=dict(color='black'))
    
    return fig, cluster_stats

# Define a function to generate a clustering report
def generate_clustering_report(cluster_stats):
    report = html.Div([
        html.H2("Clustering Report"),
        html.Table([
            html.Tr([html.Th("Cluster"), html.Th("Radius"), html.Th("Density")]),
            *[html.Tr([html.Td(cluster['Cluster']), html.Td(cluster['Radius']), html.Td(cluster['Density'])]) for cluster in cluster_stats]
        ])
    ])
    return report

# Define a callback function to update the summary report and display the report
@app.callback(
    Output('summary-report', 'figure'),
    Output('clustering-report', 'children'),  # Output for the report
    Input('analysis-technique', 'value'),
    Input('clustering-columns', 'value')
)
def update_summary_report(technique, selected_columns):
    if technique == 'clustering':
        num_clusters = 2  # You can change the number of clusters as needed
        fig, cluster_stats = perform_clustering(data, selected_columns, num_clusters)
        report = generate_clustering_report(cluster_stats)  # Generate the report
        return fig, report
    elif technique == 'classification':
        # Add your classification function and report generation here
        pass
    else:
        return None, None

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
