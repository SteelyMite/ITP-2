# Import necessary libraries
import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from sklearn.cluster import KMeans

# Load your data
df = pd.read_csv('your_data.csv')

# Assuming your data has two features named 'feature1' and 'feature2'
# Extract features
X = df[['feature1', 'feature2']].values

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
y = kmeans.labels_

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Dash Application'),

    # Scatter Plot for Clustered Data
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=X[y==cluster, 0],
                    y=X[y==cluster, 1],
                    mode='markers',
                    name=f'Cluster {cluster}'
                ) for cluster in range(4)  # we have 4 clusters
            ],
            'layout': go.Layout(
                title='Cluster Plot',
                xaxis={'title': 'Feature 1'},
                yaxis={'title': 'Feature 2'},
                hovermode='closest'
            )
        }
    )
])

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
