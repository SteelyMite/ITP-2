# Import necessary libraries
import dash
from dash import dcc, html
import plotly.graph_objs as go
from sklearn.datasets import make_blobs

# Create artificial clustered data
X, y = make_blobs(
    n_samples=200, n_features=2, centers=4, cluster_std=1.0, random_state=0
)

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Dash Application'),

    # First Graph: Bar Chart
    dcc.Graph(
        id='bar-chart',
        figure={
            'data': [
                go.Bar(
                    x=['A', 'B', 'C'],
                    y=[20, 45, 30]
                )
            ],
            'layout': go.Layout(
                title='Bar Plot',
                xaxis={'title': 'Category'},
                yaxis={'title': 'Values'},
            )
        }
    ),

    # Second Graph: Scatter Plot for Clustered Data
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=X[y == i, 0],
                    y=X[y == i, 1],
                    mode='markers',
                    name=f'Cluster {i}'
                ) for i in range(4)  # we have 4 clusters
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
