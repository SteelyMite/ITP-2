import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Read the CSV data into a DataFrame
df = pd.read_csv('social_capital.csv')

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

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
