import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

# Sample DataFrame for testing (replace with your actual DataFrame)
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 5, 2]})

# Define the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Dynamic Dash App with Variable Plots"),
    
    html.Div(id='content'),
    
    html.Button("Update Content", id='update-button'),
    
    # Use dcc.Store to store the variable-length data
    dcc.Store(id='plot-data-store'),
    
    # Use a callback to update plots dynamically
    dcc.Loading(
        id="loading",
        type="default",
        children=[html.Div(id="dynamic-plots")]
    )
])

# Define the function to update the page content and generate variable-length plots
def UpdatePage():
    # Generate new content for the div (replace with your logic)
    new_content = html.Div([
        html.P("Updated content here."),
        html.P("More updated content."),
    ])
    
    # Generate variable-length plots (replace with your logic)
    plot_data = []
    # Example: Generate random variable-length plots
    for i in range(1, 5):
        data = df.sample(i)  # Sample a subset of data
        plot = px.scatter(data, x='x', y='y', title=f'Scatter Plot {i}')
        plot_data.append(plot)
    
    return new_content, plot_data

# Define a callback to update the content and store the variable-length plots
@app.callback(
    [Output('content', 'children'),
     Output('plot-data-store', 'data')],
    Input('update-button', 'n_clicks')
)
def update_content_and_store_plots(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update
    
    content, plots = UpdatePage()
    return content, plots

# Define a callback to generate the variable-length plots
@app.callback(
    Output('dynamic-plots', 'children'),
    Input('plot-data-store', 'data')
)
def update_dynamic_plots(plot_data):
    if not plot_data:
        return []
    
    # Create dcc.Graph components for the variable-length plots
    dynamic_plots = [dcc.Graph(figure=plot) for plot in plot_data]
    return dynamic_plots

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
