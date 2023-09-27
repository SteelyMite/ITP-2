import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import temp  # Import the module containing FunctionName

app = dash.Dash(__name__)

# Define the layout of the web application
app.layout = html.Div([
    html.H1("Testing Output"),
    
    # Button to trigger the update
    html.Button("Update Page", id="update-button"),
    
    html.Div(id='output-html'),
    
    html.Div(id='output-plots'),
])

@app.callback(
    [Output('output-html', 'children'), Output('output-plots', 'children')],
    [Input('update-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_output(n_clicks):
    if n_clicks is None:
        return dash.no_update
    
    # Call the FunctionName() method from your module
    # Replace 'your_module' with the actual module name
    df = pd.read_csv('iris.csv')
    selectedColumns = ["Column One", "Column Three"]
    numClusters = 3
    print(df.columns)
    # plot_list,html_obj, = temp.clustering_KMeans(df, selectedColumns, numClusters)
    plot_list,html_obj, = temp.classification_SVM(df, selectedColumns, "Label")
    # Display the HTML object
    html_output = html_obj

    # Display the list of Dash Plots
    plot_outputs = [dcc.Graph(figure=plot) for plot in plot_list]
    return html_output, plot_outputs

if __name__ == '__main__':
    app.run_server(debug=True)
