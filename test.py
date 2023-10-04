import dash
from dash import dcc, html

app = dash.Dash(__name__)

# Define numeric options for the dropdown
# numeric_options = [{'label': str(i), 'value': i} for i in range(1, 11)]  # Creates options from 1 to 10
numeric_options = [1,"test",3,4,5]  # Creates options from 1 to 10

app.layout = html.Div([
    
    dcc.Dropdown(
        id='numeric-dropdown',
        options=numeric_options,
        value=1  # Default selected value
    ),
    html.Div(id='output-div')
])

@app.callback(
    dash.dependencies.Output('output-div', 'children'),
    [dash.dependencies.Input('numeric-dropdown', 'value')]
)
def update_output(selected_value):
    return f'You selected: {selected_value}'

if __name__ == '__main__':
    app.run_server(debug=True)
