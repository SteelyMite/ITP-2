import dash
from dash import dash_table
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd

app = dash.Dash(__name__)

# Example DataFrame stored in dcc.Store
df = pd.DataFrame({
    'numeric_col': [1, 2, 3],
    'string_col': ['a', 'b', 'c'],
    'numeric_col2': [4, 5, 6],
    'string_col2': ['d', 'e', 'f'],
    'string_col3': ['g', 'h', 'i'],
    'numeric_col3': [7, 8, 9]
})

app.layout = html.Div([
    dcc.Store(id='df-store', data=df.to_dict(orient='split')),
    dcc.Store(id='input-dict-store', data=[
        
        {
            "Input": "number_selection",
            "DataType": "numeric",
            "Text": "4. Choose a number 2-5:",
            "AcceptableValues": {"min": 2, "max": 5}
        },
        {
            "Input": "column_selection",
            "DataType": "both",
            "Text": "3. Choose 3 columns:",
            "AcceptableValues": 3
        },
        {
            "Input": "column_selection",
            "DataType": "string",
            "Text": "2. Choose 2 columns:",
            "AcceptableValues": 2
        },
        {
            "Input": "dropdown",
            "DataType": "numeric",
            "Text": "1. Choose a parameter:",
            "AcceptableValues": ["linear", "multilinear", "polynomial"]
        },
        {
            "Input": "number_selection",
            "DataType": "numeric",  
            "Text": "5. Select a number up to 10:",
            "AcceptableValues": {"min": None, "max": 10}
        },
        {
            "Input": "number_selection",
            "DataType": "numeric",  
            "Text": "6. Select any number:",
            "AcceptableValues": {"min": None, "max": None}
        },
        {
            "Input": "column_selection",
            "DataType": "both",  
            "Text": "7. Choose 1-3 columns:",
            "AcceptableValues": {"min": 1, "max": 3}
        },
        {
            "Input": "column_selection",
            "DataType": "both",  
            "Text": "8. Choose 1 or more columns:",
            "AcceptableValues": {"min": 1, "max": None}
        },
        {
            "Input": "column_selection",
            "DataType": "both",  
            "Text": "8. Choose up to 3 columns:",
            "AcceptableValues": {"min": None, "max": 3}
        }
    ]),
    html.Button('Start', id='start-button'),
    dash_table.DataTable(
        id='table',
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
        style_table={'overflowX': 'auto'},
        editable=True,
        page_size=20
    ),
    
    html.Div(id='dynamic-input-div'),
    html.Button('Perform', id='perform-button'),
    html.Div(id='perform-result-div', style={'color': 'green', 'margin-top': '10px'}),

])

@app.callback(
    Output('dynamic-input-div', 'children'),
    [Input('start-button', 'n_clicks')],
    [State('input-dict-store', 'data'),
     State('df-store', 'data')]
)
def generate_input(n_clicks, input_dicts, df_data):
    # Only generate inputs when the button is clicked
    if n_clicks is None or n_clicks == 0:
        return []
    
    df = pd.DataFrame(**df_data)
    components = []
    
    for input_dict in input_dicts:
        text_display = input_dict.get("Text", "")
        input_type = input_dict.get("Input", "")
        acceptable_value = input_dict.get("AcceptableValues", [])
        data_type = input_dict.get("DataType", "")
        
        components.append(html.H3(text_display))
        
        if input_type == "dropdown":
            components.append(dcc.Dropdown(
                options=[{'label': val, 'value': val} for val in acceptable_value],
                value=acceptable_value[0] if acceptable_value else None
            ))
            components.append(html.Button('Test', id={'type': 'test-button', 'index': input_dicts.index(input_dict)}))
            components.append(html.Div(id={'type': 'test-result-div', 'index': input_dicts.index(input_dict)}))
        elif input_type == "number_selection":
            min_val = acceptable_value.get("min")
            max_val = acceptable_value.get("max")
            
            components.append(dcc.Input(
                type='number',
                min=min_val,
                max=max_val,
                placeholder=f"Enter a number{' ≥ '+str(min_val) if min_val is not None else ''}{' ≤ '+str(max_val) if max_val is not None else ''}"
            ))
            components.append(html.Button('Test', id={'type': 'test-button', 'index': input_dicts.index(input_dict)}))
            components.append(html.Div(id={'type': 'test-result-div', 'index': input_dicts.index(input_dict)}))
        elif input_type == "column_selection":
            if data_type == "numeric":
                options = [{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns]
            elif data_type == "string":
                options = [{'label': col, 'value': col} for col in df.select_dtypes(include='object').columns]
            else:  # both
                options = [{'label': col, 'value': col} for col in df.columns]
            
            components.append(dcc.Dropdown(
                options=options,
                multi= True
            ))
            components.append(html.Button('Test', id={'type': 'test-button', 'index': input_dicts.index(input_dict)}))
            components.append(html.Div(id={'type': 'test-result-div', 'index': input_dicts.index(input_dict)}))
        else:
            components.append(html.P("Invalid input type"))
        
        
    return components

@app.callback(
    Output({'type': 'test-result-div', 'index': dash.dependencies.MATCH}, 'children'),
    [Input({'type': 'test-button', 'index': dash.dependencies.MATCH}, 'n_clicks')],
    [State({'type': 'test-button', 'index': dash.dependencies.MATCH}, 'id'),
     State('dynamic-input-div', 'children'),
     State('input-dict-store', 'data')]
)
def test_input(n_clicks, button_id, dynamic_input_children, input_dicts):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Get the index of the clicked button
    idx = button_id['index']
    
    # Get the corresponding input_dict
    input_dict = input_dicts[idx]
    
    # Get the corresponding input component
    input_component = dynamic_input_children[idx * 4 + 1]  # Adjust the index to get the correct component
    
    # Check the input
    input_type = input_dict.get("Input", "")
    acceptable_value = input_dict.get("AcceptableValues", [])
    
    print(f"Testing input {idx}: {input_type}, {acceptable_value}, component: {input_component}")  # Logging
    
    # Check if 'value' key exists in the component's props
    if 'value' not in input_component['props']:
        print(f"No value in component {idx}: {input_component}")  # Logging
        return "False"
    
    if input_type == "dropdown" and input_component['props']['value'] is not None:
        return "True"
    elif input_type == "number_selection" and input_component['props']['value'] is not None:
        return "True"
    elif input_type == "column_selection" and input_component['props']['value'] is not None:
        num_selected_columns = len(input_component['props']['value'])
        
        # Check if acceptable_value is an integer or a dictionary
        if isinstance(acceptable_value, int):
            # If it's an integer, check for exact match
            return "True" if num_selected_columns == acceptable_value else "False"
        elif isinstance(acceptable_value, dict):
            # If it's a dictionary, check for range match
            min_columns = acceptable_value.get("min", 0)  # Default to 0 if not provided
            max_columns = acceptable_value.get("max", num_selected_columns)  # Default to num_selected_columns if not provided
            
            # If min or max is None, replace with num_selected_columns for no limit
            min_columns = num_selected_columns if min_columns is None else min_columns
            max_columns = num_selected_columns if max_columns is None else max_columns
            
            return "True" if min_columns <= num_selected_columns <= max_columns else "False"
    else:
        return "False"

@app.callback(
    Output('perform-result-div', 'children'),
    [Input('perform-button', 'n_clicks')],
    [State('dynamic-input-div', 'children'),
     State('input-dict-store', 'data')]
)
def perform_operation(n_clicks, dynamic_input_children, input_dicts):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Loop through all input components and 'input-dict-store' data
    for idx, input_dict in enumerate(input_dicts):
        # Get the corresponding input component
        input_component = dynamic_input_children[idx * 4 + 1]  # Adjust the index to get the correct component
        
        # Check the input
        input_type = input_dict.get("Input", "")
        acceptable_value = input_dict.get("AcceptableValues", [])
        
        # Check if 'value' key exists in the component's props
        if 'value' not in input_component['props'] or input_component['props']['value'] is None:
            return "Missing input or something, try again."
        
        if input_type == "column_selection":
            num_selected_columns = len(input_component['props']['value'])
            
            # Check if acceptable_value is an integer or a dictionary
            if isinstance(acceptable_value, int):
                # If it's an integer, check for exact match
                if num_selected_columns != acceptable_value:
                    return "Missing input or something, try again."
            elif isinstance(acceptable_value, dict):
                # If it's a dictionary, check for range match
                min_columns = acceptable_value.get("min", 0)  # Default to 0 if not provided
                max_columns = acceptable_value.get("max", num_selected_columns)  # Default to num_selected_columns if not provided
                
                # If min or max is None, replace with num_selected_columns for no limit
                min_columns = num_selected_columns if min_columns is None else min_columns
                max_columns = num_selected_columns if max_columns is None else max_columns
                
                if not (min_columns <= num_selected_columns <= max_columns):
                    return "Missing input or something, try again."
    
    # Perform your desired operations here
    # ...
    return "Operation performed successfully!"





if __name__ == '__main__':
    app.run_server(debug=True)