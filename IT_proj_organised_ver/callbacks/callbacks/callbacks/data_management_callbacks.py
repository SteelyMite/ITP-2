from dash import Input, Output, html, dash_table
import pandas as pd
import base64
import io
from dash.dependencies import Input, Output, State, ALL

from app_instance import app
from utils import parse_contents
from state_saving_func import *

@app.callback(
    [Output('output-data-upload', 'children'), 
     Output('error-message', 'children'), 
     Output('stored-data', 'data', allow_duplicate=True)],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('file-type-dropdown', 'value'),
    prevent_initial_call=True
)
def update_output(contents, filename, file_type):
    add_callback_source_code(update_output)
    if contents is None:
        raise dash.exceptions.PreventUpdate

    try:
        df = parse_contents(contents, file_type)
        command = f"Uploaded file: {filename}"  # Include the filename in the command
        log_user_action(command)
        data_table = dash_table.DataTable(
            id='table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
            style_table={'overflowX': 'auto'},
            editable=True,
            page_size=20
        )
        return data_table, "", df.to_dict('records')
    except ValueError as e:
        return html.Div(), str(e), {}

@app.callback(
    Output('stored-data', 'data'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def update_stored_data(edited_data):
    add_callback_source_code(update_stored_data)
    return edited_data


