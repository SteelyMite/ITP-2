"""
File:           data_management_callbacks.py
Description:    This module defines the callback functionalities related to data management 
                for the Dash web application. The main features encompassed are:
                - Handling file uploads.
                - Displaying uploaded data in a table.
                - Handling editing of the displayed data.
                
Authors:        Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Last Updated:   2023-10-18
"""

# Library imports
import dash
from dash import Input, Output, html, dash_table
import pandas as pd
import base64
import io
from dash.dependencies import Input, Output, State, ALL
from app_instance import app
from utils import parse_contents
from state_saving_func import *

# Callback to handle the upload functionality, display the uploaded data in a table,
# and store the data for future use.
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
    # Record the source code and user action for auditing or versioning.
    add_callback_source_code(update_output)

    # If no content is uploaded, prevent further execution.
    if contents is None:
        raise dash.exceptions.PreventUpdate

    try:
        # Parse the uploaded content based on the selected file type.
        df = parse_contents(contents, file_type)

        # Log the user's action of uploading a file.
        command = f"Uploaded file: {filename}" 
        log_user_action(command)

        # Convert the parsed data into a DataTable for visualization.
        data_table = dash_table.DataTable(
            id='table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
            style_table={'overflowX': 'auto'},
            editable=True,
            page_size=20
        )
        return data_table, "", df.to_dict('records')
    
    # Handle potential errors that may arise during the parsing process.
    except ValueError as e:
        return html.Div(), str(e), {}

# Callback to handle the modification of data directly from the DataTable.
@app.callback(
    Output('stored-data', 'data'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def update_stored_data(edited_data):
    # Record the source code and user action for auditing or versioning.
    add_callback_source_code(update_stored_data)
    
    return edited_data
