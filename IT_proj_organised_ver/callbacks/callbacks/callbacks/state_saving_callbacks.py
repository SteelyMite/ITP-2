import dash
from dash import Input, Output, html, dash_table, dcc
import pandas as pd
import base64
import io
from dash.dependencies import Input, Output, State, ALL
from app_instance import app
from state_saving_func import *

@app.callback(
    Output('action-list', 'children'),
    Input('update-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_action_list(n_intervals):
    action_items = [dbc.Card(html.Li(f"{i + 1}: {action}", style={'padding': '10px'})) for i, action in enumerate(user_actions)]
    return action_items

@app.callback(
    Output('download', 'data', allow_duplicate=True),
    [Input('save-button', 'n_clicks'), Input('export-commands-button', 'n_clicks')],
    [State('table', 'data'), State('export-format-dropdown', 'value')],
    prevent_initial_call=True
)
def export_or_save(n_clicks_save, n_clicks_export, rows, export_format):

    add_callback_source_code(export_or_save)

    if n_clicks_save is None and n_clicks_export is None:
        raise dash.exceptions.PreventUpdate

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'save-button':
        if n_clicks_save is None:
            raise dash.exceptions.PreventUpdate

        df_to_save = pd.DataFrame(rows)
        command = f"Saved data to {export_format} file"  # Define the command for saving the data

        if export_format == 'csv':
            log_user_action(command, "edited_data.csv")  # Log the user action with the command and filename
            csv_string = df_to_save.to_csv(index=False, encoding='utf-8')
            return dict(content=csv_string, filename="edited_data.csv")
        elif export_format == 'xlsx':
            log_user_action(command, "edited_data.xlsx")  # Log the user action with the command and filename
            xlsx_io = io.BytesIO()
            df_to_save.to_excel(xlsx_io, index=False, engine='openpyxl')
            xlsx_io.seek(0)
            # Encode the Excel data to base64
            xlsx_base64 = base64.b64encode(xlsx_io.getvalue()).decode('utf-8')
            return dict(content=xlsx_base64, filename="edited_data.xlsx",
                        type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base64=True)
        elif export_format == 'json':
            log_user_action(command, "edited_data.json")  # Log the user action with the command and filename
            json_string = df_to_save.to_json(orient='records')
            return dict(content=json_string, filename="edited_data.json")

    elif trigger_id == 'export-commands-button':
        if n_clicks_export is None:
            raise dash.exceptions.PreventUpdate

        # Define the filename for the .py file where user actions will be saved
        filename = "user_actions.py"

        # Call the function to save user actions to the .py file
        write_callback_functions_to_file(filename)

        # Return the .py file for download
        with open(filename, "r") as file:
            file_content = file.read()

        return dict(content=file_content, filename=filename, type="text/python")


    raise dash.exceptions.PreventUpdate  # Handle unexpected situations