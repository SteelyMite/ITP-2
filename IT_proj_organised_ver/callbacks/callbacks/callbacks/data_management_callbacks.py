from dash import Input, Output, html, dash_table
import pandas as pd
import base64
import io
from dash.dependencies import Input, Output, State, ALL

from app_instance import app
from utils import parse_contents

@app.callback(
    [Output('output-data-upload', 'children'), Output('stored-data', 'data', allow_duplicate=True)],
    Input('upload-data', 'contents'),
    State('file-type-dropdown', 'value'),
    prevent_initial_call=True
)
def update_output(contents, file_type):
    if contents is not None:
        df = parse_contents(contents, file_type)
        if df is not None:
            # Generate the datatable
            data_table = dash_table.DataTable(
                id='table',
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
                style_table={'overflowX': 'auto'},
                editable=True,
                page_size=20
            )
            return data_table, df.to_dict('records')
        
    # Returning an empty div to ensure no data is displayed if conditions aren't met.
    return html.Div(), {}

@app.callback(
    Output('stored-data', 'data'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def update_stored_data(edited_data):
    return edited_data

@app.callback(
    Output('download', 'data'),
    Input('save-button', 'n_clicks'),
    State('stored-data', 'data'),  
    State('export-format-dropdown', 'value'),
    prevent_initial_call=True
)
def save_to_file(n_clicks, stored_data, export_format):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    df_to_save = pd.DataFrame(stored_data)  # Convert stored_data to DataFrame
    if export_format == 'csv':
        csv_string = df_to_save.to_csv(index=False, encoding='utf-8')
        return dict(content=csv_string, filename="edited_data.csv")
    elif export_format == 'xlsx':
        xlsx_io = io.BytesIO()
        df_to_save.to_excel(xlsx_io, index=False, engine='openpyxl')
        xlsx_io.seek(0)
        # Encode the Excel data to base64
        xlsx_base64 = base64.b64encode(xlsx_io.getvalue()).decode('utf-8')
        return dict(content=xlsx_base64, filename="edited_data.xlsx", type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base64=True)
    elif export_format == 'json':
        json_string = df_to_save.to_json(orient='records')
        return dict(content=json_string, filename="edited_data.json")

