# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import base64
import io
import pandas as pd
from app import app

# Initialize an empty DataFrame
df = pd.DataFrame()

# Existing layout and callback code...

# Update DataTable component in the layout
layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records') if not df.empty else None,  # Added check if df is not empty
        style_header={
            'backgroundColor': 'rgb(23, 103, 240)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(23, 103, 240)'
            }
        ],
        style_cell={
            'textAlign': 'left',
            'minWidth': '40px', 'width': '40px', 'maxWidth': '60px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
    )
])


@app.callback(
    Output('table', 'data'),
    Output('table', 'columns'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_output(list_of_contents, list_of_filenames):
    if list_of_contents is not None:
        for contents, filename in zip(list_of_contents, list_of_filenames):
            df = parse_contents(contents, filename)
            if df is not None:
                data = df.to_dict('records')
                columns = [{"name": i, "id": i} for i in df.columns]
                return data, columns
    return [], []

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in filename:
            df = pd.read_json(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df
