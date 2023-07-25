import base64
import io
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from app import app

layout = html.Div([
    dbc.Container([
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
        html.Div(id='output-data-upload'),
        dcc.Store(id='storage'),  # This is where the data is stored
        dbc.Button('Visualize Data', id='btn-visualize', href='/visualization', className='mt-3', n_clicks=0),
    ])
])

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

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = []
        for contents, name in zip(list_of_contents, list_of_names):
            df = parse_contents(contents, name)
            if df is not None:
                children.append(
                    dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in df.columns],
                        style_table={'overflowX': 'auto'}
                    )
                )
                # Store the DataFrame as a JSON string in the 'storage' component
                app.clientside_callback(
                    """
                    function(data) {
                        return JSON.stringify(data);
                    }
                    """,
                    Output('storage', 'data'),
                    Input('upload-data', 'contents'),
                    prevent_initial_call=True
                )
        return children
