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
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='file-type-dropdown',
                    options=[
                        {'label': 'CSV', 'value': 'csv'},
                        {'label': 'Excel', 'value': 'excel'},
                        {'label': 'JSON', 'value': 'json'}
                    ],
                    placeholder="Select file type...",
                    value=None
                ),
                html.Div(id='upload-container'),  # This will house the Upload component
                dash_table.DataTable(id='datatable-upload-container')
            ], width=6),
            dbc.Col([
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
                    multiple=False
                ),
                ##dash_table.DataTable(id='datatable-upload-container')
            ], width=6),
            dbc.Col([
                dcc.Dropdown(
                    id='export-format-dropdown',
                    options=[
                        {'label': 'CSV', 'value': 'csv'},
                        {'label': 'Excel', 'value': 'xlsx'},
                        {'label': 'JSON', 'value': 'json'}
                    ],
                    value='csv',
                    clearable=False,
                    style={'marginBottom': '10px'}
                ),
                html.Button('Save', id='save-button', style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'background': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'cursor': 'pointer',
                }),
                dcc.Download(id="download")
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label('Data Cleaning Options:'),
                dcc.Dropdown(
                    id='data-cleaning-dropdown',
                    options=[
                        {'label': 'Remove Duplicates', 'value': 'remove_duplicates'},
                        {'label': 'Handle Missing Values', 'value': 'handle_missing'}
                    ],
                    placeholder="Select an option",
                    style={'width': '80%', 'margin': 'auto'}
                ),
            ], className='text-center')
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='column-selector',
                    # Options would be dynamically set based on your DataFrame
                    options=[],
                    placeholder="Select a column...",
                    value=None,
                    multi=True
                ),
            ], width=6),
            dbc.Col([
                dcc.Dropdown(
                    id='missing-value-handler',
                    options=[
                        {'label': 'Replace with Max', 'value': 'max'},
                        {'label': 'Replace with Min', 'value': 'min'},
                        {'label': 'Replace with Mean', 'value': 'mean'},
                        {'label': 'Replace with Zero', 'value': 'zero'},
                    ],
                    placeholder="Handle missing values by...",
                    value=None
                ),
            ], width=6)
        ], className='mb-3'),  # 'mb-3' is just to add some margin at the bottom

        dbc.Row([
            dbc.Col([
                html.Button('Apply Data Cleaning', id='apply-cleaning-btn', className='btn btn-primary'),
            ], width=12, style={'textAlign': 'center'})
        ]),


        dbc.Row([
            dbc.Col([
                html.Label('Set Table Width (in %):'),
                dcc.Slider(id='width-slider', min=10, max=100, value=50, step=1, marks={i: str(i) for i in range(10, 101, 10)}),
            ], width=12)
        ]),
        html.Div(id='output-data-upload'),
        html.Button('Display Summary', id='display-button', style={
                    'width': '30%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'background': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'cursor': 'pointer',
                }),
        html.Div(id='summary-output'),
    ])
])

def parse_contents(contents, file_type):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if file_type == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif file_type == 'excel':
            df = pd.read_excel(io.BytesIO(decoded))
        elif file_type == 'json':
            df = pd.read_json(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

def generate_summary_table(df):
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    summary_stats = {
        'Min': numerical_df.min(),
        'Max': numerical_df.max(),
        'Mean': numerical_df.mean(),
        'Mode': numerical_df.mode().iloc[0]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_table = dash_table.DataTable(
        data=summary_df.reset_index().to_dict('records'),
        columns=[{'name': i, 'id': i} for i in ['index'] + list(summary_stats.keys())],
        style_table={'overflowX': 'auto'},
        page_size=15 # Sets the number of rows per page
    )
    return summary_table

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('file-type-dropdown', 'value'),   # Use the dropdown value as a state 
    prevent_initial_call=True
)
def update_output(contents, file_type):    # Receive the file_type as a parameter
    if contents is not None:
        df = parse_contents(contents, file_type)  # Use the file_type instead of name
        if df is not None:
            return dash_table.DataTable(
                id='table',
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
                style_table={'overflowX': 'auto'},
                editable=True,
                page_size=15
            )
        
@app.callback(
    Output('download', 'data'),
    [Input('save-button', 'n_clicks')],
    [State('table', 'data'), State('export-format-dropdown', 'value')],
    prevent_initial_call=True
)
def save_to_file(n_clicks, rows, export_format):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    df_to_save = pd.DataFrame(rows)
    
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


@app.callback(
    Output('summary-output', 'children'),
    Input('display-button', 'n_clicks'),
    State('table', 'data'),
    prevent_initial_call=True
)
def display_summary(n_clicks, data):
    if data is not None:
        df = pd.DataFrame(data)
        return generate_summary_table(df)
    return None

@app.callback(
    [Output('output-data-upload', 'style'),
     Output('summary-output', 'style')],
    Input('width-slider', 'value'),
    prevent_initial_call=True
)
def update_table_width(width_value):
    style_dict = {'width': f'{width_value}%', 'margin': 'auto'}
    return style_dict, style_dict
    
@app.callback(
    Output('table', 'data'),
    Input('apply-cleaning-btn', 'n_clicks'),
    State('data-cleaning-dropdown', 'value'),
    State('table', 'data'),
    State('column-selector', 'value'),
    State('missing-value-handler', 'value'),
    prevent_initial_call=True
)
def unified_cleaning(n_clicks, cleaning_option, table_data, selected_columns, handler):
    df = pd.DataFrame(table_data)

    if cleaning_option == 'remove_duplicates':
        df.drop_duplicates(inplace=True)
        
    elif cleaning_option == 'handle_missing' and selected_columns and handler:
        for col in selected_columns:  # Iterate over each selected column
            if handler == 'max':
                replacement = df[col].max()
            elif handler == 'min':
                replacement = df[col].min()
            elif handler == 'mean':
                replacement = df[col].mean()
            elif handler == 'zero':
                replacement = 0
            df[col].fillna(replacement, inplace=True)

    return df.to_dict('records')

@app.callback(
    Output('column-selector', 'options'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def set_column_options(table_data):
    if table_data:
        df = pd.DataFrame(table_data)
        return [{'label': col, 'value': col} for col in df.columns]
    return []
