import base64
import io
import dash
import plotly.express as px
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
                    value='csv',
                    style={
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'width': '100%',
                        'height': '60px',
                        'borderWidth': '1px',
                        'borderRadius': '5px',
                        'margin': '10px'
                    }
                ),
                html.Div(id='upload-container'),  # This will house the Upload component
                dash_table.DataTable(id='datatable-upload-container')
            ], width=2),
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
            ], width=4),
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
                    style={
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'width': '100%',
                        'height': '60px',
                        'borderWidth': '1px',
                        'borderRadius': '5px',
                        'margin': '10px'
                    }
                ),
            ], width=2),
            dbc.Col([
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
            ], width=4)    
        ]),

        # table width slider
        dbc.Row([
            dbc.Col([
                html.Label('Set Table Width (in %):'),
                dcc.Slider(id='width-slider', min=10, max=100, value=50, step=1, marks={i: str(i) for i in range(10, 101, 10)}),
            ], width=12)
        ]),
        html.Div(id='output-data-upload'),
        html.Div(id='summary-output'),
        html.Div(id='intermediate-div', style={'display': 'none'})
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

def generate_column_summary_box(df, column_name):
    # Calculate statistics
    nan_count = df[column_name].isna().sum()

    if pd.api.types.is_numeric_dtype(df[column_name]):
        # Generate bar graph for numeric columns
        fig = px.bar(df, x=column_name)
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        mean_val = df[column_name].mean()

        return dbc.Card([
            dbc.CardHeader(column_name),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '150px'}),
                html.P(f"NaN values: {nan_count}"),
                html.P(f"Min: {min_val}"),
                html.P(f"Max: {max_val}"),
                html.P(f"Mean: {mean_val}")
            ])
        ])

    else:
        # For non-numeric columns, display NaN values, most frequent string, and least frequent string
        fig = px.pie(df, names=column_name)
        value_counts = df[column_name].value_counts()
        most_frequent_string = value_counts.index[0] if not value_counts.empty else "N/A"
        least_frequent_string = value_counts.index[-1] if not value_counts.empty else "N/A"
        
        return dbc.Card([
            dbc.CardHeader(column_name),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '150px'}),
                html.P(f"NaN values: {nan_count}"),
                html.P(f"Most Frequent: {most_frequent_string}"),
                html.P(f"Least Frequent: {least_frequent_string}")
            ])
        ])

@app.callback(
    [Output('output-data-upload', 'children'), Output('summary-output', 'children')],
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
                page_size=15
            )
            
            # Generate the summary boxes
            summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

            # Arrange the summary boxes in a 2-box-per-row layout
            rows = []
            for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
                box1 = summary_boxes[i]
                # Check if there's a second box in the pair, if not, just use an empty Div
                box2 = summary_boxes[i+1] if (i+1) < len(summary_boxes) else html.Div()
                row = dbc.Row([
                    dbc.Col(box1, width=6),
                    dbc.Col(box2, width=6)
                ])
                rows.append(row)

            return data_table, rows
        
    return None, None
        
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
    Output('intermediate-div', 'children'),
    Input('width-slider', 'value'),
    prevent_initial_call=True
)
def update_table_width(width_value):
    return f'{width_value}%'
    
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
