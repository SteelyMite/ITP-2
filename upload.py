import base64
import io
import dash
import json
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
from app import app
import os

layout = html.Div([
    dcc.Store(id='stored-data'),  # store the uploaded data
    dbc.Container([
        dbc.Tabs([
            # Tab 1: Import, Export, DataFrame Display
            dbc.Tab(label='Data Management', children=[
                dbc.Row([
                    # Import File Type Dropdown
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
                                'height': '40px',
                                'borderWidth': '1px',
                                'borderRadius': '5px',
                                'margin': '10px'
                            }
                        ),
                    ], width=3),
                    # Upload Data Section
                    dbc.Col([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '40px',
                                'lineHeight': '40px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                    ], width=3),
                    # Export File Type Dropdown
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
                                'height': '40px',
                                'borderWidth': '1px',
                                'borderRadius': '5px',
                                'margin': '10px'
                            }
                        ),
                    ], width=3),
                    # Save Button and Download Component
                    dbc.Col([
                        html.Button('Save', id='save-button', style={
                            'width': '100%',
                            'height': '40px',
                            'lineHeight': '40px',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'background': '#007bff',
                            'color': 'white',
                            'border': 'none',
                            'cursor': 'pointer',
                        }),
                        dcc.Download(id="download")
                    ], width=3)
                ]),
                # DataTable to Display Uploaded Data
                dash_table.DataTable(id='datatable-upload-container'),
                html.Div(id='output-data-upload'),
            ]),

            # Tab 2: Statistical Summary
            dbc.Tab(label='Statistical Summary', children=[
                html.Div(id='summary-output'),
            ]),
            # Tab 3: Visualisation Summary
            dbc.Tab(label='Visualisation', children=[
                # selection
                dbc.Row([
                    dbc.Col([
                        html.Label('Select X-axis column:'),
                        dcc.Dropdown(
                            id='xaxis-viscolumn-dropdown',
                            options=[],
                            placeholder="Select a column for X-axis...",
                            value=None
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label('Select Y-axis column:'),
                        dcc.Dropdown(
                            id='yaxis-viscolumn-dropdown',
                            options=[],
                            placeholder="Select a column for Y-axis...",
                            value=None
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label('Select visualization type:'),
                        dcc.Dropdown(
                            id='visualization-type-dropdown',
                            options=[
                                {'label': 'Scatter Plot', 'value': 'scatter'},
                                {'label': 'Line Plot', 'value': 'line'},
                                {'label': 'Bar Chart', 'value': 'bar'},
                                {'label': 'Pie Chart', 'value': 'pie'},
                                {'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': '3D Scatter Plot', 'value': '3dscatter'},
                                {'label': 'Area Plot', 'value': 'area'},
                                {'label': 'Violin Plot', 'value': 'violin'}
                            ],
                            placeholder="Select a type...",
                            value=None
                        ),
                    ], width=6)
                ], className='mb-4'),  # spacing

                # Visualization
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='visualisation-graph')  # display the selected visualization
                    ])
                ], className='mb-4'),
                dbc.Row([
                    dbc.Col([
                    ])
                ], className='mb-4'),
                html.Button('Save Graph', id='save-graph-button', className='mt-3 mb-4'),
                html.Div(id='saved-visgraphs-container')

            ]),
            # //////////////////
            # //////////////////
            # //////////////////

            # Tab 4: Analytics Summary
            dbc.Tab(label='Analytics', children=[
                html.H5("Choose Data Analysis Method", className="mb-4"),
                dcc.Dropdown(
                    id='data-analysis-dropdown',
                    options=[
                        {'label': 'Clustering', 'value': 'clustering'},
                        {'label': 'Classification', 'value': 'classification'}
                    ],
                    placeholder="Select an analysis method...",
                    value=None
                ),
                html.Div(id='analysis-content')
            ]),
            dbc.Tab(label='State Summary', children=[
                html.Ul(id='action-list'),
                html.Button('Export Commands to .py File', id='export-commands-button', className='mt-3 mb-4', style={
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'width': '100%',
                    'height': '40px',
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin': '10px'
                }),
                dcc.Interval(
                    id='update-interval',
                    interval=10 * 1000,  # Update every 10 seconds (adjust as needed)
                    n_intervals=0
                )
            ]),
        ]),
    ])
])

user_actions = []

def log_user_action(action, filename=None):
    if filename:
        user_actions.append(f"{action} ({filename})")
    else:
        user_actions.append(action)

@app.callback(
    Output('action-list', 'children'),
    Input('update-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_action_list(n_intervals):
    action_items = [dbc.Card(html.Li(f"{i + 1}: {action}", style={'padding': '10px'})) for i, action in enumerate(user_actions)]
    return action_items


# Define a function to save user actions to a .py file
def save_user_actions_to_py_file(filename):
    with open(filename, "w") as file:
        file.write("# User Actions\n")
        file.write("user_actions = [\n")
        for action in user_actions:
            file.write(f'    "{action}",\n')
        file.write("]\n")


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
        # Generate histogram for numeric columns
        data_type = "Numeric"
        fig = px.histogram(df, x=column_name,
                           nbins=5)  # Here, 5 bins are used for simplicity; you can adjust as needed.
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        mean_val = df[column_name].mean()
        data_cleaning_options = [
            {"label": "Replace NaN with Min", "value": "min"},
            {"label": "Replace NaN with Max", "value": "max"},
            {"label": "Replace NaN with Mean", "value": "mean"},
            {"label": "Replace NaN with Zero", "value": "zero"},
        ]

        dropdown_menu = dbc.DropdownMenu(
            label="Options",
            children=[
                dbc.DropdownMenuItem("Change data type", header=True),
                dbc.DropdownMenuItem("Numeric", id={"type": "convert", "index": column_name, "to": "Numeric"}),
                dbc.DropdownMenuItem("String", id={"type": "convert", "index": column_name, "to": "String"}),

                dbc.DropdownMenuItem(divider=True),

                dbc.DropdownMenuItem("Data Cleaning", header=True),
                *[dbc.DropdownMenuItem(item["label"],
                                       id={"type": "clean", "index": column_name, "action": item["value"]}) for item in
                  data_cleaning_options],
            ],
            className="m-1",
            right=True
        )
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div(column_name),
                    html.Div(f"Data type: {data_type}", style={'fontSize': '12px', 'color': 'grey'})
                ]),
                dropdown_menu
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '250px'}),
                html.P(f"NaN values: {nan_count}"),
                html.P(f"Min: {min_val}"),
                html.P(f"Max: {max_val}"),
                html.P(f"Mean: {mean_val}")
            ])
        ])

    else:
        # For non-numeric columns, show a bar chart of top X most frequent values
        data_type = "String"
        unique_count = df[column_name].nunique()  # Calculate the number of unique strings
        top_values = df[column_name].value_counts().head(10)
        fig = px.bar(top_values, x=top_values.index, y=top_values.values, labels={'x': column_name, 'y': 'Count'})
        value_counts = df[column_name].value_counts()
        most_frequent_string = value_counts.index[0] if not value_counts.empty else "N/A"
        least_frequent_string = value_counts.index[-1] if not value_counts.empty else "N/A"
        data_cleaning_options = [
            {"label": "Replace NaN with N/A", "value": "na_string"},
            {"label": f"Replace NaN with Most Frequent: {most_frequent_string}", "value": "most_frequent"},
        ]

        dropdown_menu = dbc.DropdownMenu(
            label="Options",
            children=[
                dbc.DropdownMenuItem("Change data type", header=True),
                dbc.DropdownMenuItem("Numeric", id={"type": "convert", "index": column_name, "to": "Numeric"}),
                dbc.DropdownMenuItem("String", id={"type": "convert", "index": column_name, "to": "String"}),

                dbc.DropdownMenuItem(divider=True),

                dbc.DropdownMenuItem("Data Cleaning", header=True),
                *[dbc.DropdownMenuItem(item["label"],
                                       id={"type": "clean", "index": column_name, "action": item["value"]}) for item in
                  data_cleaning_options],
            ],
            className="m-1",
            right=True
        )
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div(column_name),
                    html.Div(f"Data type: {data_type}", style={'fontSize': '12px', 'color': 'grey'})
                ]),
                dropdown_menu  # Adding dropdown menu here for consistency
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '250px'}),
                html.P(f"NaN values: {nan_count}"),
                html.P(f"Most Frequent: {most_frequent_string}"),
                html.P(f"Least Frequent: {least_frequent_string}"),
                html.P(f"Unique Strings: {unique_count}")  # Display the number of unique strings
            ])
        ])

@app.callback(
    [Output('output-data-upload', 'children'), Output('stored-data', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),  # Add the 'filename' state
    State('file-type-dropdown', 'value'),
    prevent_initial_call=True
)
def update_output(contents, filename, file_type):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if not filename:
            filename = "Unknown_File"  # If filename cannot be obtained, use a default name

        if file_type == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif file_type == 'excel':
            df = pd.read_excel(io.BytesIO(decoded))
        elif file_type == 'json':
            df = pd.read_json(io.BytesIO(decoded))
        else:
            return None, {}

        command = f"Uploaded file: {filename}"  # Include the filename in the command
        log_user_action(command)  # Log user action with the command and filename

        # Generate the datatable
        data_table = dash_table.DataTable(
            id='table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],
            style_table={'overflowX': 'auto'},
            editable=True,
            page_size=15
        )
        return data_table, df.to_dict('records')

    # Returning an empty div to ensure no data is displayed if conditions aren't met.
    return html.Div(), {}




@app.callback(
    Output('download', 'data'),
    [Input('save-button', 'n_clicks'), Input('export-commands-button', 'n_clicks')],
    [State('table', 'data'), State('export-format-dropdown', 'value')],
    prevent_initial_call=True,
    allow_duplicate=True
)
def export_or_save(n_clicks_save, n_clicks_export, rows, export_format):
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
        save_user_actions_to_py_file(filename)

        # Return the .py file for download
        with open(filename, "r") as file:
            file_content = file.read()

        return dict(content=file_content, filename=filename, type="text/python")

    raise dash.exceptions.PreventUpdate  # Handle unexpected situations

@app.callback(
    Output('table', 'data'),
    [Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    State('table', 'data'),
    prevent_initial_call=True
)
def handle_dropdown_actions(n_clicks_convert, n_clicks_clean, table_data):
    # Get the triggering input (i.e., which dropdown item was clicked)
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Information about the clicked dropdown item
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    prop_info = json.loads(prop_id)

    df = pd.DataFrame(table_data)

    column_name = prop_info['index']

    if prop_info['type'] == "convert":
        if prop_info['to'] == "Numeric":
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        else:
            df[column_name] = df[column_name].astype(str)

        # Log the user action for conversion
        #log_user_action(f"Converted column {column_name} to {prop_info['to']}")

    elif prop_info['type'] == "clean":
        if prop_info['action'] == "min":
            df[column_name].fillna(df[column_name].min(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].min(), inplace=True)")
        elif prop_info['action'] == "max":
            df[column_name].fillna(df[column_name].max(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].max(), inplace=True)")
        elif prop_info['action'] == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
            log_user_action(f"df[{column_name}].fillna(df[{column_name}].mean(), inplace=True)")
        elif prop_info['action'] == "zero":
            df[column_name].fillna(0, inplace=True)
            log_user_action(f"df[{column_name}].fillna(0, inplace=True)")
        elif prop_info['action'] == "na_string":
            df[column_name].fillna('N/A', inplace=True)
            log_user_action(f" df[{column_name}].fillna('N/A', inplace=True)")
        elif prop_info['action'] == "most_frequent":
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].empty else "N/A"
            df[column_name].fillna(most_frequent, inplace=True)
            log_user_action(f"df[{column_name}].fillna({most_frequent}, inplace=True)")

    return df.to_dict('records')

@app.callback(
    Output('summary-output', 'children'),
    [Input('stored-data', 'data'),
     Input({'type': 'convert', 'index': ALL, 'to': ALL}, 'n_clicks'),
     Input({'type': 'clean', 'index': ALL, 'action': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_summary(data, n_clicks_convert, n_clicks_clean):
    df = pd.DataFrame(data)

    # Generate the summary boxes
    summary_boxes = [generate_column_summary_box(df, column_name) for column_name in df.columns]

    # Arrange the summary boxes in a 2-box-per-row layout
    rows = []
    for i in range(0, len(summary_boxes), 2):  # Step by 2 for pairs
        box1 = summary_boxes[i]
        # Check if there's a second box in the pair, if not, just use an empty Div
        box2 = summary_boxes[i + 1] if (i + 1) < len(summary_boxes) else html.Div()
        row = dbc.Row([
            dbc.Col(box1, width=6),
            dbc.Col(box2, width=6)
        ])
        rows.append(row)

    return rows


# VISUALISATION


@app.callback(
    Output('datatable-upload-container', 'data'),
    Output('datatable-upload-container', 'columns'),
    Output('xaxis-viscolumn-dropdown', 'options'),
    Output('yaxis-viscolumn-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('file-type-dropdown', 'value')
)
def update_output(contents, file_type):
    if contents is None:
        return [], [], [], []

    df = parse_contents(contents, file_type)
    if df is None:
        return [], [], [], []

    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')
    options = [{'label': col, 'value': col} for col in df.columns]

    return data, columns, options, options


@app.callback(
    Output('visualisation-graph', 'figure'),
    Input('xaxis-viscolumn-dropdown', 'value'),
    Input('yaxis-viscolumn-dropdown', 'value'),
    Input('visualization-type-dropdown', 'value'),
    State('datatable-upload-container', 'data')
)
def update_graph(x_column, y_column, vis_type, data):
    if not data or not x_column or not y_column or not vis_type:
        return {}

    df = pd.DataFrame(data)

    if vis_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column)
    elif vis_type == 'line':
        fig = px.line(df, x=x_column, y=y_column)
    elif vis_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column)
    elif vis_type == 'pie':
        fig = px.pie(df, names=x_column, values=y_column)
    elif vis_type == 'histogram':
        fig = px.histogram(df, x=x_column)
    elif vis_type == 'box':
        fig = px.box(df, x=x_column, y=y_column)
    elif vis_type == '3dscatter':
        fig = px.scatter_3d(df, x=x_column, y=y_column)
    elif vis_type == 'area':
        fig = px.area(df, x=x_column, y=y_column)
    elif vis_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column)
    else:
        return {}
    log_user_action(f"Updated visualization: {vis_type}")
    log_user_action(f"Command: fig = px.{vis_type}(df, x={x_column}, y={y_column})")

    return fig


@app.callback(
    Output('saved-visgraphs-container', 'children'),
    [Input('save-graph-button', 'n_clicks')],
    [State('visualisation-graph', 'figure'),
     State('saved-visgraphs-container', 'children')]
)
def save_current_graph(n_clicks, current_figure, current_saved_graphs):
    if not current_figure:
        raise dash.exceptions.PreventUpdate
    current_graph = dcc.Graph(figure=current_figure)
    log_user_action(f"Save Graph: current_graph = dcc.Graph(figure=current_figure)")
    if not current_saved_graphs:
        current_saved_graphs = []
    current_saved_graphs.append(current_graph)
    log_user_action(f"Save Graph: current_saved_graphs.append(current_graph)")
    return current_saved_graphs