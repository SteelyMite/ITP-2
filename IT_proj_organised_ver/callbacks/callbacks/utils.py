import base64
import io
import pandas as pd
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc

def parse_contents(contents, file_type):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if file_type not in ['csv', 'excel', 'json']:
        raise ValueError("Unsupported file type.")

    try:
        if file_type == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif file_type == 'excel':
            df = pd.read_excel(io.BytesIO(decoded))
        elif file_type == 'json':
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        else:
            raise ValueError("Unknown file type.")
    except Exception as e:
        raise ValueError("Error reading file. Please check if the file type matches its content.") from e
    return df



def generate_column_summary_box(df, column_name):
    # Calculate statistics
    nan_count = df[column_name].isna().sum()

    if pd.api.types.is_numeric_dtype(df[column_name]):
        # Generate histogram for numeric columns
        data_type = "Numeric"
        fig = px.histogram(df, x=column_name, nbins=5)  # Here, 5 bins are used for simplicity; adjust as needed.
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        mean_val = df[column_name].mean()
        data_cleaning_options = [
            {"label": "Replace NaN with Min", "value": "min"},
            {"label": "Replace NaN with Max", "value": "max"},
            {"label": "Replace NaN with Mean", "value": "mean"},
            {"label": "Replace NaN with Zero", "value": "zero"}
        ]

        normalization_options = [
            {"label": "Normalize (New Column)", "value": "normalize_new"},
            {"label": "Normalize (Replace)", "value": "normalize_replace"},
        ]

        dropdown_menu = dbc.DropdownMenu(
            label="Options",
            children=[
                dbc.DropdownMenuItem("Change data type", header=True),
                dbc.DropdownMenuItem("Numeric", id={"type": "convert", "index": column_name, "to": "Numeric"}),
                dbc.DropdownMenuItem("String", id={"type": "convert", "index": column_name, "to": "String"}),
                
                dbc.DropdownMenuItem(divider=True),

                dbc.DropdownMenuItem("Data Cleaning", header=True),
                *[dbc.DropdownMenuItem(item["label"], id={"type": "clean", "index": column_name, "action": item["value"]}) for item in data_cleaning_options],

                dbc.DropdownMenuItem(divider=True),

                dbc.DropdownMenuItem("Normalization", header=True),
                *[dbc.DropdownMenuItem(item["label"], id={"type": "clean", "index": column_name, "action": item["value"]}) for item in normalization_options],
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
                *[dbc.DropdownMenuItem(item["label"], id={"type": "clean", "index": column_name, "action": item["value"]}) for item in data_cleaning_options],
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
