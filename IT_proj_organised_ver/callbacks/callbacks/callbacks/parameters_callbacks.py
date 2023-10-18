from dash import Input, Output, html, dash_table, dcc
import pandas as pd
import base64
import io
from dash.dependencies import State, ALL
import plotly.express as px
import dash_bootstrap_components as dbc
import json
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import dash
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.subplots as sp
import plotly.figure_factory as ff
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

from app_instance import app
from state_saving_func import *

@app.callback(
    Output('input-dict-store', 'data'),
    [Input('data-analysis-dropdown', 'value')]
)
def update_input_dict_store(selected_analysis_method):
    add_callback_source_code(update_input_dict_store)
    if selected_analysis_method == 'clustering':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select columns:",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "number_selection",
                "DataType": "numeric",
                "Text": "Number of Clusters:",
                "AcceptableValues": {"min": 1, "max": None}
            }
        ]
    elif selected_analysis_method == 'classification':
        return [
            {
                "Input": "column_selection",
                "DataType": "numeric",
                "Text": "Select column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            }, 
            {
                "Input": "column_selection",
                "DataType": "string",
                "Text": "Select target column(s):",
                "AcceptableValues": {"min": 1, "max": None}
            },
            {
                "Input": "dropdown",
                "DataType": "string",
                "Text": "Select Kernel Type:",
                "AcceptableValues": ["linear", "poly", "rbf", "sigmoid"]
            }
        ]
    else:
        # Handle other cases or invalid selections appropriately
        return []

