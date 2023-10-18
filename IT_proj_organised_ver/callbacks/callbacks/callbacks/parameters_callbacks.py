"""
File:           parameters_callbacks.py
Description:    This module contains callbacks related to updating 
                input parameters for different analysis methods.

Authors:        Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Last Updated:   2023-10-18
"""

# Library imports
import dash
from dash import Input, Output
from app_instance import app
from state_saving_func import *

# Callback to update the input parameters based on the selected analysis method
@app.callback(
    Output('input-dict-store', 'data'),
    [Input('data-analysis-dropdown', 'value')]
)
def update_input_dict_store(selected_analysis_method):
    add_callback_source_code(update_input_dict_store)  # Log the current callback function for later use
    
    # Define input parameters for clustering analysis
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
    
    # Define input parameters for classification analysis
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
    
    # Handle other cases or invalid selections
    else:
        return []
