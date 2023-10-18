"""
File:               app.py
Description:        This is the main executable file for the Dash web application. It integrates 
                    the layout, callbacks, and the app instance to provide the desired functionality 
                    and visualization for the user.
                    This file initializes the app's layout, integrates callback functionalities from 
                    various modules, and starts the Dash server.
Authors:            Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Date Updated:       2023-10-18
"""
# Import necessary libraries
import dash

# Import app instance
from app_instance import app

# Import layout configuration for the app
from layout import layout

# Import callback functionalities for various operations
from callbacks import (
    state_saving_callbacks, 
    data_management_callbacks, 
    statistical_summary_callbacks, 
    visualisation_callbacks, 
    analytics_callbacks, 
    parameters_callbacks
)

# Assign the defined layout to the app instance
app.layout = layout

# Run the app
if __name__ == '__main__':
    # Start the Dash app server with debugging enabled
    app.run_server(debug=True)
