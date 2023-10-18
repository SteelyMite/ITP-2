"""
File:               app_instance.py
Description:        This module initializes the main Dash application instance and sets it up with 
                    the desired configurations and styles. Additionally, this module defines the server instance for the app, allowing it 
                    to be deployed on platforms like Heroku or AWS.
Authors:            Chitipat Marsri, Diego Disley, Don Le, Kyle Welsh
Date Updated:       2023-10-18
"""

# Libraries and components required for the Dash web application framework
import dash

# Dash Bootstrap components to style the app with Bootstrap themes
import dash_bootstrap_components as dbc

# Create and configure the main Dash application instance
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.YETI])

# Assign the Dash app's server for deployment on web servers
server = app.server
