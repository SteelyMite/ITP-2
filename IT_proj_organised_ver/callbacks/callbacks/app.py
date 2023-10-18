import dash
from app_instance import app
from layout import layout

from callbacks import state_saving_callbacks, data_management_callbacks, statistical_summary_callbacks, visualisation_callbacks, analytics_callbacks, parameters_callbacks

# Set the app layout
app.layout = layout

if __name__ == '__main__':
    app.run_server(debug=True)
