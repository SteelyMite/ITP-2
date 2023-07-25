from dash import dcc, html

layout = html.Div([
    html.H2("Welcome to PyExploratory!"),
    dcc.Markdown('''
        Welcome to **PyExploratory**. 
        To start exploring, upload your data by clicking the "Upload Data" link.
    ''')
])

# Add callbacks for this page here if needed
