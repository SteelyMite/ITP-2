import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

from app import app
from pages import home, upload, visualization

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Nav([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Upload Data", href="/upload")),
                dbc.NavItem(dbc.NavLink("Visualization", href="/visualization")),
            ],
            brand="PyExploratory",
            brand_href="/",
            sticky="top",
        )
    ], className='navbar navbar-expand-lg navbar-light bg-light'),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/upload':
        return upload.layout
    elif pathname == '/visualization':
        return visualization.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)
