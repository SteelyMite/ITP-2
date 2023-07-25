import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

from app import app
from pages import home, upload, visualization

navbar = dbc.NavbarSimple(
    brand="PyExploratory",
    brand_href="/", 
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Upload", href="/upload")),
        dbc.NavItem(dbc.NavLink("Visualization", href="/visualization")),
    ],
    sticky="top",
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Container(id="page-content", className="pt-4", fluid=True),
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/upload':
        return upload.layout
    elif pathname == '/visualization':
        return visualization.layout
    else:
        return dbc.Jumbotron([
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised...")
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
