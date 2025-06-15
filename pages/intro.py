import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback

dash.register_page(__name__, path="/")

layout = html.Div(
    [
        html.H1("Progetto Data Analytics"),
        html.H4("Studenti: ciccio e franco"),
        html.H4("Modificare questa pagina"),
        html.Div(
            [
                html.Button("Inizo", id="start-button", n_clicks=0),
                dcc.Location(id="page-1-url", refresh=True),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
    ]
)


@callback(
    Output("page-1-url", "pathname"),
    [Input("start-button", "n_clicks")],
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-1"
