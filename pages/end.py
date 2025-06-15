import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback

dash.register_page(__name__, path="/end")

layout = html.Div(
    [
        html.H1("Grazie per l'attenzione"),
        html.H4("Studenti: ciccio e franco"),
        html.H4("Modificare questa pagina"),
        html.Div(
            [
                html.Button("Ricomincia", id="start-button", n_clicks=0),
                dcc.Location(id="from-end-to-page-1-url", refresh=True),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
    ]
)


@callback(
    Output("from-end-to-page-1-url", "pathname"),
    [Input("start-button", "n_clicks")],
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-1"
