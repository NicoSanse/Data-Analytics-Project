import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback

dash.register_page(__name__)

layout = html.Div(
    [
        html.H1("Page 2"),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page-2", n_clicks=0),
                dcc.Location(id="from-page-2-to-page-1-url"),
                html.Button("Prossima pagina →", id="next-page-2", n_clicks=0),
                dcc.Location(id="from-page-2-to-page-3-url"),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
        # dcc.Location(id="page-2-url"),
    ]
)


@callback(
    Output("from-page-2-to-page-3-url", "pathname"),
    Input("next-page-2", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-3"


@callback(
    Output("from-page-2-to-page-1-url", "pathname"),
    Input("prev-page-2", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-1"
