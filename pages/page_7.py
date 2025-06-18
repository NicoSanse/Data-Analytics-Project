import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt


dash.register_page(__name__)


#################################################################################################

layout = html.Div(
    [
        html.H1("Parte 4"),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page", n_clicks=0),
                dcc.Location(id="from-page-7-to-page-6-url"),
                html.Button("Prossima pagina →", id="next-page", n_clicks=0),
                dcc.Location(id="from-page-7-to-page-end-url"),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
    ],
    style={"padding": "10px"},
)


#################################################################################################


@callback(
    Output("from-page-7-to-page-end-url", "pathname"),
    Input("next-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    # se la presentazione è finita, mettere /end
    return "/end"


@callback(
    Output("from-page-7-to-page-6-url", "pathname"),
    Input("prev-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-6"
