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
        html.H2("Raccomandatore di scene per emozione"),
        html.P(
            "Individuare automaticamente le scene a più alto impatto emotivo permette di suggerire agli utenti i momenti più intensi della saga, adattando l’esperienza di visione alle preferenze emozionali e facilitando la selezione di contenuti per scopi promozionali e di comunicazione.",
            style={"marginTop": "30px", "fontSize": "1.1em", "lineHeight": "1.6em", "textAlign": "justify", "whiteSpace": "pre-line",},
        ),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page", n_clicks=0),
                dcc.Location(id="from-page-6-to-page-5-url"),
                html.Button("Prossima pagina →", id="next-page", n_clicks=0),
                dcc.Location(id="from-page-6-to-page-7-url"),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
    ],
    style={"padding": "10px"},
)


#################################################################################################


@callback(
    Output("from-page-6-to-page-7-url", "pathname"),
    Input("next-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    # se la presentazione è finita, mettere /end
    return "/page-7"


@callback(
    Output("from-page-6-to-page-5-url", "pathname"),
    Input("prev-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-5"
