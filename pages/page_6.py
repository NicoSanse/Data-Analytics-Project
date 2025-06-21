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
        html.H2("Limitazioni"),
        html.Ul(
            [
                html.Li(
                    "I grafi di interazione considerano solo i dialoghi diretti tra personaggi, escludendo relazioni indirette o menzioni."
                ),
                html.Li(
                    "L’analisi delle emozioni si basa su lessici generici non adattati al vocabolario specifico di Harry Potter (nomi, incantesimi, espressioni particolari)."
                ),
                html.Li(
                    "Assenza di lexicon emotivi specifici per il mondo della saga: parole cariche di significato emotivo possono non essere riconosciute."
                ),
                html.Li(
                    "Impossibilità di analizzare elementi paralinguistici (intonazione, tono di voce, espressioni) che contribuiscono all’emozione trasmessa nei film."
                ),
                html.Li(
                    "Il modello non coglie il contesto visivo, gestuale e registico, fondamentale per alcune scene chiave."
                ),
            ],
            style={"fontSize": "1.1em", "marginTop": "20px", "textAlign": "justify"},
        ),
        html.P(
            "Questi limiti suggeriscono sviluppi futuri come la creazione di lessici emotivi personalizzati per Harry Potter, "
            "l’integrazione di segnali non testuali e l’analisi di relazioni indirette o complesse tra personaggi.",
            style={
                "marginTop": "30px",
                "fontSize": "1.1em",
                "lineHeight": "1.6em",
                "textAlign": "justify",
            },
        ),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page", n_clicks=0),
                dcc.Location(id="from-page-6-to-page-5-url"),
                html.Button("Prossima pagina →", id="next-page", n_clicks=0),
                dcc.Location(id="from-page-6-to-page-end-url"),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
    ],
    style={"padding": "20px"},
)


#################################################################################################


@callback(
    Output("from-page-6-to-page-5-url", "pathname"),
    Input("next-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    # se la presentazione è finita, mettere /end
    return "/end"


@callback(
    Output("from-page-6-to-page-end-url", "pathname"),
    Input("prev-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-5"
