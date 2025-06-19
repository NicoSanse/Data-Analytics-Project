import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback

dash.register_page(__name__, path="/")


layout = html.Div(
    children=[
        html.Div(
            style={
                "backgroundImage": "url('/assets/wordcloud.png')",
                "backgroundSize": "cover",
                "backgroundPosition": "center",
                "height": "250px",
                "width": "100%",
                "marginBottom": "30px",
            }
        ),
        html.Div(
            children=[
                html.H1(
                    "Data Analytics Project: Harry Potter",
                    style={"textAlign": "center", "fontSize": "3em"},
                ),
                html.H2(
                    "Analisi dei dialoghi nei film della saga",
                    style={
                        "textAlign": "center",
                        "fontSize": "1.8em",
                        "marginTop": "10px",
                    },
                ),
                html.P(
                    'Nel settore dell’intrattenimento, la capacità di analizzare scientificamente le dinamiche emotive di personaggi e storie è la chiave per il successo di ogni franchise. \n' 
                    'Questa dashboard nasce per esplorare la saga di Harry Potter con un approccio data-driven: analizziamo emozioni, dialoghi e relazioni nei film per fornire strumenti concreti a chi crea, promuove e valorizza prodotti narrativi, '
                    'con l’obiettivo di scoprire nuove storie da raccontare, personalizzare il marketing e ispirare la prossima generazione di successi fantasy. \n' 
                    'Lo studio si articola in 4 fasi:' ,
                    style={
                        "marginTop": "30px",
                        "fontSize": "1.1em",
                        "lineHeight": "1.6em",
                    },
                ),
                html.Ul(
                    [
                        html.Li(
                            "Identificazione di personaggi non centrali, ma ad alta carica emotiva per lo sviluppo di possibili spin-off."
                        ),
                        html.Li(
                            "Studio dell'evoluzione dell'emozione dei personaggi più importanti per trovare archetipi emotivi trasferibili ad altri franchise."
                        ),
                        html.Li(
                            "Ricerca di correlazione tra emozione rilevata e Casa a fine di marketing."
                        ),
                        html.Li("Sviluppo di un raccomandatore di scene per emozione per suggerire automaticamente le scene di maggior impatto emotivo, utili sia in contesti di fruizione che di promozione."),
                    ],
                    style={"marginTop": "20px", "fontSize": "1.05em"},
                ),
                html.Div(
                    [
                        html.Button("Inizio", id="start-button", n_clicks=0),
                        dcc.Location(id="from-start-to-page-1-url", refresh=True),
                    ],
                    style={
                        "marginTop": "30px",
                        "display": "flex",
                        "gap": "10px",
                        "justifyContent": "center",
                    },
                ),
                html.P(
                    "Autori: Alessia Novacco, Nicolò Sansevrino",
                    style={
                        "marginTop": "40px",
                        "textAlign": "center",
                        "fontSize": "1.1em",
                        "color": "#CCCCCC",
                    },
                ),
            ],
            style={
                "padding": "40px",
                "fontFamily": "Arial, sans-serif",
                "minHeight": "100vh",
            },
        ),
    ],
)


@callback(
    Output("from-start-to-page-1-url", "pathname"),
    [Input("start-button", "n_clicks")],
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-1"
