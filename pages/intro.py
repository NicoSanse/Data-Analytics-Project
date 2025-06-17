import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback

dash.register_page(__name__, path="/")


layout = html.Div(
    style={
        "padding": "40px",
        "fontFamily": "Arial, sans-serif",
        "minHeight": "100vh",
    },
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
            'Data l\'eccezionale popolarità del prodotto cinematografico "Harry Potter" nel corso del tempo '
            "sono non pochi i contenuti che sono nati a partire da tale marchio, si vedano videogiochi, gadget, "
            "blog ecc ... \n"
            "Tuttavia a livello cinematografico non si è mai andati oltre ai noti 8 film della saga, mentre un esempio opposto "
            'di come sfruttare la popolarità dei prodotti è offerto dal "Marvel Cinematic Universe", il quale ha invece '
            "creato con successo un'infinita serie di franchise. \n \n"
            'Questo progetto si pone come obiettivo quello di studiare la saga di "Harry Potter" con strumenti analitici '
            "per ottenere informazioni rivendibili utili alla creazione di prodotti commerciali appartenenti all'universo fantastico "
            "della nota scrittrice J. K. Rowling. In sintesi gli obiettivi sono: ",
            style={"marginTop": "30px", "fontSize": "1.1em", "lineHeight": "1.6em"},
        ),
        html.Ul(
            [
                html.Li(
                    "Identificazione di personaggi non centrali, ma ad alta carica emotiva per possibili spin-off"
                ),
                html.Li(
                    "Studio dell'evoluzione dell'emozione dei personaggi più importanti per riproducibilità"
                ),
                html.Li(
                    "Ricerca di correlazione tra emozione rilevata e casata a fine di marketing"
                ),
                html.Li("Raccomandatore scene per appassionati"),
            ],
            style={"marginTop": "20px", "fontSize": "1.05em"},
        ),
        html.Div(
            [
                html.Button("Inizo", id="start-button", n_clicks=0),
                dcc.Location(id="page-1-url", refresh=True),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
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
)


@callback(
    Output("page-1-url", "pathname"),
    [Input("start-button", "n_clicks")],
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-1"
