import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import os

dash.register_page(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path_full_saga_df = os.path.join(
    BASE_DIR, "outputs", "full_saga_with_emotions.csv"
)
file_path_nrc_emotion = os.path.join(BASE_DIR, "outputs", "nrc_emotions.csv")


full_saga_df = pd.read_csv(
    file_path_full_saga_df,
    encoding="utf-8-sig",
)
nrc_emotions = pd.read_csv(
    file_path_nrc_emotion,
    encoding="utf-8-sig",
)


def plot_bar(measure):
    house_emotion_sums = full_saga_df.groupby("House").sum()
    house_emotion_sums = house_emotion_sums[
        ["sadness", "joy", "love", "anger", "fear", "surprise"]
    ]
    house_emotion_sums.drop(
        ["Beauxbatons Academy of Magic", "Durmstrang Institute"], inplace=True
    )

    roberta_emotion_colors = {
        "joy": "#FFD700",
        "sadness": "#1E90FF",
        "anger": "#FF4500",
        "fear": "#8B008B",
        "surprise": "#00CED1",
        "love": "#FF69B4",
        "optimism": "#32CD32",
    }

    lexicon_emotion_colors = {
        "anticipation": "#FFD700",  # oro – attesa, eccitazione
        "disgust": "#556B2F",  # verde oliva scuro – repulsione
        "fear": "#8B0000",  # rosso scuro – paura
        "joy": "#FFA500",  # arancione brillante – felicità
        "sadness": "#1E90FF",  # blu acceso – tristezza
        "surprise": "#BA55D3",  # viola medio – sorpresa
        "trust": "#228B22",
    }

    nrc_emotions_sum = nrc_emotions.groupby("House").sum()
    nrc_emotions_sum = nrc_emotions_sum[
        ["anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    ]
    nrc_emotions_sum.drop(
        ["Beauxbatons Academy of Magic", "Durmstrang Institute"], inplace=True
    )

    if measure == "absolute":
        fig1 = go.Figure()
        fig2 = go.Figure()

        for emotion in house_emotion_sums.columns:
            fig1.add_trace(
                go.Bar(
                    name=emotion,
                    x=house_emotion_sums.index,
                    y=house_emotion_sums[emotion],
                    marker_color=roberta_emotion_colors.get(
                        emotion, "#333333"
                    ),  # colore di fallback
                )
            )

        fig1.update_layout(
            barmode="stack",
            title="Somma delle emozioni (stacked) per casata",
            xaxis_title="Casata",
            yaxis_title="Somma",
            xaxis_tickangle=-45,
            legend_title="Emozione",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        for emotion in nrc_emotions_sum.columns:
            fig2.add_trace(
                go.Bar(
                    name=emotion,
                    x=nrc_emotions_sum.index,
                    y=nrc_emotions_sum[emotion],
                    marker_color=lexicon_emotion_colors.get(emotion, "#999999"),
                )
            )

        fig2.update_layout(
            barmode="stack",
            title="Somma delle emozioni (stacked) per casata",
            xaxis_title="Casata",
            yaxis_title="Somma",
            xaxis_tickangle=45,
            legend_title="Emozione",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="rgba(0,0,0,0.2)", gridwidth=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )

    elif measure == "normalized":
        fig1 = go.Figure()
        fig2 = go.Figure()
        normalized_house_emotions = house_emotion_sums.apply(
            lambda row: row / row.sum(), axis=1
        )
        normalized_nrc_emotions = nrc_emotions_sum.apply(
            lambda row: row / row.sum(), axis=1
        )

        for emotion in normalized_house_emotions.columns:
            fig1.add_trace(
                go.Bar(
                    name=emotion,
                    x=normalized_house_emotions.index,
                    y=normalized_house_emotions[emotion],
                    marker_color=roberta_emotion_colors.get(emotion, "#666666"),
                )
            )

        fig1.update_layout(
            barmode="stack",
            title="Distribuzione relativa delle emozioni per casata",
            xaxis_title="Casata",
            yaxis_title="Percentuale",
            xaxis_tickangle=-45,
            legend_title="Emozione",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        for emotion in normalized_nrc_emotions.columns:
            fig2.add_trace(
                go.Bar(
                    name=emotion,
                    x=normalized_nrc_emotions.index,
                    y=normalized_nrc_emotions[emotion],
                    marker_color=lexicon_emotion_colors.get(emotion, "#999999"),
                )
            )

        fig2.update_layout(
            barmode="stack",
            title="Distribuzione relativa delle emozioni per casata",
            xaxis_title="Casata",
            yaxis_title="Percentuale",
            xaxis_tickangle=45,
            legend_title="Emozione",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="rgba(0,0,0,0.2)", gridwidth=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )

    else:
        raise KeyError("Valore non previsto")

    return fig1, fig2


#################################################################################################

layout = html.Div(
    [
        html.H1("Parte 3"),
        html.H2("Emozioni dominanti delle Case di appartenenza"),
        html.P(
            "L’analisi delle emozioni espresse dai personaggi nei dialoghi permette di definire per ciascuna Casa di Hogwarts un profilo emotivo distintivo, offrendo nuove prospettive rispetto alle rappresentazioni tradizionali.\n"
            "Questi risultati, oltre ad arricchire l’interpretazione letteraria, potrebbero essere utilizzati anche in ambito marketing per creare prodotti e campagne personalizzate che trasmettano l’essenza emotiva di ogni Casa, rafforzando il legame dei fan con il mondo di Harry Potter.\n"
            "Per stimare lo stile emozionale di ciascuna Casa, le emozioni sono state quantificate e aggregate in base alla Casa di appartenenza dei personaggi, utilizzando strumenti specifici di sentiment analysis.\n" ,
            style={"marginTop": "30px", "fontSize": "1.1em", "lineHeight": "1.6em" , "textAlign": "justify", "whiteSpace": "pre-line",},
        ),
        html.P(
            "In questa parte intendiamo fare un'analisi sullo stile emotivo dei personaggi usando la casata di "
            "appartenenza come criterio di suddivisione. Sono stati usati due approcci: uno basato su transformers "
            " e l'altro basato su lexicon.",
            style={"marginTop": "30px", "fontSize": "1.1em", "lineHeight": "1.6em"},
        ),
        html.Div(
            [
                html.Label("Tipo di visualizzazione:"),
                dcc.RadioItems(
                    id="view-content",
                    options=[
                        {"label": "assoluta", "value": "absolute"},
                        {"label": "normalizzata", "value": "normalized"},
                    ],
                    value="absolute",
                    labelStyle={"display": "inline-block", "margin-right": "20px"},
                ),
            ],
            style={"margin": "20px"},
        ),
        html.Div(
            [
                html.H4("Risultati ottenuti da Roberta e dal Lexicon"),
                html.Div(
                    [
                        dcc.Graph(id="roberta-graphic"),
                        html.Div(
                            style={
                                "width": "2px",
                                "backgroundColor": "#ccc",
                                "display": "inline-block",
                                "height": "400px",
                                "margin": "0 10px",
                            }
                        ),
                        dcc.Graph(id="lexicon-graphic"),
                    ],
                    style={"display": "flex"},
                ),
            ]
        ),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page", n_clicks=0),
                dcc.Location(id="from-page-5-to-page-4-url"),
                html.Button("Prossima pagina →", id="next-page", n_clicks=0),
                dcc.Location(id="from-page-5-to-page-6-url"),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
        # dcc.Location(id="page-3-url"),
    ],
    style={"padding": "10px"},
)


#################################################################################################


@callback(
    [Output("roberta-graphic", "figure"), Output("lexicon-graphic", "figure")],
    [Input("view-content", "value")],
)
def update_plot(measure):
    fig1, fig2 = plot_bar(measure)
    return fig1, fig2


@callback(
    Output("from-page-5-to-page-6-url", "pathname"),
    Input("next-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    # se la presentazione è finita, mettere /end
    return "/page-6"


@callback(
    Output("from-page-5-to-page-4-url", "pathname"),
    Input("prev-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-4"
