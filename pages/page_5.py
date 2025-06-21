import dash
from dash import html, dcc, callback, Input, Output, State
import dash
from dash import Input, Output, html, dcc, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from packcircles import pack

dash.register_page(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_final_df = os.path.join(BASE_DIR, "outputs", "final_df.csv")
path_final_df_nrc = os.path.join(BASE_DIR, "outputs", "final_df_nrc_emotions.csv")

final_df = pd.read_csv(path_final_df, encoding="utf-8-sig")
final_df_nrc = pd.read_csv(path_final_df_nrc, encoding="utf-8-sig")


def generate_random_positions(n_bubbles, grid_size=3, noise_level=0.3, spacing=1.5):
    # Genera le coordinate della griglia (in ordine riga-colonna)
    x_coords = np.tile(np.arange(grid_size), grid_size) * spacing
    y_coords = np.repeat(np.arange(grid_size)[::-1], grid_size) * spacing

    # Prendi solo le prime n posizioni, utile se n < grid_size^2
    x_coords = x_coords[:n_bubbles]
    y_coords = y_coords[:n_bubbles]

    # Aggiungi rumore per rendere le posizioni meno rigide
    x_noisy = x_coords + np.random.normal(0, noise_level, size=n_bubbles)
    y_noisy = y_coords + np.random.normal(0, noise_level, size=n_bubbles)

    return x_noisy.tolist(), y_noisy.tolist()


def best_scene_with_all_emotions_plot(method):

    if method == "bert":
        df = final_df
    else:
        df = final_df_nrc

    df = df.dropna(subset=["Intensità_media"])
    df["Intensità_media"] = df["Intensità_media"].astype(float)
    max_per_emotion = df.loc[df.groupby("Emozione")["Intensità_media"].idxmax()]

    max_intensity = max_per_emotion["Intensità_media"].max()
    min_intensity = max_per_emotion["Intensità_media"].min()
    norm_intensity = (max_per_emotion["Intensità_media"] - min_intensity) / (
        max_intensity - min_intensity
    )
    scaled_intensity = norm_intensity**2  # potenza 2 per aumentare la dinamica

    min_size = 60
    max_size = 200
    max_per_emotion["BubbleSize"] = 10 * (
        min_size + scaled_intensity * (max_size - min_size)
    )

    n_bubbles = len(max_per_emotion)
    x_manual_noisy, y_manual_noisy = generate_random_positions(
        n_bubbles, grid_size=3, noise_level=0.4, spacing=3.5
    )

    max_per_emotion["x_manual"] = x_manual_noisy
    max_per_emotion["y_manual"] = y_manual_noisy

    max_per_emotion["label"] = (
        "Film n. "
        + max_per_emotion["Film"].astype(str)
        + " - Capitolo "
        + max_per_emotion["Capitolo"].astype(str)
    )

    fig = px.scatter(
        max_per_emotion,
        x="x_manual",
        y="y_manual",
        size="BubbleSize",
        color="Emozione",
        hover_name="Riassunto_narrativo",
        text="label",
        title="Scena più intensa per ciascuna emozione",
    )

    fig.update_traces(
        textposition="top center", marker=dict(sizemode="area", sizeref=0.15)
    )
    fig.update_layout(
        xaxis=dict(range=[-4, 12], visible=False),
        yaxis=dict(range=[-4, 12], visible=False),
        height=600,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def best_five_scenes_for_one_emotion_plot(emotion, method):
    pass


#################################################################################################

layout = html.Div(
    [
        html.H1("Parte 4"),
        html.H2("Raccomandatore di scene per emozione"),
        html.P(
            "Individuare automaticamente le scene a più alto impatto emotivo permette di suggerire agli utenti i momenti più intensi della saga, adattando l’esperienza di visione alle preferenze emozionali e facilitando la selezione di contenuti per scopi promozionali e di comunicazione.",
            style={
                "marginTop": "30px",
                "fontSize": "1.1em",
                "lineHeight": "1.6em",
                "textAlign": "justify",
                "whiteSpace": "pre-line",
            },
        ),
        html.Label(
            "Metodo utilizzato:",
            style={"marginTop": "30px"},
        ),
        dcc.RadioItems(
            id="method-radio",
            options=[
                {"label": "BERT", "value": "bert"},
                {"label": "NRC", "value": "nrc"},
            ],
            value="nrc",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Scegli un'opzione di visualizzazione:",
                            style={"marginTop": "30px"},
                        ),
                        dcc.Dropdown(
                            id="visualization-dropdown",
                            options=[
                                {
                                    "label": "Voglio vedere la scena più intensa per ogni emozione disponibile",
                                    "value": "best-scene-for-all-emotions",
                                },
                                {
                                    "label": "Voglio vedere le migliori scene per un'emozione",
                                    "value": "best-n-scenes-for-one-emotion",
                                },
                            ],
                            clearable=False,
                            value="best-scene-for-all-emotions",
                            style={"width": "80%"},
                        ),
                    ],
                    style={"flex": "1", "marginRight": "20px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Scegli un'emozione:",
                            style={"marginTop": "30px"},
                        ),
                        dcc.Dropdown(
                            id="emotion-dropdown",
                            options=[
                                {"label": "Joy", "value": "joy"},
                                {"label": "Anger", "value": "anger"},
                                {"label": "Fear", "value": "fear"},
                                {"label": "Sadness", "value": "sadness"},
                                {"label": "Love", "value": "love"},
                                {"label": "Surprise", "value": "surprise"},
                            ],
                            disabled=True,
                            value="joy",
                            style={"width": "80%"},
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "alignItems": "flex-start",
            },
        ),
        html.Div(
            dcc.Graph(id="plot"),
            style={
                "margin": "0 auto",
                "padding": "20px",
            },
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
    ],
    style={"padding": "10px"},
)


#################################################################################################


@callback(
    Output("emotion-dropdown", "disabled"),
    Output("emotion-dropdown", "value"),
    Input("visualization-dropdown", "value"),
)
def toggle_dropdown_visibility(selected_value):
    if selected_value == "best-scene-for-all-emotions":
        return True, None
    else:
        return False, "joy"


@callback(
    Output("plot", "figure"),
    [
        Input("visualization-dropdown", "value"),
        Input("emotion-dropdown", "value"),
        Input("method-radio", "value"),
    ],
)
def plots(visualization_value, emotion_value, method):
    if visualization_value == "best-scene-for-all-emotions":
        return best_scene_with_all_emotions_plot(method)
    else:
        return best_five_scenes_for_one_emotion_plot(emotion_value, method)


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
