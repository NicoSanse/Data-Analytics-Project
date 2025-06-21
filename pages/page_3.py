import dash
from dash import html, dcc, callback, Input, Output, MATCH
import dash
from dash import Input, Output, html, dcc, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

dash.register_page(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path_tabella = os.path.join(BASE_DIR, "outputs", "df_emotions.csv")
file_path_dialogs = os.path.join(BASE_DIR, "outputs", "dialogs_bert_sentiment.csv")


df_emotions = pd.read_csv(
    file_path_tabella,
    encoding="utf-8-sig",
)
dialogs = pd.read_csv(
    file_path_dialogs,
    encoding="utf-8-sig",
)


def create_line_plot(k, film_id, emotions):

    centrality_list = []
    for movie_id, group in dialogs.groupby("Movie ID"):
        G = nx.DiGraph()
        speakers = group["speaker"].tolist()
        edges = [(a, b) for a, b in zip(speakers[:-1], speakers[1:]) if a != b]
        G.add_edges_from(edges)
        out_deg = G.out_degree()
        in_deg = G.in_degree()
        pagerank = nx.pagerank(G, weight=None)
        temp = pd.DataFrame(
            {
                "speaker": list(G.nodes()),
                "out_degree": [out_deg[n] for n in G.nodes()],
                "in_degree": [in_deg[n] for n in G.nodes()],
                "pagerank": [pagerank[n] for n in G.nodes()],
                "Movie ID": movie_id,
            }
        )
        centrality_list.append(temp)
    centrality_df = pd.concat(centrality_list, ignore_index=True)

    centrality_df["speaker"] = centrality_df["speaker"].str.lower().str.strip()
    df_emotions["Character Name"] = (
        df_emotions["Character Name"].str.lower().str.strip()
    )

    # --- Conta presenza in quanti film
    film_count = centrality_df.groupby("speaker")["Movie ID"].nunique()
    min_films = 2
    relevant_speakers = film_count[film_count >= min_films].index

    # --- Pagerank medio su personaggi rilevanti
    pagerank_by_char = (
        centrality_df[centrality_df["speaker"].isin(relevant_speakers)]
        .groupby("speaker")["pagerank"]
        .mean()
        .sort_values(ascending=False)
    )

    central_characters = pagerank_by_char.index.tolist()

    # Protezione: se top_k va fuori indice
    if k >= len(central_characters):
        return go.Figure().update_layout(
            title="Nessun personaggio disponibile per questo indice."
        )

    selected_character = central_characters[k]
    chapters = df_emotions[df_emotions["Movie ID"] == film_id]["Chapter ID"].unique()

    # emotion_order = ["sadness", "love", "joy", "anger", "fear", "surprise"] KeyError: 'love' not in index
    emotion_order = emotions

    subset = df_emotions[
        (df_emotions["Movie ID"] == film_id)
        & (df_emotions["Character Name"] == selected_character)
    ]
    if subset.empty:
        return go.Figure().update_layout(
            title=f"Nessuna emozione trovata per {selected_character}."
        )

    # Conta emozioni per capitolo
    pivot = subset.groupby(["Chapter ID", "Emotion"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(chapters, fill_value=0)

    # Dopo l'unstack
    for emo in emotion_order:
        if emo not in pivot.columns:
            pivot[emo] = 0

    # Re-ordina le colonne
    pivot = pivot[emotion_order]

    pivot_long = pivot.reset_index().melt(
        id_vars="Chapter ID",
        value_vars=emotion_order,
        var_name="Emotion",
        value_name="Count",
    )

    fig = px.line(
        pivot_long,
        x="Chapter ID",
        y="Count",
        color="Emotion",
        labels={"Chapter": "Capitolo", "Count": "Conteggio"},
    )

    fig.update_layout(
        legend_title_text="Emozione",
        title=dict(
            text=f"Andamento delle emozioni per <b>{selected_character.title()}</b> nel film {film_id}"
        ),
    )

    return fig


#################################################################################################

layout = html.Div(
    [
        html.H1("Parte 2"),
        html.H2("Analisi dei personaggi centrali"),
        html.P(
            "Analizzare i pattern emotivi dei protagonisti nelle storie di successo è fondamentale per ideare nuove trame coinvolgenti. In questo studio, ci siamo focalizzati sui personaggi principali della saga di Harry Potter,"
            "esaminando l’evoluzione delle loro emozioni nei film per individuare archetipi emotivi trasferibili ad altri franchise. \n"
            "La selezione dei protagonisti è stata basata sulla loro centralità narrativa, misurata tramite il valore medio di Pagerank nei grafi delle interazioni. Vengono così identificati i personaggi più influenti e connessi all’interno della storia.\n",
            style={
                "marginTop": "30px",
                "fontSize": "1.1em",
                "lineHeight": "1.6em",
                "textAlign": "justify",
                "whiteSpace": "pre-line",
            },
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label(
                            "Emozioni selezionabili:",
                            style={"fontWeight": "bold", "marginBottom": "5px"},
                        ),
                        dcc.Dropdown(
                            id="emotion-multiselect",
                            options=[
                                {"label": "joy", "value": "joy"},
                                {"label": "anger", "value": "anger"},
                                {"label": "fear", "value": "fear"},
                                {"label": "sadness", "value": "sadness"},
                                {"label": "love", "value": "love"},
                                {"label": "surprise", "value": "surprise"},
                            ],
                            value=["joy", "anger", "fear"],
                            multi=True,
                            clearable=False,
                            placeholder="Seleziona le emozioni da visualizzare",
                            style={"width": "50%"},
                        ),
                        html.Label(
                            "Top risultati:",
                            style={"fontWeight": "bold", "marginBottom": "5px"},
                        ),
                        dcc.Input(
                            id="top-n",
                            type="number",
                            min=1,
                            step=1,
                            value=3,
                            style={"width": "100px"},
                        ),
                    ],
                    style={"display": "flex", "gap": "20px"},
                ),
                html.Div(
                    id="plots-container",
                    style={
                        "margin": "0 auto",
                        "padding": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page", n_clicks=0),
                dcc.Location(id="from-page-3-to-page-2-url"),
                html.Button("Prossima pagina →", id="next-page", n_clicks=0),
                dcc.Location(id="from-page-3-to-page-4-url"),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
    ],
    style={"padding": "10px"},
)


#################################################################################################


@callback(
    Output({"type": "plots-character", "index": MATCH}, "figure"),
    [
        Input({"type": "movie-id-character", "index": MATCH}, "value"),
        Input("emotion-multiselect", "value"),
    ],
)
def update_line_plot(movie_id, emotions):
    top_k = dash.callback_context.inputs_list[0]["id"]["index"]

    try:
        return create_line_plot(top_k, movie_id, emotions)
    except Exception as e:
        print("ERROR IN CALLBACK:", e)
        raise e


@callback(
    Output("plots-container", "children"),
    [
        Input("top-n", "value"),
    ],
)
def update_line_plot(top_n):

    children = []

    for top_k in range(top_n):
        div = [
            html.Label(
                "Movie ID:",
                style={"fontWeight": "bold", "marginBottom": "5px"},
            ),
            dcc.Slider(
                id={"type": "movie-id-character", "index": top_k},
                min=1,
                max=8,
                value=1,
                step=1,
            ),
            dcc.Graph(
                id={"type": "plots-character", "index": top_k},
                config={"displayModeBar": False},
                style={"marginTop": "20px"},
            ),
            html.Hr(),
        ]
        children.extend(div)

    return children


@callback(
    Output("from-page-3-to-page-4-url", "pathname"),
    Input("next-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    # se la presentazione è finita, mettere /end
    return "/page-4"


@callback(
    Output("from-page-3-to-page-2-url", "pathname"),
    Input("prev-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-2"
