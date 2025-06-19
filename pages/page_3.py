import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


dash.register_page(__name__)


def create_plot(top_n, emotions):

    df_emotions = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/df_emotions.csv",
        encoding="utf-8-sig",
    )
    dialogs = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/dialogs_bert_sentiment.csv",
        encoding="utf-8-sig",
    )

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
    central_characters = pagerank_by_char.head(top_n).index.tolist()

    plots = []

    for character in central_characters:

        char_data = df_emotions[df_emotions["Character Name"] == character]

        fig = px.histogram(
            char_data,
            x="Movie ID",
            color="Emotion",
            barmode="group",
            category_orders={"Movie ID": sorted(char_data["Movie ID"].unique())},
            title=f"Andamento delle emozioni per {character} nei vari film",
            labels={"Movie ID": "Film", "count": "Conteggio emozioni"},
        )

        for trace in fig.data:
            if trace.name not in emotions:
                trace.opacity = 0.2
            else:
                trace.opacity = 1.0

        fig.update_layout(
            legend_title_text="Emozione",
            xaxis_title="Film",
            yaxis_title="Conteggio emozioni",
            bargap=0.2,
            margin=dict(t=60),
        )

        plots.append(fig)

    return plots


#################################################################################################

layout = html.Div(
    [
        html.H1("Parte 2"),
        html.Div(
            html.Div(
                children=[
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
                        value=["joy", "anger", "fear", "sadness"],
                        multi=True,
                        clearable=False,
                        placeholder="Seleziona le emozioni da visualizzare",
                        style={"width": "30%"},
                    ),
                    html.Div(
                        children=[
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
                        ]
                    ),
                ],
                style={"display": "flex", "gap": "20px"},
            )
        ),
        html.Div(
            id="plots-container",
            style={
                "margin": "0 auto",
                "padding": "20px",
            },
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
    Output("plots-container", "children"),
    [
        Input("top-n", "value"),
        Input("emotion-multiselect", "value"),
    ],
)
def update_graph(character, emotions):
    plots = create_plot(character, emotions)
    children = []
    for fig in plots:
        children.append(dcc.Graph(figure=fig))
        children.append(html.Hr())

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
