import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.stats import entropy


dash.register_page(__name__)


def plot(top_k):

    agg_tabella = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/agg_tabella.csv",
        encoding="utf-8-sig",
    )
    dialogs = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/dialogs_bert_sentiment.csv",
        encoding="utf-8-sig",
    )
    best_characters = agg_tabella.head(top_k)["speaker_lower"].tolist()

    def bert_entropy_for_group(df):
        binned = pd.cut(
            df["bert_sentiment"],
            bins=[0.5, 2.5, 3.5, 5.5],
            labels=["neg", "neu", "pos"],
        )
        probs = binned.value_counts(normalize=True)
        return entropy(probs)

    entropy_line = (
        dialogs.copy()
        .assign(speaker_lower=lambda d: d["speaker"].str.lower().str.strip())
        .query("speaker_lower in @best_characters")
        .groupby(["Movie ID", "speaker_lower"])
        .apply(bert_entropy_for_group)
        .reset_index(name="bert_entropy")
    )
    entropy_line = entropy_line.merge(
        agg_tabella[["speaker_lower", "Character_Name"]], on="speaker_lower", how="left"
    )

    fig = px.line(
        entropy_line,
        x="Movie ID",
        y="bert_entropy",
        color="Character_Name",
        markers=True,
        title="Andamento Entropia (BERT) nei migliori personaggi secondari attivi",
        labels={
            "bert_entropy": "Entropia BERT",
            "Movie ID": "Movie ID",
            "Character_Name": "Personaggio",
        },
    )

    fig.update_layout(
        legend_title_text="Personaggio",
        legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1),
        margin=dict(r=200),
    )

    return fig


def heatmap(top_k):
    agg_tabella = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/agg_tabella.csv",
        encoding="utf-8-sig",
    )

    best_characters = agg_tabella.head(top_k)["speaker_lower"].tolist()
    tab_best = agg_tabella[
        agg_tabella["speaker_lower"].isin(best_characters)
    ].set_index("speaker_lower")
    tab_best.head()
    tab_best = tab_best[
        ["bert_entropy_mean", "out_degree_mean", "in_degree_mean", "pagerank_mean"]
    ]

    tab_norm = (tab_best - tab_best.min()) / (tab_best.max() - tab_best.min())
    annotations = tab_best.round(2).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=tab_norm.values,
            x=tab_norm.columns,
            y=tab_norm.index,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="Valore normalizzato"),
            zmin=0,
            zmax=1,
            # text=tab_best.round(2).astype(str),
            text=annotations.values,
            texttemplate="%{text}",
            hoverinfo="text+x+y",
        )
    )

    fig.update_layout(
        title="Top Personaggi - Entropia, Out-degree, In-degree, Pagerank",
        xaxis_title="Metrica",
        yaxis_title="Personaggio",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=20, t=50, b=50),
        height=500,
    )

    return fig


def heatmap_2(top_k):
    agg_tabella = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/agg_tabella.csv",
        encoding="utf-8-sig",
    )
    dialogs = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/dialogs_bert_sentiment.csv",
        encoding="utf-8-sig",
    )
    best_characters = agg_tabella.head(top_k)["speaker_lower"].tolist()

    def bert_entropy_for_group(df):
        binned = pd.cut(
            df["bert_sentiment"],
            bins=[0.5, 2.5, 3.5, 5.5],
            labels=["neg", "neu", "pos"],
        )
        probs = binned.value_counts(normalize=True)
        return entropy(probs)

    entropy_line = (
        dialogs.copy()
        .assign(speaker_lower=lambda d: d["speaker"].str.lower().str.strip())
        .query("speaker_lower in @best_characters")
        .groupby(["Movie ID", "speaker_lower"])
        .apply(bert_entropy_for_group)
        .reset_index(name="bert_entropy")
    )
    entropy_line = entropy_line.merge(
        agg_tabella[["speaker_lower", "Character_Name"]], on="speaker_lower", how="left"
    )
    entropy_line["Movie ID"] = entropy_line["Movie ID"].astype(int)

    # Pivot: riga = personaggio, colonna = film (ID 1-8 ordinati)
    heatmap_df = entropy_line.pivot(
        index="Character_Name", columns="Movie ID", values="bert_entropy"
    )
    heatmap_df = heatmap_df.reindex(columns=range(1, 9))

    annotations = heatmap_df.round(2).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale="YlGnBu",
            colorbar=dict(title="Entropia BERT"),
            text=annotations.values,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>Movie ID %{x}: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Heatmap entropia BERT (personaggi vs film)",
        xaxis_title="Movie ID",
        yaxis_title="Personaggio",
        yaxis=dict(autorange="reversed"),  # per imitare Seaborn
        margin=dict(l=100, r=20, t=50, b=50),
        height=500,
    )

    return fig


def bert_entropy_plots(top_k):
    agg_tabella = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/agg_tabella.csv",
        encoding="utf-8-sig",
    )
    dialogs = pd.read_csv(
        "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/outputs/dialogs_bert_sentiment.csv",
        encoding="utf-8-sig",
    )
    best_characters = agg_tabella.head(top_k)["speaker_lower"].tolist()

    def bert_entropy_for_group(df):
        binned = pd.cut(
            df["bert_sentiment"],
            bins=[0.5, 2.5, 3.5, 5.5],
            labels=["neg", "neu", "pos"],
        )
        probs = binned.value_counts(normalize=True)
        return entropy(probs)

    entropy_line = (
        dialogs.copy()
        .assign(speaker_lower=lambda d: d["speaker"].str.lower().str.strip())
        .query("speaker_lower in @best_characters")
        .groupby(["Movie ID", "speaker_lower"])
        .apply(bert_entropy_for_group)
        .reset_index(name="bert_entropy")
    )
    entropy_line = entropy_line.merge(
        agg_tabella[["speaker_lower", "Character_Name"]], on="speaker_lower", how="left"
    )
    entropy_line["Movie ID"] = entropy_line["Movie ID"].astype(int)

    fig = px.line(
        entropy_line.sort_values(["Movie ID"]),
        x="Movie ID",
        y="bert_entropy",
        color="Character_Name",
        markers=True,
        title="Andamento entropia BERT per personaggio",
        labels={
            "Movie ID": "Movie ID",
            "bert_entropy": "Entropia BERT",
            "Character_Name": "Personaggio",
        },
    )

    fig.update_layout(
        title="Andamento entropia BERT per personaggio",
        xaxis_title="Movie ID",
        yaxis_title="Entropia BERT",
        xaxis=dict(range=[1, 8], dtick=1),
        yaxis=dict(rangemode="tozero"),
        legend_title_text="Personaggio",
        height=500,
        margin=dict(l=50, r=200),
        legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1),
    )

    return fig


#################################################################################################


layout = html.Div(
    [
        html.H1("Parte 1"),
        html.H2("Individuazione e studio dei personaggi secondari"),
        html.P(
            "Ora osserviamo come cambia l'intensità emotiva rilevata con due strumenti distinti: Vader e Bert. Grazie a questi due "
            "strumenti linguistici possiamo capire l'intensità di una frase di un personaggio e calcolarne l'entropia, sotto rappresentata. "
            "I risultati sono sempre ordinati dalla betweeness maggiore alla minore. È inoltre "
            "possibile scegliere lo strumento usato per tale calcolo.",
            style={"marginTop": "30px", "fontSize": "1.1em", "lineHeight": "1.6em"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Top risultati:",
                            style={"fontWeight": "bold", "marginBottom": "5px"},
                        ),
                        dcc.Input(
                            id="top-k-results",
                            type="number",
                            min=1,
                            step=1,
                            value=5,
                            max=17,
                            style={"width": "100px"},
                        ),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "flex-end",
                "marginTop": "20px",
                "marginBottom": "20px",
                "gap": "30px",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            dcc.Graph(id="grafici-container"),
            style={
                "margin": "0 auto",
                "padding": "20px",
            },
        ),
        html.Div(
            dcc.Graph(id="heatmap"),
            style={
                "margin": "0 auto",
                "padding": "20px",
            },
        ),
        html.Div(
            dcc.Graph(id="heatmap-2"),
            style={
                "margin": "0 auto",
                "padding": "20px",
            },
        ),
        html.Div(
            dcc.Graph(id="bert-entropy-plots"),
            style={
                "margin": "0 auto",
                "padding": "20px",
            },
        ),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page", n_clicks=0),
                dcc.Location(id="from-page-2-to-page-1-url"),
                html.Button("Prossima pagina →", id="next-page", n_clicks=0),
                dcc.Location(id="from-page-2-to-page-3-url"),
            ],
            style={"marginTop": "30px", "display": "flex", "gap": "10px"},
        ),
        # dcc.Location(id="page-2-url"),
    ],
    style={"padding": "10px"},
)

#################################################################################################


@callback(
    Output("grafici-container", "figure"),
    [
        Input("top-k-results", "value"),
    ],
)
def update_graph(top_k):
    fig = plot(top_k)

    return fig


@callback(
    Output("heatmap", "figure"),
    [
        Input("top-k-results", "value"),
    ],
)
def update_heatmap(top_k):
    fig = heatmap(top_k)

    return fig


@callback(
    Output("heatmap-2", "figure"),
    [
        Input("top-k-results", "value"),
    ],
)
def update_heatmap_2(top_k):
    fig = heatmap_2(top_k)

    return fig


@callback(
    Output("bert-entropy-plots", "figure"),
    [
        Input("top-k-results", "value"),
    ],
)
def update_bert_entropy_plots(top_k):
    fig = bert_entropy_plots(top_k)

    return fig


@callback(
    Output("from-page-2-to-page-3-url", "pathname"),
    Input("next-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-3"


@callback(
    Output("from-page-2-to-page-1-url", "pathname"),
    Input("prev-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-1"
