"""import dash
from dash import html, dcc, callback, Input, Output
import dash
from dash import Input, Output, html, dcc, callback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


dash.register_page(__name__)


def plot(criterion, number_films, top_k):
    emotional_characters = pd.read_csv(
        "outputs/emotional_characters.csv",
        encoding="utf-8-sig",
    )
    dialogs = pd.read_csv(
        "outputs/dialogs_bert_vader.csv"
    )

    filtered = emotional_characters[
        emotional_characters["num_films"] >= int(number_films)
    ]
    pre_sorted = filtered.sort_values(by="betweenness_mean", ascending=False)
    sorted = pre_sorted.sort_values(by=f"{criterion}", ascending=False)
    top_k_characters = sorted.head(top_k)

    plots = []

    for character in top_k_characters["Character_Name"].tolist():
        # Prendi tutti i dialoghi del personaggio su tutti i film
        sub = dialogs[
            dialogs["speaker"].str.lower().str.strip() == character.lower().strip()
        ]

        # Raggruppa per Film + Capitolo per mantenere ordine temporale globale
        sub = sub.sort_values(["Movie ID", "Chapter ID"])
        sub["film_chapter"] = (
            sub["Movie ID"].astype(str) + "-" + sub["Chapter ID"].astype(str)
        )

        agg = (
            sub.groupby(["Movie ID", "Chapter ID"])
            .agg(
                vader_compound_mean=("vader_compound", "mean"),
                bert_sentiment_mean=("bert_sentiment", "mean"),
                vader_count=("vader_compound", "size"),
            )
            .reset_index()
        )

        # Crea una colonna unica che rappresenta film-capitolo
        agg["film_chapter"] = (
            agg["Movie ID"].astype(str) + "-" + agg["Chapter ID"].astype(str)
        )

        # Line plot
        fig1 = go.Figure()

        fig1.add_trace(
            go.Scatter(
                x=agg["film_chapter"],
                y=agg["vader_compound_mean"],
                mode="lines+markers",
                name="VADER Sentiment",
                line=dict(color="goldenrod"),
                marker=dict(symbol="circle"),
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=agg["film_chapter"],
                y=agg["bert_sentiment_mean"],
                mode="lines+markers",
                name="BERT Sentiment",
                line=dict(color="mediumseagreen"),
                marker=dict(symbol="circle"),
            )
        )

        fig1.update_layout(
            title=f"Andamento emozionale di {character} su tutti i film",
            xaxis_title="Film-Capitolo",
            yaxis_title="Sentiment medio",
            xaxis=dict(tickangle=-45),
            legend=dict(x=0, y=1),
            margin=dict(t=50, b=100),
            height=500,
        )

        # Heatmap
        pivot = agg.pivot_table(
            index="film_chapter", values=["vader_compound_mean", "bert_sentiment_mean"]
        ).T

        # Convertiamo in long format per Plotly
        pivot_long = pivot.reset_index().melt(
            id_vars="index", var_name="film_chapter", value_name="sentiment"
        )
        pivot_long.rename(columns={"index": "model"}, inplace=True)

        fig2 = px.imshow(
            pivot.values,
            labels=dict(x="Film-Capitolo", y="Modello", color="Sentiment"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )

        fig2.update_layout(
            title=f"Heatmap emozionale - {character} (Film-Capitolo)",
            xaxis_title="Film-Capitolo",
            yaxis_title="",
            height=250,
            margin=dict(t=50, b=50),
        )

        plots.append((fig1, fig2))

    return plots


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
                            "Criterio di scelta:",
                            style={"fontWeight": "bold", "marginBottom": "5px"},
                        ),
                        dcc.Dropdown(
                            id="criterio-dropdown",
                            options=[
                                {
                                    "label": "Vader entropy mean",
                                    "value": "vader_entropy_mean",
                                },
                                {
                                    "label": "Bert entropy mean",
                                    "value": "bert_entropy_mean",
                                },
                            ],
                            value="vader_entropy_mean",
                            clearable=False,
                            style={"width": "250px"},
                        ),
                    ],
                    style={"marginRight": "40px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Min. numero di film per personaggio:",
                            style={"fontWeight": "bold", "marginBottom": "5px"},
                        ),
                        dcc.Input(
                            id="threshold-movies",
                            type="number",
                            min=1,
                            max=8,
                            step=1,
                            value=5,
                            style={"width": "100px"},
                        ),
                    ]
                ),
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
                "gap": "30px",  # per armonizzare meglio la distanza
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            id="grafici-container",
            style={
                "margin": "0 auto",  # centra il contenitore orizzontalmente
                "padding": "20px",  # spazio interno per respiro visivo
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
    Output("grafici-container", "children"),
    [
        Input("criterio-dropdown", "value"),
        Input("threshold-movies", "value"),
        Input("top-k-results", "value"),
    ],
)
def update_graph(criterion, number_films, top_k):
    plots = plot(criterion, number_films, top_k)
    children = []
    for idx, (fig, fig_heat) in enumerate(plots):
        children.append(html.H4(f"Risultato {idx + 1}"))
        children.append(dcc.Graph(figure=fig))
        children.append(dcc.Graph(figure=fig_heat))
        children.append(html.Hr())

    return children


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
"""
