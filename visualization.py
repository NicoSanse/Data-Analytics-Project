from readline import redisplay
import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
import pandas as pd
import re
import csv
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from dash.dependencies import Input, Output

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

characters = pd.read_csv(
    "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/Harry_Potter_Movies/Characters.csv",
    encoding="latin-1",
)
movies = pd.read_csv(
    "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/Harry_Potter_Movies/Movies.csv",
    encoding="utf-8-sig",
)
dialogues = pd.read_csv(
    "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/Harry_Potter_Movies/Dialogue.csv",
    encoding="latin-1",
)
chapters = pd.read_csv(
    "/Users/nicosanse/Desktop/Uni/1' sem/Lab/Data Analytics/Data Analytics Project/Harry_Potter_Movies/Chapters.csv",
    encoding="latin-1",
)


###############################################################################################


dialogs = dialogues.merge(chapters[["Chapter ID", "Movie ID"]], on="Chapter ID")
dialogs = dialogs.sort_values(by=["Movie ID", "Chapter ID", "Dialogue ID"])
dialogs = dialogs.merge(
    characters[["Character ID", "Character Name"]], on="Character ID"
)
dialogs["speaker"] = dialogs["Character Name"].str.lower().str.strip()

###############################################################################################

graphs = {}  # film_id → grafo
centrality_results = {}  # film_id → df con centralità

for mid, g in dialogs.groupby("Movie ID"):

    G = nx.DiGraph()
    speakers = g["Character Name"].tolist()
    pairs = [(a, b) for a, b in zip(speakers[:-1], speakers[1:]) if a != b]
    interaction_count = defaultdict(int)
    interlocutors = defaultdict(set)

    for a, b in pairs:
        interaction_count[(a, b)] += 1
        interlocutors[a].add(b)
    max_count = max(interaction_count.values())
    for (a, b), count in interaction_count.items():
        norm_weight = count / max_count
        G.add_edge(a, b, weight=norm_weight)

    graphs[mid] = G

    betweenness = nx.betweenness_centrality(G, weight="weight")

    #  Calcolo degree centrality su grafo non orientato temporaneo
    G_undirected = G.to_undirected()
    degree = nx.degree_centrality(G_undirected)

    # Crea DataFrame centralità
    df_centrality = pd.DataFrame(
        {
            "Character Name": list(betweenness.keys()),
            "Betweenness": [betweenness[k] for k in betweenness.keys()],
            "Centrality": [degree[k] for k in betweenness.keys()],
            "Movie ID": mid,
        }
    )

    centrality_results[mid] = df_centrality

    centrality_all = pd.concat(centrality_results, ignore_index=True)

    # Elimina le colonne che causano conflitto PRIMA del merge
    dialogs = dialogs.drop(columns=["Centrality", "Betweenness"], errors="ignore")

    # Ora puoi eseguire il merge tranquillamente
    """dialogs = dialogs.merge(
        centrality_all, on=["Movie ID", "Character Name"], how="left"
    )"""

    # Usa pesi per le centralità
    degree_c = nx.degree_centrality(G)  # NON usa pesi (opzionale)
    in_deg = G.in_degree(weight="weight")
    out_deg = G.out_degree(weight="weight")
    pagerank = nx.pagerank(G, weight="weight")

    df_c = pd.DataFrame(
        {
            "character": list(G.nodes),
            "in_degree": [in_deg[n] for n in G.nodes],
            "out_degree": [out_deg[n] for n in G.nodes],
            "pagerank": [pagerank[n] for n in G.nodes],
            "raw_degree": [degree_c[n] for n in G.nodes],
            "movie_id": mid,
        }
    )

    centrality_results[mid] = df_c

    centrality_df = pd.concat(centrality_results.values(), ignore_index=True)

##################################################################################################################

app.layout = html.Div(
    [
        html.H1("Analisi Interazioni Personaggi - Harry Potter"),
        html.Div(
            [
                html.Label("Seleziona un film:"),
                dcc.Slider(
                    id="movie--slider",
                    min=min(graphs.keys()),
                    max=max(graphs.keys()),
                    value=min(graphs.keys()),
                    marks={mid: f"Film {mid}" for mid in graphs.keys()},
                    step=None,
                ),
            ],
            style={"margin": "40px"},
        ),
        html.Div(
            [
                html.Label("Tipo di visualizzazione:"),
                dcc.RadioItems(
                    id="view-type",
                    options=[
                        {"label": "Grafo", "value": "graph"},
                        {"label": "Heatmap", "value": "heatmap"},
                    ],
                    value="graph",
                    labelStyle={"display": "inline-block", "margin-right": "20px"},
                ),
            ],
            style={"margin": "20px"},
        ),
        dcc.Graph(id="indicator-graphic"),
    ]
)

#################################################################################################


@app.callback(
    Output("indicator-graphic", "figure"),
    [Input("movie--slider", "value"), Input("view-type", "value")],
)
def update_graph(mid, view_type):
    if view_type == "graph":
        return networkx_to_plotly(graphs[mid])

    elif view_type == "heatmap":
        return plot_heatmap(graphs[mid])


#################################################################################################


def networkx_to_plotly(G):
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15, title="Pagerank", xanchor="left", titleside="right"
            ),
            line_width=2,
        ),
    )

    # Colora i nodi in base al pagerank
    pagerank = nx.pagerank(G, weight="weight")
    node_trace.marker.color = [pagerank[n] for n in G.nodes()]

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Grafo delle interazioni",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )
    return fig


def plot_heatmap(G):
    adj_matrix = nx.to_pandas_adjacency(G, weight="weight")
    fig = go.Figure(
        data=go.Heatmap(
            z=adj_matrix.values,
            x=adj_matrix.columns,
            y=adj_matrix.index,
            colorscale="YlOrRd",
        )
    )
    fig.update_layout(
        title="Heatmap delle interazioni tra personaggi",
        xaxis_title="Personaggio",
        yaxis_title="Personaggio",
    )
    return fig


#################################################################################################


if __name__ == "__main__":
    app.run(debug=True)
