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
import dash
from dash import Input, Output, html, dcc, callback

dash.register_page(__name__)

###############################################################################################

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

##################################################################################################################s

layout = html.Div(
    [
        html.H1("Prima parte"),
        html.H2("Analisi su grafo dell'interazione dei personaggi"),
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
                html.Br(),
                html.Label("Valore soglia (0-1):"),
                dcc.Input(
                    id="threshold-input",
                    type="number",
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.15,
                    style={"width": "100px", "margin-left": "10px"},
                ),
            ],
            style={"margin": "20px"},
        ),
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
        dcc.Graph(id="indicator-graphic"),
        html.Button("Prossima pagina →", id="next-page-1", n_clicks=0),
        dcc.Location(id="page-2-url"),
    ]
)


#################################################################################################


@callback(
    Output("indicator-graphic", "figure"),
    [
        Input("movie--slider", "value"),
        Input("view-type", "value"),
        Input("threshold-input", "value"),
    ],
)
def update_graph(mid, view_type, cut_threshold):
    if view_type == "graph":
        return networkx_to_plotly(graphs[mid], cut_threshold)

    elif view_type == "heatmap":
        return plot_heatmap(graphs[mid], cut_threshold)


# Callback per routing tra tutte le pagine
@callback(
    Output("page-2-url", "pathname"),
    Input("next-page-1", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-2"


#################################################################################################


def networkx_to_plotly(G, cut_threshold):
    pos = nx.spring_layout(G, seed=42, k=0.6)
    edge_x = []
    edge_y = []
    edge_width = []
    edge_widths = []
    edge_alphas = []
    max_weight = max((G[u][v].get("weight") for u, v in G.edges()))
    edge_labels = {
        (u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)
    }  # pesi delle interazioni

    ################################### parte che crea gli archi e li caratterizza con spessore e alfa

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        w = G[u][v].get("weight")
        edge_width.append(2 + 10 * (w / max_weight))
        width = min(20 * w, 5)  # max edge size = 2
        width = 3 if w >= cut_threshold else 0.5
        edge_widths.append(width)
        alpha = 1.0 if w >= cut_threshold else 0.2
        edge_alphas.append(alpha)

    edge_width_full = []
    for w in edge_width:
        edge_width_full.extend([w, w, 0])

    ################################### parte che aggiunge gli archi allo Scatter

    edge_traces = []
    for i, (u, v) in enumerate(G.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=edge_widths[i], color=f"rgba(10,100,10,{edge_alphas[i]})"),
            hoverinfo="none",
            mode="lines",
        )
        edge_traces.append(trace)

    ################################### parte che disegna le frecce
    annotations = []

    for i, (u, v) in enumerate(G.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        alpha = edge_alphas[i]

        # Calcolo punto intermedio leggermente spostato verso il target per inserire la freccia
        dx = x1 - x0
        dy = y1 - y0
        norm = (dx**2 + dy**2) ** 0.5
        if norm == 0:
            continue
        offset = 0.05  # distanza dalla fine del segmento per mettere la freccia

        x_arrow = x0 + (1 - offset) * dx
        y_arrow = y0 + (1 - offset) * dy

        annotations.append(
            dict(
                ax=x0,
                ay=y0,
                x=x_arrow,
                y=y_arrow,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=5,
                arrowsize=3,
                arrowwidth=edge_widths[i] / 5,
                arrowcolor=f"rgba(10,100,10,{alpha})",
                opacity=alpha,
            )
        )

    ################################### parte che riproporziona i nodi in base a page rank

    pagerank = nx.pagerank(G, weight="weight")
    pr_values = list(pagerank.values())
    min_pr, max_pr = min(pr_values), max(pr_values)

    node_sizes = [
        10 + 30 * (pagerank[n] - min_pr) / (max_pr - min_pr) if max_pr > min_pr else 20
        for n in G.nodes()
    ]

    ################################### parte che aggiunge i nodi allo Scatter

    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    ################################### parte che aggiunge le note ai nodi

    info_node = []
    for node in G.nodes():
        pr = pagerank[node]
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        info = f"{node}<br>Pagerank: {pr:.2f}<br>Out degree: {out_degree:.2f} <>In degree: {in_degree:.2f}"
        node_text.append(info)

    ################################### parte che caratterizza lo Scatter

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
            color=[pagerank[n] for n in G.nodes()],
            size=node_sizes,
            colorbar=dict(
                thickness=15, title="Pagerank", xanchor="left", title_side="right"
            ),
            line_width=2,
        ),
    )

    ################################### parte che colora i nodi in base pagerank

    pagerank = nx.pagerank(G, weight="weight")
    node_trace.marker.color = [pagerank[n] for n in G.nodes()]

    ################################### parte che disegna il grafo

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title="Grafo delle interazioni",
            title_font_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
        ),
    )
    return fig


def plot_heatmap(G, cut_threshold):
    adj_matrix = nx.to_pandas_adjacency(G, weight="weight")

    total_interactions = {
        node: sum(G[node][nbr]["weight"] for nbr in G.successors(node))
        + sum(G[pre][node]["weight"] for pre in G.predecessors(node))
        for node in G.nodes()
    }

    def format_label(name):
        return f"<b>{name}</b>" if total_interactions[name] >= cut_threshold else name

    formatted_x = [format_label(name) for name in adj_matrix.columns]
    formatted_y = [format_label(name) for name in adj_matrix.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=adj_matrix.values,
            x=formatted_x,
            y=formatted_y,
            colorscale="YlOrRd",
        )
    )
    fig.update_layout(
        title="Heatmap delle interazioni tra personaggi",
        xaxis_title="Personaggio",
        yaxis_title="Personaggio",
    )
    return fig
