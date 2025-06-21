import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import os

dash.register_page(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_final_df_nrc = os.path.join("outputs/final_df.csv")
final_df_nrc = pd.read_csv(path_final_df_nrc, encoding="utf-8-sig")

layout = html.Div(
    [
        html.H1("Raccomandatore di scene per emozione", style={"marginBottom": "10px"}),
        html.P(
            "Seleziona un'emozione per scoprire le scene più intense della saga:",
            style={"fontSize": "1.1em"},
        ),
        html.Div(
            [
                html.Label(
                    "Emozione:", style={"fontWeight": "bold", "marginRight": "10px"}
                ),
                dcc.Dropdown(
                    id="emotion-dropdown",
                    options=[
                        {"label": "Gioia", "value": "joy"},
                        {"label": "Rabbia", "value": "anger"},
                        {"label": "Paura", "value": "fear"},
                        {"label": "Tristezza", "value": "sadness"},
                        {"label": "Amore", "value": "love"},
                        {"label": "Sorpresa", "value": "surprise"},
                    ],
                    value="joy",
                    clearable=False,
                    style={"width": "260px"},
                ),
            ],
            style={"margin": "28px 0"},
        ),
        html.Div(id="scene-recommendations"),
        html.Div(
            [
                html.Button("← Pagina precedente", id="prev-page", n_clicks=0),
                dcc.Location(id="from-page-5-to-page-4-url"),
                html.Button("Prossima pagina →", id="next-page", n_clicks=0),
                dcc.Location(id="from-page-5-to-page-6-url"),
            ],
            style={"marginTop": "40px", "display": "flex", "gap": "10px"},
        ),
    ],
    style={"maxWidth": "800px", "margin": "0 auto", "padding": "38px 12px"},
    className="page-enter",
)

@callback(
    Output("scene-recommendations", "children"),
    Input("emotion-dropdown", "value"),
)
def update_scene_list(selected_emotion):
    df = final_df_nrc
    filtered = df[df["Emozione"].str.lower() == selected_emotion]
    filtered = filtered.sort_values(by="Intensità_media", ascending=False).head(5)

    if filtered.empty:
        return html.Div("Nessuna scena trovata per questa emozione.", style={"color": "red", "marginTop": "30px"})

    items = []
    for _, row in filtered.iterrows():
        items.append(
            html.Div([
                html.H4(f"Film {row['Film']} | Capitolo {row['Capitolo']}", style={"marginBottom": "4px"}),
                html.Div(f"Intensità media: {row['Intensità_media']:.2f}", style={"fontWeight": "bold", "marginBottom": "4px"}),
                html.Div(row['Riassunto_narrativo'], style={"fontStyle": "italic", "marginBottom": "8px"}),
            ],
            style={
                "background": "#f8f9fa",
                "borderRadius": "12px",
                "boxShadow": "0 1px 6px #0002",
                "padding": "16px 20px",
                "marginBottom": "20px",
            })
        )
    return html.Div(items)

@callback(
    Output("from-page-5-to-page-6-url", "pathname"),
    Input("next-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_next_page(n_clicks):
    return "/page-6"

@callback(
    Output("from-page-5-to-page-4-url", "pathname"),
    Input("prev-page", "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True,
)
def go_to_previous_page(n_clicks):
    return "/page-4"
