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

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    use_pages=True,
    suppress_callback_exceptions=True,
)


if __name__ == "__main__":
    app.run(debug=True)
