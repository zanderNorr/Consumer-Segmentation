from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, html, dcc, Output, Input

df = pd.read_csv("preprocessed_marketing_campaign.csv")

x1 = df["Income"]

drop_down_selection = [
    {"label": "Wine", "value": "MntWines"},
    {"label": "Fruits", "value": "MntFruits"},
    {"label": "Meat", "value": "MntMeatProducts"},
    {"label": "Fish", "value": "MntFishProducts"},
    {"label": "Sweet", "value": "MntSweetProducts"},
    {"label": "Gold", "value": "MntGoldProds"},
    {"label": "Total Purchases", "value": "MntTotalPurchased"},
]

histogram_drop_down_selection = [
    {"label": "Year_Birth", "value": 'Year_Birth'},
    {"label": "Education", "value": "Education"},
    {"label": "Marital Status", "value": "Marital_Status"},
    {'label': "Income", 'value': 'Income'},
    {"label": "Days since last purchase", 'value': "Recency"},
    {"label": "Number of Deal Purchases",
     'value': "NumDealsPurchases"},
    {"label": "Number of Web Purchases", 'value': "NumWebPurchases"},
    {"label": "Number of Catalog Purchases",
     "value": "NumCatalogPurchases"},
    {"label": "Number of Store Purchases", "value": 'NumStorePurchases'}
]
app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Histogram of Selected frequency",
            style={"textAlign": "center"}
        ),
        dcc.Slider(id='Histogram-bin-slider', min=5, max=30, value=5, step=1),
        dcc.Dropdown(
            id='Histogram-x-selector',
            options=histogram_drop_down_selection,
            value="Income",
        ),
        dcc.Graph(id='Histogram'),
        html.H1(
            "Scatterplot of Consumer Income vs Selected Sub-category Purchase",
            style={"textAlign": "center"},
        ),
        dcc.Dropdown(
            id="Scatterplot-selection",
            options=drop_down_selection,
            value="MntTotalPurchased",
        ),
        dcc.Graph(id="Scatterplot"),
        html.H1(
            "K-Means Clustering of Consumer Income vs Group Purchases",
            style={"textAlign": "center"},
        ),
        dcc.Dropdown(
            id="K-means-selection",
            options=drop_down_selection,
            value="MntTotalPurchased",
        ),
        dcc.Slider(id="K-Means-slider", min=1, max=10, value=5, step=1),
        dcc.Graph(id="K-Means-plot"),
        html.H1(
            "Gaussian Mixture Model (GMM) Clustering Of Consumer Income vs Group Purchases",
            style={"textAlign": "center"},
        ),
        dcc.Dropdown(
            id="GMM-selection",
            options=drop_down_selection,
            value="MntTotalPurchased",
        ),
        dcc.Slider(id="GMM-slider", min=1, max=10, value=5, step=1),
        dcc.Graph(id="GMM-plot"),
    ]
)


@app.callback(Output("Histogram", 'figure'), [Input("Histogram-bin-slider", "value"), Input('Histogram-x-selector', 'value')])
def update_histogram(bins, x1):
    fig = px.histogram(
        df,
        x=df[x1],
        nbins=bins,
        title=f"Histogram of {x1} with {bins} bins",
        labels={"x": f"{x1}", "y": "Frequency"},
    )
    return fig


@app.callback(Output("K-Means-plot", "figure"), [Input("K-Means-slider", "value"), Input('K-means-selection', 'value')])
def update_GMM_graph(k, variable):
    x2 = df[variable]
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    gmm = KMeans(n_clusters=k)
    gmm.fit(X)
    labels = gmm.predict(X)
    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=labels,
        color_continuous_scale="plasma",
        title=f"K-Means Clustering of Income vs {variable} with # of clusters = {k}",
        labels={"x": "Income (in $)", "y": "Total Spending (in $)"},
    )
    return fig


@app.callback(Output("GMM-plot", "figure"), [Input("GMM-slider", "value"), Input('GMM-selection', 'value')])
def update_GMM_graph(n, variable):
    x2 = df[variable]
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    gmm = GaussianMixture(n_components=n, covariance_type="full")
    gmm.fit(X)
    labels = gmm.predict(X)
    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=labels,
        color_continuous_scale="plasma",
        title=f"Gaussian Mixture Model (GMM) of Income vs {variable} with # of clusters = {n}",
        labels={"x": "Income (in $)", "y": "Total Spending (in $)"},
    )
    return fig


@app.callback(Output("Scatterplot", "figure"), Input("Scatterplot-selection", "value"))
def update_scatterplot(y_type):
    fig = px.scatter(
        x=x1,
        y=df[y_type],
        labels={"x": "Income (in $)", "y": "Total Spending (in $)"},
        title=f"Scatterplot of Consumer Income vs {y_type}"
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
