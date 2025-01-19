#Author: Allan Ilyasov
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from scipy.stats import gaussian_kde

url = "https://media.geeksforgeeks.org/wp-content/uploads/20240517142248/country_wise_latest.csv"
data = pd.read_csv(url)

data.fillna(0, inplace=True)

data['CFR'] = np.where(data['Confirmed'] > 0, (data['Deaths'] / data['Confirmed']) * 100, 0)
data['RecoveryRate'] = np.where(data['Confirmed'] > 0, (data['Recovered'] / data['Confirmed']) * 100, 0)

grouped_data = data.groupby(['Country/Region', 'WHO Region']).agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum',
    'Active': 'sum',
    'New cases': 'sum',
    'New deaths': 'sum',
    'New recovered': 'sum',
    'CFR': 'mean',
    'RecoveryRate': 'mean'
}).reset_index()
grouped_data.to_csv("cleaned_covid19_data.csv", index=False)

country_options = [{'label': c, 'value': c} for c in sorted(grouped_data['Country/Region'].unique())]

# Decided to make every visualization interactive, a much easier process than using Matplotlib
def create_line_chart(data):
    top_data = data if data['Country/Region'].nunique() == 1 else data.nlargest(5, 'Deaths')
    return px.line(top_data, x='Country/Region', y=['Deaths', 'Confirmed'],
                   title="Top 5 Countries: Deaths and Confirmed Cases",
                   labels={"value": "Count", "variable": "Metric"},
                   markers=True)

def create_scatter_plot(data):
    return px.scatter(data, x='Confirmed', y='Deaths', color='WHO Region',
                      title="Confirmed vs Deaths with WHO Region Colors",
                      labels={"Confirmed": "Confirmed Cases", "Deaths": "Deaths"})

def create_bar_chart(data):
    top_data = data if data['Country/Region'].nunique() == 1 else data.nlargest(10, 'RecoveryRate')
    return px.bar(top_data, x='RecoveryRate', y='Country/Region', orientation='h',
                  title="Top 10 Countries by Recovery Rate",
                  labels={"RecoveryRate": "Recovery Rate (%)", "Country/Region": "Country"},
                  color='RecoveryRate', color_continuous_scale='Spectral')

def create_box_plot(data):
    return px.box(data, x='WHO Region', y='Deaths',
                  title="Deaths Distribution by WHO Region",
                  labels={"WHO Region": "WHO Region", "Deaths": "Deaths"})

def create_faceted_plot(data):
    return px.scatter(data, x='New cases', y='New deaths', facet_col='WHO Region', facet_col_wrap=3,
                      title="New Cases vs New Deaths by WHO Region")

def create_density_plot(data):
    cfr_vals = data['CFR'].dropna().values
    if len(cfr_vals) > 1:  
        kde = gaussian_kde(cfr_vals)
        x_range = np.linspace(cfr_vals.min(), cfr_vals.max(), 200)
        y_density = kde(x_range)

        # Create a density plot with Plotly
        fig = px.area(x=x_range, y=y_density,
                      labels={"x": "Case Fatality Rate (%)", "y": "Density"},
                      title="Distribution of Case Fatality Rates (Density Plot)")
        return fig
    else:
        # Handle cases with insufficient data
        return px.area(title="Distribution of Case Fatality Rates (Insufficient Data)")


app = Dash(__name__)

app.layout = html.Div([
    html.H1("COVID-19 Dashboard"),
    html.Div([
        # Dropdown: Add a dropdown menu to select a specific country or region.
        html.Div([
            html.Label("Select a Country/Region:"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': 'All', 'value': 'all'}] + country_options,
                value='all',
                clearable=False
            )
        ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
    
        #Slider: Add a slider to adjust the deaths displayed on visualizations.
        html.Div([
            html.Label("Select Death Range:"),
            dcc.RangeSlider(
                id='death-range-slider',
                min=0,
                max=int(grouped_data['Deaths'].max()),
                step=1000,
                value=[0, int(grouped_data['Deaths'].max())],  # Default range: 0 to max deaths
                marks={
                    0: '0',
                    10000: '10K',
                    50000: '50K',
                    100000: '100K+',
                    int(grouped_data['Deaths'].max()): f"{int(grouped_data['Deaths'].max())}+"  # Max value
                },
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '80%', 'padding': '0 20px'})

    ]),

    # Graphs
    dcc.Graph(id='line-chart'),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='bar-chart'),
    dcc.Graph(id='box-plot'),
    dcc.Graph(id='faceted-plot'),
    dcc.Graph(id='density-plot')
])


@app.callback(
    [Output('line-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('bar-chart', 'figure'),
     Output('box-plot', 'figure'),
     Output('faceted-plot', 'figure'),
     Output('density-plot', 'figure')],
    [Input('country-dropdown', 'value'),
     Input('death-range-slider', 'value')]
)
def update_charts(selected_country, death_range):
    min_deaths, max_deaths = death_range
    
    filtered = grouped_data[
        (grouped_data['Deaths'] >= min_deaths) &
        (grouped_data['Deaths'] <= max_deaths)
    ]
    
    if selected_country != 'all':
        filtered = filtered[filtered['Country/Region'] == selected_country]

    line_fig = create_line_chart(filtered)
    scatter_fig = create_scatter_plot(filtered)
    bar_fig = create_bar_chart(filtered)
    box_fig = create_box_plot(filtered)
    faceted_fig = create_faceted_plot(filtered)
    density_fig = create_density_plot(filtered)

    return (line_fig, scatter_fig, bar_fig, box_fig, faceted_fig, density_fig)



if __name__ == '__main__':
    app.run_server(debug=True)
