# TO DO:
# - Better spike detection and handling in evaluate_forecast_for_item
# - Forecast validation
# - Status filtering in analysis tab
# - Descriptive status messages
# - Filter template should use all filter fields

print("[DEBUG] ===== GUI LOADED v2.1 WITH SELECTED_ROWS CALLBACK =====", flush=True)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import os
import sys
import math
from datetime import datetime, timedelta, timezone
import time
import pandas as pandas_mod

APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

# --- Add/Remove Watchlist via forecast-table cell click ---
from dash.exceptions import PreventUpdate

import numpy as np
from backend import (
    analyze_forecast_gui,
    get_item_timeseries,
    get_update_times,
    save_filter_template,
    load_filter_template,
    list_filter_templates,
    delete_filter_template,
    get_latest_prices
)
from forecast_helpers import forecast_avg_price
def evaluate_forecast_for_item(hist, forecast_horizon=60, train_ratio=0.8):
    """
    Split hist into train/test, forecast test points, and print error metrics.
    hist: list of dicts with 'timestamp', 'avgHighPrice', 'avgLowPrice'
    forecast_horizon: minutes ahead to forecast
    train_ratio: fraction of data to use for training
    """
    import numpy as np
    if not hist or len(hist) < 10:
        print("Not enough data for evaluation.")
        return
    # Use mid price for evaluation
    prices = [(e.get('avgHighPrice',0) + e.get('avgLowPrice',0))/2 for e in hist if e.get('avgHighPrice') is not None and e.get('avgLowPrice') is not None]

def exclude_outliers(series):
    """Remove outliers using IQR method"""
    if len(series) < 4:
        return series
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]
    n = len(prices)
    split = int(n * train_ratio)
    train_hist = hist[:split]
    test_hist = hist[split:]
    test_prices = prices[split:]
    forecasts = []
    for i, e in enumerate(test_hist):
        # Use all data up to this test point for forecasting
        hist_for_forecast = train_hist + test_hist[:i]
        forecast = forecast_avg_price(hist_for_forecast, forecast_horizon)
        forecasts.append(forecast if forecast is not None else 0)
    # Error metrics
    test_prices = np.array(test_prices)
    forecasts = np.array(forecasts)
    mae = np.mean(np.abs(test_prices - forecasts))
    rmse = np.sqrt(np.mean((test_prices - forecasts)**2))
    print(f"Forecast evaluation (horizon={forecast_horizon} min): MAE={mae:.2f}, RMSE={rmse:.2f}, N={len(test_prices)}")
    return mae, rmse, test_prices, forecasts




# --- Dash Implementation ---


app = dash.Dash(__name__, suppress_callback_exceptions=True)

filter_defaults = {
    'MIN_FORECAST_PROFIT': 1,
    'MIN_POTENTIAL_PROFIT': 10000,
    'MIN_VOLUME_POTENTIAL': 6,
    'MIN_FORECAST_ROI': 5,
    'FORECAST_HOURS': 168
}

one_h_dt, five_m_dt = get_update_times()

# Convert UTC times to local time for display
if one_h_dt:
    one_h_utc = one_h_dt.replace(tzinfo=timezone.utc)
    one_h_local = one_h_utc.astimezone()
    hour_end_local = one_h_local + timedelta(hours=1)
    one_h_str = one_h_local.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + hour_end_local.strftime('%H:%M:%S')
else:
    one_h_str = 'Unavailable'

if five_m_dt:
    five_m_utc = five_m_dt.replace(tzinfo=timezone.utc)
    five_m_local = five_m_utc.astimezone()
    five_m_end_local = five_m_local + timedelta(minutes=5)
    five_m_str = five_m_local.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + five_m_end_local.strftime('%H:%M:%S')
else:
    five_m_str = 'Unavailable'

app.layout = html.Div([
    html.Div([
        html.Div('OSRS GE Watch', style={'fontSize': '24px', 'fontWeight': 'bold', 'flex': '1'}),
        html.Button(
            '🌙 Dark Mode',
            id='dark-mode-toggle',
            n_clicks=0,
            style={'padding': '8px 16px', 'marginRight': '10px', 'cursor': 'pointer', 'background': '#333', 'color': '#fff', 'border': 'none', 'borderRadius': '4px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '12px', 'marginBottom': '12px', 'background': '#f5f5f5', 'borderBottom': '1px solid #ddd'}),
    dcc.Tabs(id='main-tabs', value='analysis', children=[
        dcc.Tab(label='Analysis', value='analysis'),
        dcc.Tab(label='Watchlist', value='watchlist'),
    ]),
    html.Div(id='tab-content'),
    dcc.Store(id='forecasted-prices-store'),
    dcc.Store(id='dark-mode-store', data={'dark_mode': False}),
], id='app-container', style={'fontFamily': 'Arial, sans-serif', 'minHeight': '100vh'})

@app.callback(
    [Output('dark-mode-store', 'data'), Output('dark-mode-toggle', 'style')],
    Input('dark-mode-toggle', 'n_clicks'),
    State('dark-mode-store', 'data'),
    prevent_initial_call=False
)
def toggle_dark_mode(n_clicks, store_data):
    dark_mode = store_data.get('dark_mode', False) if store_data else False
    if n_clicks and n_clicks > 0:
        dark_mode = not dark_mode
    
    store_data = {'dark_mode': dark_mode}
    
    btn_style = {
        'padding': '8px 16px',
        'marginRight': '10px',
        'cursor': 'pointer',
        'background': '#1a1a1a' if dark_mode else '#333',
        'color': '#fff',
        'border': 'none',
        'borderRadius': '4px',
        'fontWeight': 'bold'
    }
    
    return store_data, btn_style


@app.callback(
    Output('app-container', 'className'),
    Input('dark-mode-store', 'data')
)
def update_theme(store_data):
    dark_mode = store_data.get('dark_mode', False) if store_data else False
    return 'dark-mode' if dark_mode else ''







## Remove the separate toggle_watchlist_cell callback and merge its logic into update_table_and_times below



# --- Tab content callback ---
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'analysis':
        # Render the original analysis layout
        return html.Div([
            html.H2('OSRS GE Forecast Analysis (Dash)'),
            html.Div([
                html.Label('Search Item Name', style={'marginRight': '10px'}),
                dcc.Input(id='item-name-search', type='text', placeholder='Enter item name', debounce=True, style={'width': '220px', 'marginRight': '30px'}),
                dcc.Checklist(
                    id='filter-by-watchlist',
                    options=[{'label': 'Show only watchlist items', 'value': 'watchlist'}],
                    value=[],
                    style={'marginRight': '30px'}
                ),
                html.Label('Risk Filter:', style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='risk-filter',
                    options=[
                        {'label': 'All Items', 'value': 'all'},
                        {'label': 'Low Risk Only', 'value': 'low'},
                        {'label': 'Low + Medium Risk', 'value': 'low_medium'},
                        {'label': 'Exclude High Risk', 'value': 'exclude_high'},
                    ],
                    value='all',
                    clearable=False,
                    style={'width': '180px', 'display': 'inline-block', 'marginRight': '30px'}
                ),
                html.Label('Status Filter:', style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='status-filter',
                    options=['Normal', 'Volume imbalance', 'Possible manipulation', 'Abnormally high', 'Abnormally low', 'Spread unusually high'
                             ,'Low price dump', 'High price spike', 'Low-Volume spike', 'High-Volume spike'
                             ,'Low-Volume low', 'Low-Volume high', 'High-Volume low', 'High-Volume high'],  # Will be set dynamically

                    value=[],
                    multi=True,
                    clearable=True,
                    style={'width': 'auto', 'minWidth': '250px', 'display': 'inline-block', 'marginRight': '30px'},
                    maxHeight=300,
                    optionHeight=36,
                    placeholder='Select status/warning...'
                ),
                dcc.Checklist(
                    id='status-filter-exclude',
                    options=[{'label': 'Exclude selected', 'value': 'exclude'}],
                    value=[],
                    style={'display': 'inline-block'}
                ),

            
            ], style={'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center'}),
            html.Div(id='update-times', style={'marginBottom': '12px', 'fontSize': '16px'}),
            dcc.Interval(id='refresh-interval', interval=24*60*60*1000, n_intervals=0, disabled=True),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label('Forecast Strategy', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='forecast-strategy',
                           options = [
                                {'label': 'Weighted Moving Avg (WMA)', 'value': 'wma'},
                                {'label': 'WMA + Trend', 'value': 'wma_trend'},
                                # {'label': "Brown's Double Moving Avg", 'value': 'brown'},
                                {'label': 'Exp Weighted Moving Avg (EWMA)', 'value': 'ewma'},

                                {'label': 'Linear Extrapolation', 'value': 'linear'},
                                {'label': 'Polynomial Extrapolation', 'value': 'poly'},

                                # {'label': 'ARIMA', 'value': 'arima'},
                                {'label': 'Holt-Winters', 'value': 'holt_winters'},
                                # {'label': 'SARIMAX', 'value': 'sarimax'},
                                # {'label': 'Prophet', 'value': 'prophet'},

                                # ---- ROBUST / STABLE MODELS ----
                                # {'label': 'Robust EWMA (Spike-Resistant)', 'value': 'robust_ewma'},
                                # {'label': 'Median Trend (Stable)', 'value': 'median_trend'},
                                {'label': 'Median Reversion (Weekly Bias)', 'value': 'median_reversion_weekly'},
                                {'label': 'Long-Range Stable Forecast', 'value': 'long_range'},

                                # ---- RECOMMENDED DEFAULT ----
                                {'label': 'Auto (Horizon-Aware Mix)', 'value': 'mix'},
                            ],
                            value='wma',
                            clearable=False,
                            style={'width': 'auto', 'minWidth': '250px'}
                        ),], style={'display': 'flex', 'width': 'auto', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label('Forecast Sell Time', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='forecast-sell-time', type='number', value=60, min=5, step=5, debounce=True, style={'width': '40px', 'marginRight': '30px'}),
                        html.Label('Min Price', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-price', type='number', value=0, debounce=True, style={'width': '70px', 'marginRight': '30px'}),
                        html.Label('Max Price', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='max-price', type='number', value=0, debounce=True, style={'width': '70px', 'marginRight': '30px'}),
                        html.Label('Min Avg Daily Volume', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-avg-daily-volume', type='number', value=0, debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Max Avg Trade Time (min)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='max-avg-trade-time', type='number', value=0, debounce=True, style={'width': '120px', 'marginRight': '30px'}),
                    ], style={'display': 'flex', 'width': 'auto', 'marginBottom': '12px'}),
                    html.Div([
                        html.Label('Buy Price Type', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='buy-price-type',
                            options=[
                                {'label': 'Low Price', 'value': 'low'},
                                {'label': 'High Price', 'value': 'high'},
                                {'label': 'Avg Low Price (30min)', 'value': 'avg_low_30min'},
                            ],
                            value='low',
                            clearable=False,
                            style={'width': '200px', 'marginRight': '30px'}
                        ),
                        html.Label('Forecast Price Type', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='forecast-price-type',
                            options=[
                                {'label': 'Low Price', 'value': 'low'},
                                {'label': 'High Price', 'value': 'high'},
                                {'label': 'Average Price', 'value': 'avg'},
                            ],
                            value='avg',
                            clearable=False,
                            style={'width': '140px', 'marginRight': '30px'}
                        ),
                    ], style={'display': 'flex', 'width': 'auto', 'marginBottom': '12px'}),
                    html.Div([
                        html.Label('Min Forecast Profit', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-profit', type='number', value=filter_defaults['MIN_FORECAST_PROFIT'], debounce=True, style={'width': '70px', 'marginRight': '30px'}),
                        html.Label('Min Potential Profit', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-potential-profit', type='number', value=filter_defaults['MIN_POTENTIAL_PROFIT'], debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Min Volume Potential', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-volume', type='number', value=filter_defaults['MIN_VOLUME_POTENTIAL'], debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Min ROI', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-roi', type='number', value=filter_defaults['MIN_FORECAST_ROI'], debounce=True, style={'width': '60px', 'marginRight': '30px'}),
                        html.Label('Forecast Hours', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='forecast-hours', type='number', value=filter_defaults['FORECAST_HOURS'], debounce=True, style={'width': '60px', 'marginRight': '30px'}),
                        html.Label('Volume Power', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='volume-power', type='number', value=1, debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Max Qty Factor', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='max-qty-factor', type='number', value=1, min=0, step=0.1, debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                    ], style={'display': 'flex', 'width': 'auto', 'marginBottom': '16px'}),
                ], style={'display': 'block', 'flexWrap': 'wrap', 'gap': '6px', 'alignItems': 'center', 'marginBottom': '12px'}),
                html.Div([
                    html.Label('Trend', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='trend-filter',
                        options=[
                            {'label': 'Any', 'value': ''},
                            {'label': 'UP', 'value': 'UP'},
                            {'label': 'DOWN', 'value': 'DOWN'},
                            {'label': 'FLAT', 'value': 'FLAT'},
                        ],
                        value='',
                        clearable=False,
                        style={'width': '80px', 'marginRight': '30px'}
                    ),
                    html.Label('Top N', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                    dcc.Input(id='top-n', type='number', value=10, min=1, debounce=True, style={'width': '60px', 'marginRight': '30px'}),
                    html.Label('Sort by Attribute', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='sort-attribute',
                        options=[
                            {'label': 'Potential Profit', 'value': 'potential_profit'},
                            {'label': 'Volume Potential', 'value': 'volume_potential'},
                            {'label': 'Profit', 'value': 'profit'},
                            {'label': 'ROI', 'value': 'roi'},
                            {'label': 'Forecast Price', 'value': 'forecast_price'},
                            {'label': 'Low Price', 'value': 'lowP'},
                            {'label': 'High Price', 'value': 'highP'},
                            {'label': 'Avg Daily Volume', 'value': 'avg_daily_volume'},
                            {'label': 'Buy Limit', 'value': 'buy_limit'},
                        ],
                        value='potential_profit',
                        clearable=False,
                        style={'width': '120px', 'marginRight': '30px'}
                    ),
                    html.Button('Apply Filters', id='apply-filters', n_clicks=0, style={'height': '40px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '6px', 'alignItems': 'center', 'marginBottom': '12px'}),
                html.Div([
                    html.Label('Template:', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='template-dropdown',
                        options=[{'label': name, 'value': name} for name in list_filter_templates()],
                        value=list_filter_templates()[0] if list_filter_templates() else None,
                        placeholder='Select template',
                        style={'width': '180px', 'marginRight': '10px'}
                    ),
                    dcc.Input(id='template-name', type='text', placeholder='Template name', style={'width': '120px', 'marginRight': '10px'}),
                    html.Button('Save', id='save-template', n_clicks=0, style={'marginRight': '6px'}),
                    html.Button('Load', id='load-template', n_clicks=0, style={'marginRight': '6px'}),
                    html.Button('Delete', id='delete-template', n_clicks=0, style={'marginRight': '6px'}),
                    html.Span(id='template-status', style={'marginLeft': '10px', 'color': '#888'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '8px'}),
                # Insert Add selected to watchlist button and store
                html.Button('Add selected to the watchlist', id='add-selected-to-watchlist-btn', n_clicks=0, style={'marginBottom': '10px', 'marginRight': '10px'}),
                dcc.Store(id='selected-row-store'),
                html.Div(id='add-selected-status', style={'marginLeft': '12px', 'color': '#1976d2', 'fontWeight': 'bold'}),
                # Insert forecast table with loading spinner
                dcc.Loading(
                    id='loading-forecast-table',
                    type='default',
                    children=[
                        dash_table.DataTable(
                            id='forecast-table',
                            columns=[],  # Columns will be set by callback
                            data=[],
                            active_cell=None,
                            page_size=25,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'center'},
                            style_header={'fontWeight': 'bold'},
                            style_data_conditional=[
                                # Color codes for key columns (background only)
                                {'if': {'column_id': 'lowP'}, 'backgroundColor': '#e0f7fa'},
                                {'if': {'column_id': 'highP'}, 'backgroundColor': '#ffe0b2'},
                                {'if': {'column_id': 'rel_spread'}, 'backgroundColor': '#ffecb3'},
                                {'if': {'column_id': 'forecast_price'}, 'backgroundColor': '#fff9c4'},
                                {'if': {'column_id': 'name'}, 'backgroundColor': '#f7f7f7'},
                                {'if': {'column_id': 'max_qty'}, 'backgroundColor': '#dcedc8'},
                                {'if': {'column_id': 'trend_direction'}, 'backgroundColor': '#f0f4c3'},
                                {'if': {'column_id': 'status'}, 'backgroundColor': '#ffcdd2'},
                                {'if': {'column_id': 'add_to_watchlist'}, 'backgroundColor': '#b2ebf2', 'fontWeight': 'bold'},
                                # Color code for risk_level
                                {'if': {'column_id': 'risk_level', 'filter_query': '{risk_level} = "Low Risk"'}, 'backgroundColor': '#c8e6c9', 'fontWeight': 'bold'},
                                {'if': {'column_id': 'risk_level', 'filter_query': '{risk_level} = "Medium Risk"'}, 'backgroundColor': '#fff9c4', 'fontWeight': 'bold'},
                                {'if': {'column_id': 'risk_level', 'filter_query': '{risk_level} = "High Risk"'}, 'backgroundColor': '#ffcdd2', 'fontWeight': 'bold'},
                                {'if': {'column_id': 'risk_level', 'filter_query': '{risk_level} = "Unknown Risk"'}, 'backgroundColor': '#e0e0e0'},
                                # Color code item name by status
                                {'if': {'column_id': 'name', 'filter_query': '{status} = "Abnormally high!"'}, 'backgroundColor': "#d7515e", 'fontWeight': 'bold'},
                                {'if': {'column_id': 'name', 'filter_query': '{status} = "Low Profit"'}, 'backgroundColor': '#fffde7', 'fontWeight': 'bold'},
                                {'if': {'column_id': 'name', 'filter_query': '{status} = "Low Volume"'}, 'backgroundColor': '#e1bee7', 'fontWeight': 'bold'},
                                {'if': {'column_id': 'name', 'filter_query': '{status} = "Low ROI"'}, 'backgroundColor': "#95edf8", 'fontWeight': 'bold'},
                                {'if': {'column_id': 'name', 'filter_query': '{status} = "Normal"'}, 'backgroundColor': "#85d285"},
                                # Color code for max_qty (high = green, low = red)
                                {'if': {'column_id': 'max_qty', 'filter_query': '{max_qty} >= 10000'}, 'backgroundColor': '#c8e6c9'},
                                {'if': {'column_id': 'max_qty', 'filter_query': '{max_qty} < 1000'}, 'backgroundColor': '#ffcdd2'},
                                # Color code for trend_direction
                                {'if': {'column_id': 'trend_direction', 'filter_query': '{trend_direction} = "UP"'}, 'backgroundColor': "#7ac07c"},
                                {'if': {'column_id': 'trend_direction', 'filter_query': '{trend_direction} = "DOWN"'}, 'backgroundColor': "#e9505f"},
                                {'if': {'column_id': 'trend_direction', 'filter_query': '{trend_direction} = "FLAT"'}, 'backgroundColor': "#ded475"},
                                # Color code for status
                                {'if': {'column_id': 'status', 'filter_query': '{status} = "Abnormally high!"'}, 'backgroundColor':"#d7515e"},
                                {'if': {'column_id': 'status', 'filter_query': '{status} = "Low Profit"'}, 'backgroundColor':'#fffde7'},
                                {'if': {'column_id': 'status', 'filter_query': '{status} = "Low Volume"'}, 'backgroundColor':'#e1bee7'},
                                {'if': {'column_id': 'status', 'filter_query': '{status} = "Normal"'}, 'backgroundColor': "#95edf8"},
                                {'if': {'column_id': 'status', 'filter_query': '{status} = "Normal"'}, 'backgroundColor': "#85d285"},
                                # Color code for add_to_watchlist
                                {'if': {'column_id': 'add_to_watchlist'}, 'backgroundColor': '#b2ebf2', 'fontWeight': 'bold'},
                                # Color code for rel_spread (high = orange, low = light)
                                {'if': {'column_id': 'rel_spread', 'filter_query': '{rel_spread} >= 10'}, 'backgroundColor': "#f16562"},
                                {'if': {'column_id': 'rel_spread', 'filter_query': '{rel_spread} >= 5 && {rel_spread} < 10'}, 'backgroundColor': "#82b2ff"},
                                {'if': {'column_id': 'rel_spread', 'filter_query': '{rel_spread} < 5'}, 'backgroundColor': "#7fe07c"},
                                # # Color code for forecast_price (high = red, mid = orange, low = yellow)
                                # {'if': {'column_id': 'forecast_price', 'filter_query': '{forecast_price} >= 10000000'}, 'backgroundColor': '#ff5252'},
                                # {'if': {'column_id': 'forecast_price', 'filter_query': '{forecast_price} >= 1000000 && {forecast_price} < 10000000'}, 'backgroundColor': '#ffb74d'},
                                # {'if': {'column_id': 'forecast_price', 'filter_query': '{forecast_price} < 1000000'}, 'backgroundColor': '#fff9c4'},
                                # Alternating row hues for columns not explicitly color coded
                                {'if': {'row_index': 'odd', 'column_id': 'id'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'id'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'lowTime'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'lowTime'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'highTime'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'highTime'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'profit'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'profit'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'roi'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'roi'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'avg_daily_volume'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'avg_daily_volume'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'buy_limit'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'buy_limit'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'potential_profit'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'potential_profit'}, 'backgroundColor': 'white'},
                                {'if': {'row_index': 'odd', 'column_id': 'volume_potential'}, 'backgroundColor': '#f9f9f9'},
                                {'if': {'row_index': 'even', 'column_id': 'volume_potential'}, 'backgroundColor': 'white'},
                            ],
                            row_selectable='single',
                            selected_rows=[0],  # Select first row by default
                        )
                    ]
                ),
                html.Div(id='timeseries-container', style={'marginTop': '30px'}),
            ]),
        ], className='tab-page')
    elif tab == 'watchlist':
        watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
        try:
            with open(watchlist_path, 'r', encoding='utf-8') as f:
                watchlist = json.load(f)
        except Exception:
            watchlist = []
        latest_prices = get_latest_prices()

        for item in watchlist:
            item_id = str(item.get('item_id'))
            latest_price = latest_prices.get(item_id, {})
            item['lowP'] = latest_price.get('low', 0)
            item['highP'] = latest_price.get('high', 0)
            item['lowTime'] = latest_price.get('lowTime', 'N/A')
            item['highTime'] = latest_price.get('highTime', 'N/A')


        # Add Potential Profit column and make Quantity editable
        return html.Div([
            html.H2('Watchlist'),
            html.Div([
                dcc.Input(id='add-watchlist-item-id', type='number', placeholder='Item ID', style={'width': '120px', 'marginRight': '8px'}),
                dcc.Input(id='add-watchlist-item-name', type='text', placeholder='Item Name', style={'width': '180px', 'marginRight': '8px'}),
                dcc.Input(id='add-watchlist-entry-price', type='number', placeholder='Entry Price', style={'width': '120px', 'marginRight': '8px'}),
                dcc.Input(id='add-watchlist-quantity', type='number', placeholder='Quantity', style={'width': '100px', 'marginRight': '8px'}),
                html.Button('Add to Watchlist', id='add-watchlist-btn', n_clicks=0, style={'height': '38px'}),
                html.Div(id='add-watchlist-status', style={'marginLeft': '12px', 'color': '#1976d2', 'fontWeight': 'bold'}),
                html.Button('Refresh', id='refresh-watchlist-btn', n_clicks=0, style={'height': '38px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '18px'}),
            html.Button('Remove selected from watchlist', id='remove-selected-from-watchlist-btn', n_clicks=0, style={'marginBottom': '10px', 'marginRight': '10px'}),
            html.Div(id='remove-selected-status', style={'marginLeft': '12px', 'color': '#d32f2f', 'fontWeight': 'bold'}),
            dash_table.DataTable(
                id='watchlist-table',
                columns=[
                    {'name': 'Item ID', 'id': 'item_id', 'type': 'text'},
                    {'name': 'Item Name', 'id': 'item_name', 'type': 'text'},
                    {'name': 'Entry Price', 'id': 'entry_price', 'type': 'numeric', 'editable': True},
                    {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric', 'editable': True},
                    {'name': 'Potential Profit', 'id': 'potential_profit', 'type': 'numeric'},
                    {'name': 'Added Time', 'id': 'added_time', 'type': 'text'},
                    {'name': 'Low Price', 'id': 'lowP', 'type': 'numeric'},
                    {'name': 'Low Time', 'id': 'lowTime', 'type': 'text'},
                    {'name': 'High Price', 'id': 'highP', 'type': 'numeric'},
                    {'name': 'High Time', 'id': 'highTime', 'type': 'text'},
                ],
                data=watchlist,
                page_size=25,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                    {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                ],
                row_selectable='single',
                selected_rows=[],
                editable=True,
            )
        ], className='tab-page')


    # --- Combined callback for removing selected and persisting entry price edits ---
    @app.callback(
        # [Output('remove-selected-status', 'children'), Output('watchlist-table', 'data')],
        [Input('remove-selected-from-watchlist-btn', 'n_clicks'), Input('watchlist-table', 'data_timestamp')],
        [State('watchlist-table', 'selected_rows'), State('watchlist-table', 'data')],
        prevent_initial_call=True
    )
    def handle_watchlist_table(remove_n_clicks, data_timestamp, selected_rows, rows):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
        status_msg = ''
        # Remove selected row
        if trigger == 'remove-selected-from-watchlist-btn':
            if not rows or not selected_rows:
                status_msg = 'No row selected.'
                return status_msg, rows
            idx = selected_rows[0]
            if idx < 0 or idx >= len(rows):
                status_msg = 'Invalid row selected.'
                return status_msg, rows
            removed_item = rows.pop(idx)
            # Save updated watchlist
            try:
                with open(watchlist_path, 'w', encoding='utf-8') as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
                status_msg = f"Removed item: {removed_item.get('item_name', removed_item.get('item_id', ''))}"
            except Exception as e:
                status_msg = f"[ERROR] Failed to remove item: {e}"
            return status_msg, rows
        # Persist entry price edits
        elif trigger == 'watchlist-table':
            if not rows:
                raise PreventUpdate
            try:
                with open(watchlist_path, 'r', encoding='utf-8') as f:
                    old_watchlist = json.load(f)
            except Exception:
                old_watchlist = []
            old_watchlist_map = {str(item.get('item_id')): item for item in old_watchlist}
            for row in rows:
                item_id = str(row.get('item_id'))
                if item_id in old_watchlist_map:
                    old_watchlist_map[item_id]['entry_price'] = row.get('entry_price', old_watchlist_map[item_id].get('entry_price'))
            try:
                with open(watchlist_path, 'w', encoding='utf-8') as f:
                    json.dump(list(old_watchlist_map.values()), f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[ERROR] Failed to update entry price in watchlist: {e}")
            return dash.no_update, rows
        else:
            raise PreventUpdate
    # Add selected to the watchlist button and store
    html.Button('Add selected to the watchlist', id='add-selected-to-watchlist-btn', n_clicks=0, style={'marginBottom': '10px', 'marginRight': '10px'}),
    dcc.Store(id='selected-row-store'),
    html.Div(id='add-selected-status', style={'marginLeft': '12px', 'color': '#1976d2', 'fontWeight': 'bold'}),




# --- Combined callback for removing selected, persisting entry price edits, and refreshing table ---

@app.callback(
    [Output('remove-selected-status', 'children'), Output('watchlist-table', 'data')],
    [Input('remove-selected-from-watchlist-btn', 'n_clicks'), Input('watchlist-table', 'data_timestamp'), Input('refresh-watchlist-btn', 'n_clicks')],
    [State('watchlist-table', 'selected_rows'), State('watchlist-table', 'data')],
    prevent_initial_call=True
)
def handle_watchlist_table(remove_n_clicks, data_timestamp, refresh_n_clicks, selected_rows, rows):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
    status_msg = ''

    # Helper to calculate potential profit
    def calc_potential_profit(row):
        try:
            highP = float(row.get('highP', 0))
            entry_price = float(row.get('entry_price', 0))
            quantity = int(row.get('quantity', 1))
            tax = highP * 0.02
            return round((highP - tax - entry_price) * quantity, 2)
        except Exception:
            return ''

    # Remove selected row
    if trigger == 'remove-selected-from-watchlist-btn':
        if not rows or not selected_rows:
            return 'No row selected.', rows
        idx = selected_rows[0]
        if idx < 0 or idx >= len(rows):
            return 'Invalid row selected.', rows
        try:
            removed_item = rows.pop(idx)
        except Exception:
            return 'Item not found in watchlist.', rows
        # Save updated watchlist
        try:
            with open(watchlist_path, 'w', encoding='utf-8') as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            status_msg = f"Removed item: {removed_item.get('item_name', removed_item.get('item_id', ''))}"
        except Exception as e:
            status_msg = f"[ERROR] Failed to remove item: {e}"
        # Update potential profit for all rows (do not update prices)
        for row in rows:
            row['potential_profit'] = calc_potential_profit(row)
        return status_msg, rows
    # Persist entry price or quantity edits
    elif trigger == 'watchlist-table':
        if not rows:
            return '', []
        try:
            with open(watchlist_path, 'r', encoding='utf-8') as f:
                old_watchlist = json.load(f)
        except Exception:
            old_watchlist = []
        old_watchlist_map = {str(item.get('item_id')): item for item in old_watchlist}
        for row in rows:
            item_id = str(row.get('item_id'))
            if item_id in old_watchlist_map:
                old_watchlist_map[item_id]['entry_price'] = row.get('entry_price', old_watchlist_map[item_id].get('entry_price'))
                old_watchlist_map[item_id]['quantity'] = row.get('quantity', old_watchlist_map[item_id].get('quantity', 1))
        try:
            with open(watchlist_path, 'w', encoding='utf-8') as f:
                json.dump(list(old_watchlist_map.values()), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to update entry price/quantity in watchlist: {e}")
        # Update potential profit for all rows (do not update prices)
        for row in rows:
            row['potential_profit'] = calc_potential_profit(row)
        return '', rows
    # Refresh table and prices (fetch latest prices only here)
    elif trigger == 'refresh-watchlist-btn':
        try:
            with open(watchlist_path, 'r', encoding='utf-8') as f:
                watchlist = json.load(f)
        except Exception:
            watchlist = []
        latest_prices = get_latest_prices()
        for item in watchlist:
            item_id = str(item.get('item_id'))
            latest_price = latest_prices.get(item_id, {})
            item['lowP'] = latest_price.get('low', 0)
            item['highP'] = latest_price.get('high', 0)
            item['lowTime'] = latest_price.get('lowTime', 'N/A')
            item['highTime'] = latest_price.get('highTime', 'N/A')
            item['potential_profit'] = calc_potential_profit(item)
        return '', watchlist
    else:
        raise PreventUpdate

@app.callback(
    [Output('forecast-table', 'data'), Output('forecast-table', 'columns'), Output('update-times', 'children'), Output('forecasted-prices-store', 'data')],
    [Input('apply-filters', 'n_clicks'), Input('item-name-search', 'value'), Input('filter-by-watchlist', 'value')],
    [State('item-name-search', 'value'),
     State('forecast-strategy', 'value'),
     State('min-profit', 'value'),
     State('min-potential-profit', 'value'),
     State('min-volume', 'value'),
     State('min-roi', 'value'),
     State('forecast-hours', 'value'),
     State('volume-power', 'value'),
    State('max-qty-factor', 'value'),
     State('top-n', 'value'),
     State('sort-attribute', 'value'),
     State('trend-filter', 'value'),
     State('forecast-sell-time', 'value'),
     State('min-price', 'value'),
     State('max-price', 'value'),
    State('min-avg-daily-volume', 'value'),
    State('max-avg-trade-time', 'value'),
    State('forecast-table', 'data'),
    State('template-dropdown', 'value'),
    State('status-filter', 'value'),
    State('risk-filter', 'value'),
    State('buy-price-type', 'value'),
    State('forecast-price-type', 'value'),]
)
def update_table_and_times(item_name_search_input, n_clicks, filter_by_watchlist_input, item_name_search, forecast_strategy, min_profit, min_potential_profit, min_volume, min_roi, forecast_hours, volume_power, max_qty_factor, top_n, sort_attribute, trend_filter, forecast_sell_time, min_price, max_price, min_avg_daily_volume, max_avg_trade_time, table_data, template_selected, status_filter, risk_filter, buy_price_type, forecast_price_type):
    if forecast_sell_time is None:
        forecast_sell_time = 0
    
    print('[DEBUG] update_table_and_times callback called')
    t_debug_start = time.time()
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    print(f'[DEBUG] triggered: {triggered}')
    print(f'[DEBUG] Start of callback, time: {t_debug_start}')
    # --- Handle watchlist cell click ---
    print('[DEBUG] Before active_cell branch, time:', time.time() - t_debug_start)
    if triggered == 'forecast-table.active_cell':
        print('[DEBUG] forecast-table.active_cell triggered')
        active_cell = ctx.inputs.get('forecast-table.active_cell')
        print(f'[DEBUG] active_cell: {active_cell}')
        print(f'[DEBUG] table_data: {table_data}')
        if not active_cell or not table_data:
            print('[DEBUG] No active_cell or table_data')
            raise PreventUpdate
        col = active_cell.get('column_id')
        row = active_cell.get('row')
        print(f'[DEBUG] col: {col}, row: {row}')
        # Only handle if user clicked the add/remove cell
        if col != 'add_to_watchlist' or row is None or row < 0 or row >= len(table_data):
            print('[DEBUG] Not add_to_watchlist column or invalid row')
            raise PreventUpdate
        cell_value = table_data[row].get('add_to_watchlist')
        print(f'[DEBUG] cell_value: {cell_value}')
        if cell_value not in ['Add', 'Remove']:
            print('[DEBUG] cell_value not Add/Remove')
            raise PreventUpdate
        item = table_data[row]
        item_id = item.get('id')
        print(f'[DEBUG] item_id: {item_id}')
        if not item_id:
            print('[DEBUG] No item_id')
            raise PreventUpdate
        # Load current watchlist
        watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
        try:
            with open(watchlist_path, 'r', encoding='utf-8') as f:
                watchlist = json.load(f)
            print(f'[DEBUG] Loaded watchlist: {watchlist}')
        except Exception as e:
            print(f'[DEBUG] Failed to load watchlist: {e}')
            watchlist = []
        watchlist_ids = set(str(w.get('item_id')) for w in watchlist)
        print(f'[DEBUG] watchlist_ids: {watchlist_ids}')
        # Add or remove
        if str(item_id) in watchlist_ids:
            print('[DEBUG] Removing from watchlist')
            watchlist = [w for w in watchlist if str(w.get('item_id')) != str(item_id)]
            table_data[row]['add_to_watchlist'] = 'Add'
        else:
            print('[DEBUG] Adding to watchlist')
            new_entry = {
                'item_id': item_id,
                'item_name': item.get('name', ''),
                'entry_price': float(item.get('lowP', 0)),
                'quantity': 1,
                'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            watchlist.append(new_entry)
            table_data[row]['add_to_watchlist'] = 'Remove'
        # Save updated watchlist
        try:
            with open(watchlist_path, 'w', encoding='utf-8') as f:
                json.dump(watchlist, f, ensure_ascii=False, indent=2)
            print('[DEBUG] Saved updated watchlist')
        except Exception as e:
            print(f"[ERROR] Failed to update watchlist: {e}")
        from dash import no_update
        print('[DEBUG] Returning updated table_data')
        return table_data, no_update, no_update, no_update
    # Ensure filter_by_watchlist_input is always a list
    print('[DEBUG] Before filter_by_watchlist_input normalization, time:', time.time() - t_debug_start)
    if filter_by_watchlist_input is None:
        filter_by_watchlist_input = []
    print(f'[DEBUG] filter_by_watchlist_input (normalized): {filter_by_watchlist_input}')
    print('[DEBUG] After filter_by_watchlist_input normalization, time:', time.time() - t_debug_start)
    # import time module already at top, avoid shadowing
    t0 = time.time()
    print(f'[DEBUG] update_table_and_times called! n_clicks={n_clicks}')
    print(f'[DEBUG] filter params: item_name_search={item_name_search}, forecast_strategy={forecast_strategy}, min_profit={min_profit}, min_potential_profit={min_potential_profit}, min_volume={min_volume}, min_roi={min_roi}, forecast_hours={forecast_hours}, top_n={top_n}, sort_attribute={sort_attribute}, trend_filter={trend_filter}, forecast_sell_time={forecast_sell_time}, min_price={min_price}, max_price={max_price}, min_avg_daily_volume={min_avg_daily_volume}')
    print('[DEBUG] After filter param print, time:', time.time() - t_debug_start)
    t1 = time.time()
    print(f'[DEBUG] Time after param print: {t1-t0:.3f}s')
    forecast_start = time.time()
    print('[DEBUG] Before main callback logic, time:', time.time() - t_debug_start)


    # If item-name-search or watchlist filter is triggered, apply those filters
    print('[DEBUG] Before item-name-search/watchlist branch, time:', time.time() - t_debug_start)
    if 'item-name-search' in triggered or 'filter-by-watchlist' in triggered or (item_name_search and isinstance(item_name_search, str) and item_name_search.strip()) or (filter_by_watchlist_input and 'watchlist' in filter_by_watchlist_input):
        
        search_term = (item_name_search or '').strip().lower()
        
        # if search_term == '' and not watchlist_filter_active:
        #     print('[DEBUG] No item-name-search or watchlist filter active, skipping forecast recalculation.')
        #     from dash import no_update
        #     return table_data, no_update, no_update, no_update
        
        t2 = time.time()
        print(f'[DEBUG] Entered item-name-search/watchlist branch: {t2-t1:.3f}s')
        print('[DEBUG] Before 5m data check/fetch, time:', time.time() - t_debug_start)
        # Only fetch new 5m data if latest chunk is missing
        import os, json, datetime
        FIVE_M_FILE = os.path.join(APP_DIR, 'Data', '5m_data_cache.json')
        latest_ts = None
        if os.path.exists(FIVE_M_FILE):
            with open(FIVE_M_FILE, 'r', encoding='utf-8') as f:
                try:
                    cache = json.load(f)
                    chunks = cache.get('chunks', [])
                    if chunks:
                        latest_ts = chunks[-1]['timestamp']
                except Exception:
                    pass
        now = datetime.datetime.now(datetime.timezone.utc)
        expected_latest_ts = ((now.replace(second=0, microsecond=0) - datetime.timedelta(minutes=now.minute % 5)).strftime('%Y-%m-%dT%H:%M:%S'))
        print('[DEBUG] Checking/fetching 5m data...')
        print('[DEBUG] Before analyze_forecast_gui call, time:', time.time() - t_debug_start)
        # Only fetch if latest chunk is older than 10 minutes
        fetch_needed = True
        # if latest_ts is None:
        #     fetch_needed = True
        # else:
        #     try:
        #         latest_dt = datetime.datetime.strptime(latest_ts, '%Y-%m-%dT%H:%M:%S')
        #         age_minutes = (now - latest_dt).total_seconds() / 60.0
        #         if age_minutes > 10: 
        #             fetch_needed = True
        #     except Exception as e:
        #         print(f"[DEBUG] Error parsing latest_ts: {e}")
        #         fetch_needed = True
        if fetch_needed:
            try:
                from fetch_5m_data import fetch_and_save_5m_data
                fetch_and_save_5m_data()
                print('[DEBUG] 5m data check/fetch complete.')
            except Exception as e:
                print(f"Error fetching 5m data: {e}")
        t2b = time.time()
        print(f'[DEBUG] 5m data update duration: {t2b-t2:.3f}s')
        print('[DEBUG] Preparing historical data for forecast...')
        hist_data_source = 'Price_History.json'
        hist = None
        if forecast_sell_time is not None and forecast_sell_time <= 180:
            # Use last 3h of 5m chunks from 5m_data_cache.json
            import json
            from datetime import datetime, timezone
            five_m_file = os.path.join(APP_DIR, 'Data', '5m_data_cache.json')
            item_hist_map = {}
            try:
                with open(five_m_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                chunks = cache.get('chunks', [])
                if len(chunks) >= 36:
                    last_chunks = chunks[-36:]
                else:
                    last_chunks = chunks
                for chunk in last_chunks:
                    ts = chunk.get('timestamp')
                    data_chunk = chunk.get('data', {}).get('data', {})
                    for item_id, entry in data_chunk.items():
                        if item_id not in item_hist_map:
                            item_hist_map[item_id] = []
                        entry_copy = entry.copy()
                        entry_copy['timestamp'] = ts
                        item_hist_map[item_id].append(entry_copy)
                hist = item_hist_map
            except Exception as e:
                print(f'[DEBUG] Error loading 5m_data_cache.json: {e}')
        print('[DEBUG] Calling analyze_forecast_gui...')
        print('[DEBUG] Before analyze_forecast_gui actual call, time:', time.time() - t_debug_start)
        print('[DEBUG] About to call analyze_forecast_gui, time:', time.time() - t_debug_start)

        import os, json
        
        with open(os.path.join(APP_DIR, 'Data', 'mapping_cache.json'), 'r', encoding='utf-8') as f:
            mapping_cache = json.load(f)
        
        # --- Watchlist filter logic ---
        watchlist_ids = set()
        watchlist_filter_active = filter_by_watchlist_input and 'watchlist' in filter_by_watchlist_input
        print(f'[DEBUG] filter_by_watchlist_input: {filter_by_watchlist_input}')
        if watchlist_filter_active:
            watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
            try:
                with open(watchlist_path, 'r', encoding='utf-8') as f:
                    watchlist = json.load(f)
                print(f'[DEBUG] Loaded watchlist: {watchlist}')
                watchlist_ids = set(str(w['item_id']) for w in watchlist if 'item_id' in w)
                print(f'[DEBUG] Extracted watchlist_ids: {watchlist_ids}')
            except Exception as e:
                print(f'[DEBUG] Error loading watchlist: {e}')
                watchlist_ids = set()
            print(f'[DEBUG] Watchlist filter active. Watchlist IDs: {watchlist_ids}')
        print(f'[DEBUG] Filtering by name ("{search_term}") and/or watchlist ({watchlist_filter_active})...')
        if watchlist_filter_active:
            if not watchlist_ids:
                print('[DEBUG] No watchlist IDs found, filtered will be empty.')
                name_filter = []
            else:
                # a = mapping_cache
                # b = mapping_cache['data']

                # for c_ in mapping_cache['data']:
                #     c = mapping_cache['data'][c_]
                #     d = mapping_cache['data'][c_]['id']
                #     e = str(mapping_cache['data'][c_]['id'])
                #     f = watchlist_ids
                #     g = mapping_cache['data'][c_]['name']




                name_filter = [mapping_cache['data'][r]['name'] for r in mapping_cache['data'] if str(mapping_cache['data'][r]['id']) in watchlist_ids]
                # print(f'[DEBUG] Filtered items by watchlist: {[str(r["id"]) for r in filtered]}')
        else:
            name_filter = [mapping_cache['data'][r]['name'] for r in mapping_cache['data'] if mapping_cache['data'][r]['name'].lower().find(search_term) != -1]


        # Gather status filter from GUI dropdown ONLY (not from template)
        status_filter_ctx = []
        status_filter_mode = 'include'
        status_exclude_ctx = []
        
        # Only use status filters if user explicitly selected them in the UI
        ctx_local = dash.callback_context
        if ctx_local and ctx_local.inputs:
            gui_status_filter = ctx_local.inputs.get('status-filter.value', []) or []
            if gui_status_filter and len(gui_status_filter) > 0:
                status_filter_ctx = gui_status_filter
                status_filter_mode = 'exclude' if 'exclude' in ctx_local.inputs.get('status-filter-exclude.value', []) else 'include'
                print(f'[DEBUG-GUI] Using GUI status filter: {status_filter_ctx}, mode: {status_filter_mode}')
            else:
                print(f'[DEBUG-GUI] No status filter selected in UI')

        print(f'[DEBUG-GUI] Final status filter to send: {status_filter_ctx}, exclude: {status_exclude_ctx}, mode: {status_filter_mode}')
        
        try:
            print('[DEBUG] Entering analyze_forecast_gui...')
            all_items, forecasted_prices, latest_prices = analyze_forecast_gui({
                'MIN_FORECAST_PROFIT': 0,
                'MIN_POTENTIAL_PROFIT': 0,
                'MIN_VOLUME_POTENTIAL': 0,
                'MIN_FORECAST_ROI': 0,
                'FORECAST_HOURS': forecast_hours,
                'STATUS_FILTER': status_filter_ctx,
                'STATUS_FILTER_MODE': status_filter_mode,
                'STATUS_EXCLUDE': status_exclude_ctx,
            }, forecast_sell_time, forecast_strategy, name_filter=name_filter, volume_power = volume_power, max_qty_factor=max_qty_factor, forecast_hours=forecast_hours, buy_price_type=buy_price_type, forecast_price_type=forecast_price_type)
            print('[DEBUG] analyze_forecast_gui finished, time:', time.time() - t_debug_start)
        except Exception as e:
            print(f'[DEBUG] Exception in analyze_forecast_gui: {e}')
            all_items, forecasted_prices = [], []
        print('[DEBUG] After analyze_forecast_gui execution, time:', time.time() - t_debug_start)
        forecast_end = time.time()
        forecast_duration = forecast_end - forecast_start
        print('[DEBUG] After analyze_forecast_gui returned, time:', time.time() - t_debug_start)
        
        t4 = time.time()
        data = all_items
        print('[DEBUG] Before apply-filters branch, time:', time.time() - t_debug_start)
    elif 'apply-filters' in triggered:
        t2 = time.time()
        print(f'[DEBUG] Entered apply-filters branch: {t2-t1:.3f}s')
        print('[DEBUG] Before 5m data check/fetch (apply-filters), time:', time.time() - t_debug_start)
        # Always fetch new 5m data on update
        print('[DEBUG] Checking/fetching 5m data...')
        print('[DEBUG] Before analyze_forecast_gui call (apply-filters), time:', time.time() - t_debug_start)
        try:
            from fetch_5m_data import fetch_and_save_5m_data
            fetch_and_save_5m_data()
            print('[DEBUG] 5m data check/fetch complete.')
        except Exception as e:
            print(f"Error fetching 5m data: {e}")
        t2b = time.time()

        name_filter = []

        import os, json

        with open(os.path.join(APP_DIR, 'Data', 'mapping_cache.json'), 'r', encoding='utf-8') as f:
            mapping_cache = json.load(f)
        if item_name_search and isinstance(item_name_search, str) and item_name_search.strip():
            search_term = item_name_search.strip().lower()
            print(f"[DEBUG] Performing item name search: '{search_term}' (disregarding all filters)")
            print('[DEBUG] Filtering by name...')
            name_filter = [mapping_cache['data'][r]['name'] for r in mapping_cache['data'] if not search_term or search_term in r['name'].lower()]


        print(f'[DEBUG] 5m data update duration: {t2b-t2:.3f}s')
        print('[DEBUG] Calling analyze_forecast_gui...')
        print('[DEBUG] Before analyze_forecast_gui actual call (apply-filters), time:', time.time() - t_debug_start)
        print('[DEBUG] Before analyze_forecast_gui execution (apply-filters), time:', time.time() - t_debug_start)

        # Gather status filter from GUI dropdown parameter
        status_filter_ctx = []
        status_filter_mode = 'include'
        status_exclude_ctx = []
        
        # Use status filter from State parameter
        if status_filter and isinstance(status_filter, list) and len(status_filter) > 0:
            status_filter_ctx = status_filter
            # Check if exclude mode is enabled (you'd need to add this as a State too if needed)
            print(f'[DEBUG-GUI] (apply-filters) Using GUI status filter: {status_filter_ctx}, mode: {status_filter_mode}')
        else:
            print(f'[DEBUG-GUI] (apply-filters) No status filter selected in UI')

        print(f'[DEBUG-GUI] (apply-filters) Final status filter to send: {status_filter_ctx}, exclude: {status_exclude_ctx}, mode: {status_filter_mode}')

        # Send no status filter to backend; apply UI status filter on the GUI-rendered rows
        all_items, forecasted_prices_raw, latest_prices = analyze_forecast_gui({
            'MIN_FORECAST_PROFIT': 0,
            'MIN_POTENTIAL_PROFIT': 0,
            'MIN_VOLUME_POTENTIAL': 0,
            'MIN_FORECAST_ROI': 0,
            'FORECAST_HOURS': forecast_hours,
            'STATUS_FILTER': [],
            'STATUS_FILTER_MODE': 'include',
            'STATUS_EXCLUDE': [],
        }, forecast_sell_time, forecast_strategy, name_filter=name_filter, max_avg_trade_time=max_avg_trade_time, volume_power = volume_power, max_qty_factor=max_qty_factor, forecast_hours=forecast_hours, buy_price_type=buy_price_type, forecast_price_type=forecast_price_type)
        print('[DEBUG] After analyze_forecast_gui execution (apply-filters), time:', time.time() - t_debug_start)
        forecast_end = time.time()
        forecast_duration = forecast_end - forecast_start
        t3 = time.time()
        print('[DEBUG] After analyze_forecast_gui returned (apply-filters), time:', time.time() - t_debug_start)
        # Build forecasted_prices map by id
        forecasted_prices_map = {entry['id']: entry for entry in forecasted_prices_raw if 'id' in entry}
        # If item name search is provided, filter by name and disregard all other filters
        if item_name_search and isinstance(item_name_search, str) and item_name_search.strip():
            pass
        else:
            print('[DEBUG] Filtering by all filters...')
            t4 = time.time()
            filtered = [r for r in all_items
                        if (min_potential_profit is None or min_potential_profit == 0 or (r.get('potential_profit', 0) >= min_potential_profit))
                        and (min_volume is None or min_volume == 0 or (r.get('volume_potential', 0) >= min_volume))
                        and (min_profit is None or min_profit == 0 or (r.get('profit', 0) >= min_profit))
                        and (min_roi is None or min_roi == 0 or (r.get('roi', 0) >= min_roi/100))
                        and (not trend_filter or r.get('trend_direction', '') == trend_filter)
                        and (min_price is None or min_price == 0 or (r.get('lowP', 0) >= min_price))
                        and (max_price is None or max_price == 0 or (r.get('lowP', 0) <= max_price))
                        and (min_avg_daily_volume is None or min_avg_daily_volume == 0 or (r.get('avg_daily_volume', 0) >= min_avg_daily_volume))]
            t5 = time.time()
            print(f'[DEBUG] After filter: {t5-t4:.3f}s, filtered={len(filtered)}')
            # Sorting logic
            if isinstance(filtered, list) and filtered and sort_attribute:
                if all(isinstance(x, dict) for x in filtered):
                    try:
                        def safe_sort_key(x):
                            val = x.get(sort_attribute, 0)
                            if val is None:
                                return float('-inf')
                            try:
                                return float(val)
                            except Exception:
                                return float('-inf')
                        filtered.sort(key=safe_sort_key, reverse=True)
                        t5b = time.time()
                        print(f'[DEBUG] After sort: {t5b-t5:.3f}s, filtered={len(filtered)}')
                    except Exception as e:
                        print(f'[DEBUG] Error sorting by {sort_attribute}: {e}')
            forecasted_prices = [forecasted_prices_map[item['id']] for item in filtered if item['id'] in forecasted_prices_map]
            # Validate forecast_price structure
            for fp in forecasted_prices:
                if not isinstance(fp.get('forecast_price'), list):
                    fp['forecast_price'] = []
            # Defer top_n slicing until after UI status filtering on rendered rows
            data = filtered
            t6 = time.time()
            print(f'[DEBUG] After top_n: {t6-t5:.3f}s, data={len(data)}')


            HARD_CAP = 100
            
            if isinstance(data, list):
                # (Removed early ctx-based status filtering to avoid mismatches with GUI-rendered status.)

                # --- Min/Max Price Filter using number inputs ---
                def parse_price(val):
                    try:
                        return float(str(val).replace('.', '').replace(',', ''))
                    except Exception:
                        return None
                if min_price is not None and min_price > 0:
                    data = [row for row in data if parse_price(row.get('lowP', 0)) is not None and parse_price(row.get('lowP', 0)) >= min_price]
                if max_price is not None and max_price > 0:
                    data = [row for row in data if parse_price(row.get('lowP', 0)) is not None and parse_price(row.get('lowP', 0)) <= max_price]

                if top_n and isinstance(top_n, int) and top_n > 0:
                    data = data[:min(top_n, HARD_CAP)]
                else:
                    data = data[:HARD_CAP]
                if len(data) == HARD_CAP:
                    print(f'[DEBUG] Table row count capped at {HARD_CAP}')
                print(f'[DEBUG] Final data row count: {len(data)}')

    if not isinstance(data, list):
        # Defensive: if data is not a list, return empty table and log error
        print('[ERROR] Data is not a list in update_table_and_times, returning empty table.')
        columns = [
            {'name': 'id', 'id': 'id'},
            {'name': 'name', 'id': 'name'},
            {'name': 'lowP', 'id': 'lowP'},
            {'name': 'highP', 'id': 'highP'},
            {'name': 'lowTime', 'id': 'lowTime'},
            {'name': 'highTime', 'id': 'highTime'},
            {'name': 'rel_spread', 'id': 'rel_spread'},
            {'name': 'forecast_price', 'id': 'forecast_price'},
            {'name': 'profit', 'id': 'profit'},
            {'name': 'roi', 'id': 'roi'},
            {'name': 'avg_daily_volume', 'id': 'avg_daily_volume'},
            {'name': 'buy_limit', 'id': 'buy_limit'},
            {'name': 'max_qty', 'id': 'max_qty'},
            {'name': 'potential_profit', 'id': 'potential_profit'},
            {'name': 'volume_potential', 'id': 'volume_potential'},
            {'name': 'trend_direction', 'id': 'trend_direction'},
            {'name': 'risk_level', 'id': 'risk_level'},
            {'name': 'status', 'id': 'status'},
            {'name': 'lowVol_recent', 'id': 'lowVol_recent'},
            {'name': 'highVol_recent', 'id': 'highVol_recent'},
            {'name': 'buy_price', 'id': 'buy_price', 'hideable': False, 'hidden': True},
            {'name': 'sell_price', 'id': 'sell_price', 'hideable': False, 'hidden': True},
        ]
        update_text = 'No data available.'
        forecasted_prices_dict = {}
        return [], columns, update_text, forecasted_prices_dict
    columns = [
        {'name': 'id', 'id': 'id'},
        {'name': 'name', 'id': 'name'},
        {'name': 'lowP', 'id': 'lowP'},
        {'name': 'highP', 'id': 'highP'},
        {'name': 'lowTime', 'id': 'lowTime'},
        {'name': 'highTime', 'id': 'highTime'},
        {'name': 'rel_spread', 'id': 'rel_spread'},
        {'name': 'forecast_price', 'id': 'forecast_price'},
        {'name': 'profit', 'id': 'profit'},
        {'name': 'roi', 'id': 'roi'},
        {'name': 'avg_daily_volume', 'id': 'avg_daily_volume'},
        {'name': 'buy_limit', 'id': 'buy_limit'},
        {'name': 'max_qty', 'id': 'max_qty'},
        {'name': 'potential_profit', 'id': 'potential_profit'},
        {'name': 'volume_potential', 'id': 'volume_potential'},
        {'name': 'trend_direction', 'id': 'trend_direction'},
        {'name': 'risk_level', 'id': 'risk_level'},
        {'name': 'status', 'id': 'status'},
        {'name': 'lowVol_recent', 'id': 'lowVol_recent'},
        {'name': 'highVol_recent', 'id': 'highVol_recent'},
        {'name': 'buy_price', 'id': 'buy_price', 'hideable': False, 'hidden': True},
        {'name': 'sell_price', 'id': 'sell_price', 'hideable': False, 'hidden': True},
    ]
    HARD_CAP = 100
    # # --- Filter by max average trade time (average of high and low price times) ---
    # if isinstance(data, list) and max_avg_trade_time is not None and max_avg_trade_time > 0:
    #     def parse_minutes_ago(val):
    #         try:
    #             if isinstance(val, str) and 'min ago' in val:
    #                 return float(val.split(' ')[0])
    #             return float(val)
    #         except Exception:
    #             return None
    #     filtered_data = []
    #     for row in data:
    #         low_time = parse_minutes_ago(row.get('lowTime', '999'))
    #         high_time = parse_minutes_ago(row.get('highTime', '999'))
    #         if low_time is not None and high_time is not None:
    #             avg_time = (low_time + high_time) / 2
    #             if avg_time <= max_avg_trade_time:
    #                 filtered_data.append(row)
    #     data = filtered_data
    if isinstance(data, list):
        # --- Dynamically extract all unique status values from data for filter options ---
        all_statuses = set()
        for row in data:
            status_val = row.get('status', '')
            # Split on ";" and also on manipulation warnings
            for part in status_val.split(';'):
                part = part.strip()
                if part:
                    # Also split on '⚠' to get manipulation warnings as separate statuses
                    for subpart in part.split('⚠'):
                        subpart = subpart.strip()
                        if subpart:
                            all_statuses.add(subpart)
        all_statuses = sorted(all_statuses)

        # --- Filter by status column using the State parameter ---
        status_filter_values = status_filter if isinstance(status_filter, list) else []
        if status_filter_values:
            print(f"[DEBUG-GUI] Applying UI status filter on rendered rows: {status_filter_values}")
            def _matches_status(row_status: str, selected_terms: list) -> bool:
                s = (row_status or '').strip()
                s_lower = s.lower()
                for term in selected_terms:
                    if not term:
                        continue
                    if term == 'Normal':
                        if s == 'Normal':
                            return True
                    else:
                        if term.lower() in s_lower:
                            return True
                return False
            data = [row for row in data if _matches_status(row.get('status', ''), status_filter_values)]
        
        # --- Filter by risk level ---
        if risk_filter and risk_filter != 'all':
            print(f"[DEBUG-GUI] Applying risk filter: {risk_filter}")
            if risk_filter == 'low':
                data = [row for row in data if row.get('risk_level') == 'Low Risk']
            elif risk_filter == 'low_medium':
                data = [row for row in data if row.get('risk_level') in ['Low Risk', 'Medium Risk']]
            elif risk_filter == 'exclude_high':
                data = [row for row in data if row.get('risk_level') != 'High Risk']
        
        if top_n and isinstance(top_n, int) and top_n > 0:
            data = data[:min(top_n, HARD_CAP)]
        else:
            data = data[:HARD_CAP]
        if len(data) == HARD_CAP:
            print(f'[DEBUG] Table row count capped at {HARD_CAP}')
        print(f'[DEBUG] Final data row count: {len(data)}')

    # Provide all_statuses for use in the status filter dropdown elsewhere (e.g., as a return value or via a store)

    import math as _math
    def format_number(num, decimals=0):
        if num is None or num == '':
            return ''
        try:
            num = float(num)
        except Exception:
            return str(num)
        if decimals == 0:
            return '{:,.0f}'.format(num).replace(',', '.')
        else:
            # Use comma as decimal separator, period for thousands
            return '{:,.2f}'.format(num).replace(',', 'X').replace('.', ',').replace('X', '.')

    # --- Caching latest prices from RS Wiki API ---
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    # Add a minimum timeout between API attempts (e.g., 30 seconds)
    

    def clean_row(row):
        new_row = {}
        new_row['id'] = row.get('id', '')
        new_row['name'] = row.get('name', '')
        # Preserve buy_price and sell_price from backend for dashed lines
        new_row['buy_price'] = row.get('buy_price')
        new_row['sell_price'] = row.get('sell_price')
        # Use latest price from RS Wiki API if available


        import time
        import math

        now = time.time()

        item_id = str(row.get('id', ''))
        if item_id in latest_prices:
            # Preserve lowP from backend if it's already set (e.g., for 30min avg)
            # Only override if it matches the backend's original value
            backend_lowP = row.get('lowP', 0)
            current_api_low = latest_prices[item_id].get('low', backend_lowP)
            
            # If backend set a custom lowP (like 30min avg), keep it; otherwise use API value
            new_row['lowP'] = format_number(backend_lowP, 0)
            new_row['highP'] = format_number(latest_prices[item_id].get('high', row.get('highP')), 0)
            new_row['lowTime'] = f'{math.floor((now - latest_prices[item_id].get('lowTime', row.get('lowTime', 0)))/60)} min ago' if latest_prices[item_id].get('lowTime', 0) > 0 else 'None'
            new_row['highTime'] =  f'{math.floor((now - latest_prices[item_id].get('highTime', row.get('highTime', 0)))/60)} min ago' if latest_prices[item_id].get('highTime', 0) > 0 else 'None'
        else:
            new_row['lowP'] = format_number(row.get('lowP'), 0)
            new_row['highP'] = format_number(row.get('highP'), 0)
            new_row['lowTime'] = row.get('lowTime', 'None')
            new_row['highTime'] = row.get('highTime', 'None')
        # # Relative spread: (highP - lowP) / avg((highP + lowP)/2), as percent
        # try:
        #     lp = float(row.get('lowP', 0))
        #     hp = float(row.get('highP', 0))
        #     if lp > 0 and hp > 0:
        #         avgp = (lp + hp) / 2
        #         rel_spread = (hp - lp) / avgp * 100
        #         new_row['rel_spread'] = format_number(rel_spread, 2)
        #     else:
        #         new_row['rel_spread'] = ''
        # except Exception as e:
        #     print(f'[DEBUG] Error calculating rel_spread for item {row.get("id", "")}: {e}')
        #     new_row['rel_spread'] = ''

        new_row['rel_spread'] = '{:,.2f}'.format(row.get('rel_spread')) # format_number(row.get('rel_spread'), 2)
        new_row['forecast_price'] = format_number(row.get('forecast_price'), 0)
        new_row['profit'] = format_number(row.get('profit'), 0)
        new_row['roi'] = '{:,.1f}'.format(row.get('roi')*100) # format_number(row.get('roi')*100, 1)
        new_row['avg_daily_volume'] = format_number(row.get('avg_daily_volume'), 0)
        new_row['buy_limit'] = format_number(row.get('buy_limit'), 0)
        new_row['max_qty'] = format_number(row.get('max_qty'), 0)
        new_row['potential_profit'] = format_number(row.get('potential_profit'), 0)
        new_row['volume_potential'] = format_number(row.get('volume_potential'), 2)
        new_row['trend_direction'] = row.get('trend_direction', '')
        new_row['risk_level'] = row.get('risk_level', 'Unknown Risk')
        new_row['lowVol_recent'] = row.get('lowVol_recent', '')
        new_row['highVol_recent'] = row.get('highVol_recent', '')
        # Status logic + manipulation warning + spike detection
        status = ''
        try:
            fp = float(row.get('forecast_price', 0))
            profit = float(row.get('profit', 0))
            low_vol = float(row.get('lowVol', 0))
            high_vol = float(row.get('highVol', 0))
            roi = float(row.get('roi', 0))
            rel_spread = float(row.get('rel_spread', 0)) if row.get('rel_spread', '') else 0

            new_row['rel_spread'] = format_number(rel_spread, 2)
            manipulation = False
            warning_msgs = []
            # if abs(fp) > 1e7:
            #     status = 'Abnormally high!'
            #     warning_msgs.append('Forecast price extremely high.')
            # elif profit < 1000:
            #     status = 'Low Profit'
            # elif low_vol < 10 or high_vol < 10:
            #     status = 'Low Volume'
            # elif roi < 0.1:
            #     status = 'Low ROI'
            # else:
            #     status = 'OK'
            status = ''
            # Manipulation detection
            if rel_spread > 20:
                manipulation = True
                warning_msgs.append('Spread unusually high.')
            # if high_vol > 0 and low_vol / high_vol < 0.1:
            #     manipulation = True
            #     warning_msgs.append('Volume imbalance.')
            if 'Abnormally high!' in status:
                manipulation = True

            

            lowVol_high = False
            lowVol_low = False
            highVol_high = False
            highVol_low = False

            if 'Low-Volume low!' in status:
                lowVol_low = True
            if 'High-Volume low!' in status:
                highVol_low = True
            if 'Low-Volume high!' in status:
                lowVol_high = True
            if 'High-Volume high!' in status:
                highVol_high = True


            

            # --- Spike/Outlier Detection ---
            # Only run if timeseries data is available
            try:
                from statistics import mean, stdev
                item_id = row.get('id', None)
                if item_id is not None:
                    try:
                        ts = get_item_timeseries(item_id)
                    except Exception as e:
                        print(f'[DEBUG] Error loading timeseries for item {item_id}: {e}')
                        ts = None
                    if ts and len(ts) == 5:
                        _, avgLowPrice, avgHighPrice, lowVol, highVol = ts
                        # Defensive: ensure all are lists of numbers
                        def safe_list(lst):
                            return [x for x in lst if isinstance(x, (int, float)) and x is not None]
                        avgHighPrice = safe_list(avgHighPrice)
                        avgLowPrice = safe_list(avgLowPrice)
                        highVol = safe_list(highVol)
                        lowVol = safe_list(lowVol)
                        # Check for price spike in last point (current)
                        if len(avgHighPrice) > 10:
                            last = avgHighPrice[-1]
                            prev = avgHighPrice[:-1]
                            if prev:
                                m = mean(prev)
                                s = stdev(prev) if len(prev) > 1 else 0
                                if s > 0 and abs(last - m) > 3 * s:
                                    manipulation = True
                                    warning_msgs.append('High price spike detected.')
                        if len(avgLowPrice) > 10:
                            last = avgLowPrice[-1]
                            prev = avgLowPrice[:-1]
                            if prev:
                                m = mean(prev)
                                s = stdev(prev) if len(prev) > 1 else 0
                                if s > 0 and abs(last - m) > 3 * s:
                                    manipulation = True
                                    warning_msgs.append('Low price dump detected.')
                        if len(highVol) > 10:
                            last = highVol[-1]
                            prev = highVol[:-1]
                            if prev:
                                m = mean(prev)
                                s = stdev(prev) if len(prev) > 1 else 0
                                if s > 0 and abs(last - m) > 3 * s:
                                    manipulation = True
                                    warning_msgs.append('High-Volume spike detected.')
                        if len(lowVol) > 10:
                            last = lowVol[-1]
                            prev = lowVol[:-1]
                            if prev:
                                m = mean(prev)
                                s = stdev(prev) if len(prev) > 1 else 0
                                if s > 0 and abs(last - m) > 3 * s:
                                    manipulation = True
                                    warning_msgs.append('Low-Volume spike detected.')
                        if lowVol_high:
                            warning_msgs.append('Low-Volume high!')
                        if lowVol_low:
                            warning_msgs.append('Low-Volume low!')
                        if highVol_high:
                            warning_msgs.append('High-Volume high!')
                        if highVol_low:
                            warning_msgs.append('High-Volume low!')

            except Exception as e:
                print(f'[DEBUG] Error in spike detection for item {row.get("id", "")}: {e}')

            if manipulation:
                status += ' ⚠ Possible manipulation! ' + '; '.join(warning_msgs)

            if status == '':
                status = 'Normal'
        except Exception as e:
            print(f'[DEBUG] Error calculating status for item {row.get("id", "")}: {e}')
            status = ''
        new_row['status'] = status
        return new_row
    try:
        if data:
            data = [clean_row(row) for row in data]
            # Apply status filter AFTER clean_row so we filter on GUI-rendered status
            status_filter_values = status_filter if isinstance(status_filter, list) else []
            if status_filter_values:
                print(f"[DEBUG-GUI] Applying UI status filter on rendered rows: {status_filter_values}")
                def _matches_status(row_status: str, selected_terms: list) -> bool:
                    s = (row_status or '').strip()
                    s_lower = s.lower()
                    for term in selected_terms:
                        if not term:
                            continue
                        if term == 'Normal':
                            if s == 'Normal':
                                return True
                        else:
                            if term.lower() in s_lower:
                                return True
                    return False
                data = [row for row in data if _matches_status(row.get('status', ''), status_filter_values)]
    except Exception as e:
        print(f'[DEBUG] Error cleaning data rows or applying status filter: {e}')
    # print('[DEBUG] Final data sample:', data[:3])
    # print('[DEBUG] Final columns:', columns)
    print('[DEBUG] Before return, time:', time.time() - t_debug_start)
    # Generate update_text with data timestamps and refresh time
    try:
        one_h_dt, five_m_dt = get_update_times()
        from datetime import datetime as dt_cls, timezone, timedelta
        refresh_time = dt_cls.now().strftime('%Y-%m-%d %H:%M:%S')
        
        time_parts = []
        time_parts.append(f"Refresh: {refresh_time}")
        
        if one_h_dt:
            # Convert from UTC to local timezone
            one_h_utc = one_h_dt.replace(tzinfo=timezone.utc)
            one_h_local = one_h_utc.astimezone()
            hour_end_local = one_h_local + timedelta(hours=1)
            range_str = f"{one_h_local.strftime('%Y-%m-%d %H:%M')} - {hour_end_local.strftime('%H:%M')}"
            time_parts.append(f"1h Data: {range_str}")
        
        if five_m_dt:
            # Convert from UTC to local timezone
            utc_time = five_m_dt.replace(tzinfo=timezone.utc)
            local_time = utc_time.astimezone()
            time_parts.append(f"5m Data: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        update_text = " | ".join(time_parts)
    except Exception as e:
        print(f"[DEBUG] Error generating update_text: {e}")
        update_text = f"Refresh: {dt_cls.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print('[DEBUG] Final update_text:', update_text)
    # Convert forecasted_prices to dict for charting
    forecasted_prices_dict = {}
    if isinstance(forecasted_prices, list):
        for entry in forecasted_prices:
            if isinstance(entry, dict) and 'id' in entry and 'forecast_price' in entry:
                forecasted_prices_dict[entry['id']] = entry['forecast_price']
    elif isinstance(forecasted_prices, dict):
        forecasted_prices_dict = forecasted_prices
    else:
        forecasted_prices_dict = {}
    # Return only the four expected outputs - let DataTable manage selected_rows
    return data, columns, update_text, forecasted_prices_dict

@app.callback(
    Output('timeseries-container', 'children'),
    [Input('forecast-table', 'selected_rows'), Input('forecast-table', 'active_cell'), Input('forecast-table', 'data'), Input('forecasted-prices-store', 'data'), Input('forecast-sell-time', 'value'), Input('dark-mode-store', 'data')],
    prevent_initial_call=False
)
def show_timeseries(selected_rows, active_cell, table_data, forecasted_prices, forecast_sell_time=0, dark_mode_data=None):
    dark_mode = dark_mode_data.get('dark_mode', False) if dark_mode_data else False
    t_callback_start = time.perf_counter()
    t_fig1h_end = t_callback_start  # Initialize to avoid UnboundLocalError
    
    # Write to file to test if callback fires
    try:
        with open('C:\\Mehmet\\Python\\OSRS\\GEWatch2GUI\\callback_test.txt', 'a') as f:
            f.write(f"CALLBACK FIRED! selected_rows={selected_rows}\n")
    except:
        pass
    
    print("=" * 80, flush=True)
    print("[DEBUG-CALLBACK] show_timeseries CALLED!", flush=True)
    print(f"[DEBUG-CALLBACK] selected_rows = {selected_rows}", flush=True)
    print(f"[DEBUG-CALLBACK] active_cell = {active_cell}", flush=True)
    print("=" * 80, flush=True)
    
    if forecast_sell_time is None:
        forecast_sell_time = 0

    # If no row selected, use the first row as default
    row = None
    if isinstance(active_cell, dict) and active_cell.get('row') is not None:
        row = active_cell.get('row')
    elif selected_rows and len(selected_rows) > 0:
        row = selected_rows[0]
    elif table_data and len(table_data) > 0:
        row = 0  # Default to first row
    
    if row is not None and table_data and isinstance(table_data, list) and 0 <= row < len(table_data):
        item_id = table_data[row]['id']
        print(f"[DEBUG] Row {row}: table_data[row].keys() = {list(table_data[row].keys())}")
        print(f"[DEBUG] Row {row}: lowP = {repr(table_data[row].get('lowP'))}, type = {type(table_data[row].get('lowP'))}")
        print(f"[DEBUG] Row {row}: highP = {repr(table_data[row].get('highP'))}, type = {type(table_data[row].get('highP'))}")

        def parse_table_price(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            try:
                cleaned = str(value).strip()
                if not cleaned or cleaned.lower() in ('none', 'nan'):
                    return None
                cleaned = cleaned.replace(' ', '').replace('.', '').replace(',', '')
                result = float(cleaned) if cleaned else None
                print(f"[DEBUG] Parsed '{value}' -> '{cleaned}' -> {result}")
                return result
            except Exception as e:
                print(f"[DEBUG] Failed to parse '{value}': {e}")
                return None

        line_low_price = parse_table_price(table_data[row].get('lowP'))
        line_high_price = parse_table_price(table_data[row].get('highP'))
        print(f"[DEBUG] After parsing: line_low_price={line_low_price}, line_high_price={line_high_price}")

        if line_low_price is None:
            line_low_price = parse_table_price(table_data[row].get('buy_price'))
        if line_high_price is None:
            line_high_price = parse_table_price(table_data[row].get('sell_price'))
        if line_high_price is None:
            line_high_price = parse_table_price(table_data[row].get('forecast_price'))

        print(f"[DEBUG-GUI] Item {item_id}: dashed line lowP={line_low_price}, highP={line_high_price}")
        t_parse_end = time.perf_counter()
        print(f"[PERF] Price parsing: {(t_parse_end - t_callback_start):.3f}s", flush=True)
        
        # --- 1h chart (existing) ---
        t_data_load_start = time.perf_counter()
        timestamps, avgLowPrice, avgHighPrice, lowVol, highVol = get_item_timeseries(item_id)
        if not timestamps:
            print(f"[PERF] Data loaded (1h): {(time.perf_counter() - t_data_load_start):.3f}s", flush=True)
            t_callback_end = time.perf_counter()
            elapsed = t_callback_end - t_callback_start
            print(f"[PERF] show_timeseries callback (no data) in {elapsed:.3f}s", flush=True)
            return html.Div('No time series data available for this item.')
        # Convert 1h timestamps from UTC to local timezone for chart display
        timestamps_shifted = []
        for ts in timestamps:
            try:
                local_ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                local_ts = ts
            timestamps_shifted.append(local_ts)
        
        t_fig1h_start = time.perf_counter()
        print(f"[PERF] Data loaded (1h): {(t_fig1h_start - t_data_load_start):.3f}s", flush=True)

        def build_dense_hover_x(time_values, target_points=150):
            if not time_values or len(time_values) < 2:
                return time_values
            try:
                start_dt = pandas_mod.to_datetime(time_values[0])
                end_dt = pandas_mod.to_datetime(time_values[-1])
                if start_dt >= end_dt:
                    return time_values
                point_count = max(target_points, len(time_values))
                dense_range = pandas_mod.date_range(start=start_dt, end=end_dt, periods=point_count)
                return dense_range.strftime('%Y-%m-%d %H:%M:%S').tolist()
            except Exception:
                return time_values

        low_price_color = '#6EA8FF' if dark_mode else '#1f77b4'
        high_price_color = '#FF8A65' if dark_mode else '#ff7f0e'
        forecast_color = '#FF5C8A' if dark_mode else 'red'
        low_vol_color = 'rgba(110,168,255,0.45)' if dark_mode else 'rgba(0,100,200,0.3)'
        high_vol_color = 'rgba(255,138,101,0.45)' if dark_mode else 'rgba(200,100,0,0.3)'

        fig_1h = go.Figure()
        fig_1h.add_trace(go.Scatter(x=timestamps_shifted, y=avgLowPrice, mode='lines+markers', name='Low Price',
                     line=dict(color=low_price_color, width=3),
                     marker=dict(color=low_price_color, size=5),
                                 yaxis='y1',
                                 hovertemplate='Time: %{x}<br>Low Price: %{y}<extra></extra>'))
        fig_1h.add_trace(go.Scatter(x=timestamps_shifted, y=avgHighPrice, mode='lines+markers', name='High Price',
                     line=dict(color=high_price_color, width=3),
                     marker=dict(color=high_price_color, size=5),
                                 yaxis='y1',
                                 hovertemplate='Time: %{x}<br>High Price: %{y}<extra></extra>'))

        fig_1h.add_trace(go.Bar(x=timestamps_shifted, y=lowVol, name='Low Vol', marker_color=low_vol_color, yaxis='y2',
                             hovertemplate='Time: %{x}<br>Low Vol: %{y}<extra></extra>'))
        fig_1h.add_trace(go.Bar(x=timestamps_shifted, y=highVol, name='High Vol', marker_color=high_vol_color, yaxis='y2',
                             hovertemplate='Time: %{x}<br>High Vol: %{y}<extra></extra>'))
        
        # Add dashed horizontal lines for table lowP/highP values
        if timestamps_shifted and len(timestamps_shifted) > 0:
            dense_x_1h = build_dense_hover_x(timestamps_shifted)
            
            # Compute fallback values from chart data
            fallback_low = None
            fallback_high = None
            if avgLowPrice and len(avgLowPrice) > 0:
                try:
                    valid_prices = [p for p in avgLowPrice if p is not None and isinstance(p, (int, float)) and p > 0]
                    if valid_prices:
                        fallback_low = sum(valid_prices) / len(valid_prices)
                except Exception:
                    pass
            if avgHighPrice and len(avgHighPrice) > 0:
                try:
                    valid_prices = [p for p in avgHighPrice if p is not None and isinstance(p, (int, float)) and p > 0]
                    if valid_prices:
                        fallback_high = sum(valid_prices) / len(valid_prices)
                except Exception:
                    pass
            
            # Use table values if available, otherwise fall back to chart averages
            final_low = line_low_price if line_low_price is not None else fallback_low
            final_high = line_high_price if line_high_price is not None else fallback_high
            
            print(f"[DEBUG] Final dashed line values: low={final_low}, high={final_high}")
            
            # Always render dashed lines even if we had to use fallback values
            if final_low is not None and final_low > 0:
                fig_1h.add_hline(
                    y=final_low,
                    line_dash='dash',
                    line_color='rgba(0, 72, 255, 0.75)',
                    line_width=2,
                    opacity=0.55
                )
            
            if final_high is not None and final_high > 0:
                fig_1h.add_hline(
                    y=final_high,
                    line_dash='dash',
                    line_color='rgba(255, 0, 0, 0.75)',
                    line_width=2,
                    opacity=0.55
                )
        
        fig_1h.update_layout(
            title=f"Price & Volume History (1h) for {table_data[row].get('name', item_id)}",
            hovermode='closest',
            yaxis=dict(
                title='Price',
                showgrid=True,
                zeroline=True,
                side='left',
            ),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=False,
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_dark' if dark_mode else 'plotly'
        )
        # Add forecasted_prices to the 1h chart if available
        forecasted = None
        if forecasted_prices:
            if isinstance(forecasted_prices, dict):
                forecasted = forecasted_prices.get(item_id)
            elif isinstance(forecasted_prices, list):
                for entry in forecasted_prices:
                    if isinstance(entry, dict) and entry.get('id') == item_id:
                        forecasted = entry.get('forecasted_prices')
                        break
        if forecasted and isinstance(forecasted, list):
            forecast_times = []
            all_x = list(timestamps_shifted)
            if timestamps:
                last_time = datetime.strptime(timestamps_shifted[-1], '%Y-%m-%d %H:%M:%S')
                if len(forecasted) > 0:
                    if ("forecast_sell_time" in locals() and forecast_sell_time is not None and forecast_sell_time > 180):
                        interval = 60
                    else:
                        interval = 5
                else:
                    interval = 5
                if interval == 5:
                    minute = last_time.minute
                    next_minute = ((minute // 5) + 1) * 5
                    if next_minute >= 60:
                        next_hour = last_time.hour + 1
                        next_minute = 0
                    else:
                        next_hour = last_time.hour
                    first_forecast_time = last_time.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)
                else:
                    if last_time.minute > 0 or last_time.second > 0:
                        first_forecast_time = last_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    else:
                        first_forecast_time = last_time + timedelta(hours=1)
                for i, price in enumerate(forecasted):
                    future_time = first_forecast_time + timedelta(minutes=i*interval)
                    forecast_times.append(future_time.strftime('%Y-%m-%d %H:%M:%S'))
            all_x.extend(forecast_times)
            forecast_x = forecast_times
            forecast_y = list(forecasted)
            forecast_marker_sizes = []
            for t in forecast_times:
                dt = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                if forecast_sell_time <= 180:
                    forecast_marker_sizes.append(10 if dt.minute == 0 else 1)
                else:
                    forecast_marker_sizes.append(10 if dt.minute == 0 else 1)
            fig_1h.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode='lines+markers',
                marker=dict(size=forecast_marker_sizes),
                name='Forecasted Prices',
                line=dict(color=forecast_color, width=3, dash='dot'),
                hovertemplate='Time: %{x}<br>Forecasted Price: %{y}<extra></extra>'
            ))

        # Extend dashed line hover traces to cover forecast range if forecasts exist
        if forecasted and isinstance(forecasted, list) and len(forecast_times) > 0:
            full_x_range = list(timestamps_shifted) + forecast_times
        else:
            full_x_range = list(timestamps_shifted)
        
        # Add hover traces with the full x-axis range
        if full_x_range and len(full_x_range) > 0:
            if final_low is not None and final_low > 0:
                fig_1h.add_trace(go.Scatter(
                    x=full_x_range,
                    y=[final_low] * len(full_x_range),
                    mode='lines',
                    line=dict(width=0),
                    hovertemplate=f'Buy Price: {final_low}<extra></extra>',
                    showlegend=False,
                    name='Buy Line'
                ))
            if final_high is not None and final_high > 0:
                fig_1h.add_trace(go.Scatter(
                    x=full_x_range,
                    y=[final_high] * len(full_x_range),
                    mode='lines',
                    line=dict(width=0),
                    hovertemplate=f'Sell Price: {final_high}<extra></extra>',
                    showlegend=False,
                    name='Sell Line'
                ))
            # Set x-axis range to show all data without excessive whitespace
            fig_1h.update_xaxes(range=[full_x_range[0], full_x_range[-1]])

        # Mark the end of 1h chart construction
        t_fig1h_end = time.perf_counter()

        # --- 5m chart (new) ---
        # Load 5m data for this item
        t_5m_load_start = time.perf_counter()
        t_5m_load_end = t_5m_load_start
        try:
            from backend import get_5m_prices
            five_m_prices = get_5m_prices()
            item_5m = five_m_prices.get(str(item_id)) or five_m_prices.get(int(item_id))
            if item_5m and len(item_5m) > 0:
                # Convert 5m timestamps from UTC to local timezone for chart display
                ts_5m = []
                for e in item_5m:
                    raw_ts = e['timestamp']
                    try:
                        local_ts = datetime.strptime(raw_ts, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        local_ts = raw_ts
                    ts_5m.append(local_ts)
                avgLow_5m = [e.get('avgLowPrice') for e in item_5m]
                avgHigh_5m = [e.get('avgHighPrice') for e in item_5m]
                lowVol_5m = [e.get('lowPriceVolume') for e in item_5m]
                highVol_5m = [e.get('highPriceVolume') for e in item_5m]
                fig_5m = go.Figure()
                fig_5m.add_trace(go.Scatter(x=ts_5m, y=avgLow_5m, mode='lines+markers', name='Low Price',
                                            line=dict(color=low_price_color, width=3),
                                            marker=dict(color=low_price_color, size=5),
                                            yaxis='y1',
                                            hovertemplate='Time: %{x}<br>Low Price: %{y}<extra></extra>'))
                fig_5m.add_trace(go.Scatter(x=ts_5m, y=avgHigh_5m, mode='lines+markers', name='High Price',
                                            line=dict(color=high_price_color, width=3),
                                            marker=dict(color=high_price_color, size=5),
                                            yaxis='y1',
                                            hovertemplate='Time: %{x}<br>High Price: %{y}<extra></extra>'))
                fig_5m.add_trace(go.Bar(x=ts_5m, y=lowVol_5m, name='Low Vol', marker_color=low_vol_color, yaxis='y2',
                                       hovertemplate='Time: %{x}<br>Low Vol: %{y}<extra></extra>'))
                fig_5m.add_trace(go.Bar(x=ts_5m, y=highVol_5m, name='High Vol', marker_color=high_vol_color, yaxis='y2',
                                       hovertemplate='Time: %{x}<br>High Vol: %{y}<extra></extra>'))
                
                t_5m_load_end = time.perf_counter()
                print(f"[PERF] 5m data loaded & traces added: {(t_5m_load_end - t_5m_load_start):.3f}s", flush=True)
                
                t_5m_lines_start = time.perf_counter()
                
                # Add dashed horizontal lines for table lowP/highP values
                if ts_5m and len(ts_5m) > 0:
                    dense_x_5m = build_dense_hover_x(ts_5m)
                    
                    # Compute fallback values from chart data
                    fallback_low_5m = None
                    fallback_high_5m = None
                    if avgLow_5m and len(avgLow_5m) > 0:
                        try:
                            valid_prices = [p for p in avgLow_5m if p is not None and isinstance(p, (int, float)) and p > 0]
                            if valid_prices:
                                fallback_low_5m = sum(valid_prices) / len(valid_prices)
                        except Exception:
                            pass
                    if avgHigh_5m and len(avgHigh_5m) > 0:
                        try:
                            valid_prices = [p for p in avgHigh_5m if p is not None and isinstance(p, (int, float)) and p > 0]
                            if valid_prices:
                                fallback_high_5m = sum(valid_prices) / len(valid_prices)
                        except Exception:
                            pass
                    
                    # Use table values if available, otherwise fall back to chart averages
                    final_low_5m = line_low_price if line_low_price is not None else fallback_low_5m
                    final_high_5m = line_high_price if line_high_price is not None else fallback_high_5m
                    
                    # Always render dashed lines even if we had to use fallback values
                    if final_low_5m is not None and final_low_5m > 0:
                        fig_5m.add_hline(
                            y=final_low_5m,
                            line_dash='dash',
                            line_color='rgba(0, 72, 255, 0.75)',
                            line_width=2,
                            opacity=0.55
                        )
                        # Add invisible hover trace for buy line (5m)
                        fig_5m.add_trace(go.Scatter(
                            x=dense_x_5m,
                            y=[final_low_5m] * len(dense_x_5m),
                            mode='lines',
                            line=dict(width=0),
                            hovertemplate=f'Buy Price: {final_low_5m}<extra></extra>',
                            showlegend=False,
                            name='Buy Line'
                        ))
                    
                    if final_high_5m is not None and final_high_5m > 0:
                        fig_5m.add_hline(
                            y=final_high_5m,
                            line_dash='dash',
                            line_color='rgba(255, 0, 0, 0.75)',
                            line_width=2,
                            opacity=0.55
                        )
                        # Add invisible hover trace for sell line (5m)
                        fig_5m.add_trace(go.Scatter(
                            x=dense_x_5m,
                            y=[final_high_5m] * len(dense_x_5m),
                            mode='lines',
                            line=dict(width=0),
                            hovertemplate=f'Sell Price: {final_high_5m}<extra></extra>',
                            showlegend=False,
                            name='Sell Line'
                        ))
                # Support/resistance for 5m
                t_support_start = time.perf_counter()
                try:
                    price_df_5m = pandas_mod.DataFrame({
                        'timestamp': ts_5m,
                        'low': avgLow_5m,
                        'high': avgHigh_5m
                    })
                    price_df_5m['low_no_outlier'] = price_df_5m['low'].rolling(window=12, min_periods=1).apply(lambda x: exclude_outliers(pandas_mod.Series(x)).mean(), raw=False)
                    price_df_5m['high_no_outlier'] = price_df_5m['high'].rolling(window=12, min_periods=1).apply(lambda x: exclude_outliers(pandas_mod.Series(x)).mean(), raw=False)
                    price_df_5m['support'] = price_df_5m['low_no_outlier'].rolling(window=12, min_periods=1).quantile(0.10)
                    price_df_5m['resistance'] = price_df_5m['high_no_outlier'].rolling(window=12, min_periods=1).quantile(0.90)
                    fig_5m.add_trace(go.Scatter(
                        x=price_df_5m['timestamp'],
                        y=price_df_5m['support'],
                        mode='lines',
                        name='Support (10th %ile, robust)',
                        line=dict(color='green', width=2, dash='dash'),
                        yaxis='y1',
                        hovertemplate='Time: %{x}<br>Support: %{y}<extra></extra>'
                    ))
                    fig_5m.add_trace(go.Scatter(
                        x=price_df_5m['timestamp'],
                        y=price_df_5m['resistance'],
                        mode='lines',
                        name='Resistance (90th %ile, robust)',
                        line=dict(color='orange', width=2, dash='dash'),
                        yaxis='y1',
                        hovertemplate='Time: %{x}<br>Resistance: %{y}<extra></extra>'
                    ))
                    t_support_end = time.perf_counter()
                    print(f"[PERF] Support/resistance calculation: {(t_support_end - t_support_start):.3f}s", flush=True)
                except Exception as e:
                    print(f"[DEBUG] Error adding 5m support/resistance bands: {e}")
                    t_support_end = time.perf_counter()
                    print(f"[PERF] Support/resistance calculation (failed): {(t_support_end - t_support_start):.3f}s", flush=True)
                fig_5m.update_layout(
                    title=f"Price & Volume History (5m) for {table_data[row].get('name', item_id)}",
                    hovermode='closest',
                    yaxis=dict(
                        title='Price',
                        showgrid=True,
                        zeroline=True,
                        side='left',
                    ),
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        zeroline=False,
                    ),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    template='plotly_dark' if dark_mode else 'plotly'
                )
                # Return both charts stacked vertically
                t_callback_end = time.perf_counter()
                elapsed = t_callback_end - t_callback_start
                t_data_load_dur = t_fig1h_start - t_data_load_start
                t_fig1h_dur = t_fig1h_end - t_fig1h_start
                t_5m_all_dur = t_callback_end - t_5m_load_start
                print(f"[PERF] Breakdown: data_load={t_data_load_dur:.3f}s, fig1h={t_fig1h_dur:.3f}s, 5m_all={t_5m_all_dur:.3f}s", flush=True)
                print(f"[PERF] show_timeseries callback (both charts) in {elapsed:.3f}s", flush=True)
                return html.Div([
                    dcc.Graph(figure=fig_1h),
                    html.Hr(),
                    dcc.Graph(figure=fig_5m)
                ])
            else:
                # No 5m data, just show 1h chart
                t_callback_end = time.perf_counter()
                elapsed = t_callback_end - t_callback_start
                print(f"[PERF] Breakdown: data_load={(t_fig1h_start - t_data_load_start):.3f}s, fig1h={(t_fig1h_end - t_fig1h_start):.3f}s, 5m_load=0s, 5m_construct=0s", flush=True)
                print(f"[PERF] show_timeseries callback (1h only) in {elapsed:.3f}s", flush=True)
                return html.Div([
                    dcc.Graph(figure=fig_1h),
                    html.Div('No 5m data available for this item.', style={'color': 'gray', 'marginTop': '10px'})
                ])
        except Exception as e:
            print(f"[DEBUG] Error loading/plotting 5m data: {e}")
            if 't_fig1h_end' not in locals():
                t_fig1h_end = time.perf_counter()
            t_callback_end = time.perf_counter()
            elapsed = t_callback_end - t_callback_start
            t_data_load_dur = (t_fig1h_start - t_data_load_start) if 't_fig1h_start' in locals() else 0
            t_fig1h_dur = (t_fig1h_end - t_fig1h_start) if 't_fig1h_start' in locals() else 0
            print(f"[PERF] Breakdown: data_load={t_data_load_dur:.3f}s, fig1h={t_fig1h_dur:.3f}s, 5m_error", flush=True)
            print(f"[PERF] show_timeseries callback (error) in {elapsed:.3f}s", flush=True)
            return html.Div([
                dcc.Graph(figure=fig_1h),
                html.Div('Error loading 5m data.', style={'color': 'red', 'marginTop': '10px'})
            ])
    t_callback_end = time.perf_counter()
    elapsed = t_callback_end - t_callback_start
    print(f"[PERF] show_timeseries callback (no row) in {elapsed:.3f}s", flush=True)
    return html.Div('No time series data available for this item.')

# --- Debug callback for template-dropdown selection ---

# --- Merged Save/Load Filter Template Callback ---
@app.callback(
    [
        Output('template-status', 'children'),
        Output('template-dropdown', 'options'),
        Output('template-dropdown', 'value'),
        Output('forecast-strategy', 'value'),
        Output('forecast-sell-time', 'value'),
        Output('min-price', 'value'),
        Output('max-price', 'value'),
        Output('min-avg-daily-volume', 'value'),
        Output('max-avg-trade-time', 'value'),
        Output('min-profit', 'value'),
        Output('min-potential-profit', 'value'),
        Output('min-volume', 'value'),
        Output('min-roi', 'value'),
        Output('forecast-hours', 'value'),
        Output('volume-power', 'value'),
        Output('max-qty-factor', 'value'),
        Output('trend-filter', 'value'),
        Output('top-n', 'value'),
        Output('sort-attribute', 'value'),
        Output('risk-filter', 'value'),
        Output('status-filter', 'value'),
        Output('buy-price-type', 'value'),
        Output('forecast-price-type', 'value'),
        Output('template-name', 'value')
    ],
    [Input('save-template', 'n_clicks'), Input('load-template', 'n_clicks'), Input('delete-template', 'n_clicks')],
    [State('template-name', 'value'),
     State('forecast-strategy', 'value'),
     State('forecast-sell-time', 'value'),
     State('min-price', 'value'),
     State('max-price', 'value'),
     State('min-avg-daily-volume', 'value'),
     State('max-avg-trade-time', 'value'),
     State('min-profit', 'value'),
     State('min-potential-profit', 'value'),
     State('min-volume', 'value'),
     State('min-roi', 'value'),
     State('forecast-hours', 'value'),
     State('volume-power', 'value'),
    State('max-qty-factor', 'value'),
     State('trend-filter', 'value'),
     State('top-n', 'value'),
     State('sort-attribute', 'value'),
     State('risk-filter', 'value'),
     State('status-filter', 'value'),
     State('buy-price-type', 'value'),
     State('forecast-price-type', 'value'),
     State('template-dropdown', 'value')]
)
def template_save_load_callback(save_clicks, load_clicks, delete_clicks, template_name,
    forecast_strategy, forecast_sell_time, min_price, max_price, min_avg_daily_volume, max_avg_trade_time, min_profit,
    min_potential_profit, min_volume, min_roi, forecast_hours, volume_power, max_qty_factor, trend_filter,
    top_n, sort_attribute, risk_filter, status_filter, buy_price_type, forecast_price_type, selected_template):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    options = [{'label': name, 'value': name} for name in list_filter_templates()]
    # Save template (auto-include all filter fields)
    if trigger == 'save-template':
        if not save_clicks or not template_name:
            raise PreventUpdate
        # List of all filter input IDs in the UI (update this list as you add new filters)
        filter_ids = [
            'forecast_strategy', 'forecast_sell_time', 'min_price', 'max_price', 'min_avg_daily_volume', 'max_avg_trade_time', 'min_profit',
            'min_potential_profit', 'min_volume', 'min_roi', 'forecast_hours', 'volume_power', 'max_qty_factor', 'trend_filter',
            'top_n', 'sort_attribute', 'risk_filter', 'status_filter', 'buy_price_type', 'forecast_price_type'
        ]
        # Map input IDs to their values from the function arguments or dash callback context
        arg_map = {
            'forecast_strategy': forecast_strategy,
            'forecast_sell_time': forecast_sell_time,
            'min_price': min_price,
            'max_price': max_price,
            'min_avg_daily_volume': min_avg_daily_volume,
            'max_avg_trade_time': max_avg_trade_time,
            'min_profit': min_profit,
            'min_potential_profit': min_potential_profit,
            'min_volume': min_volume,
            'min_roi': min_roi,
            'forecast_hours': forecast_hours,
            'volume_power': volume_power,
            'max_qty_factor': max_qty_factor,
            'trend_filter': trend_filter,
            'top_n': top_n,
            'sort_attribute': sort_attribute,
            'risk_filter': risk_filter,
            'status_filter': status_filter,
            'buy_price_type': buy_price_type,
            'forecast_price_type': forecast_price_type,
        }
        # Add any extra filter values from dash callback context (for new filters)
        for fid in filter_ids:
            if fid not in arg_map:
                # Try to get from dash callback context states
                val = None
                for s in ctx.states:
                    if s.startswith(fid + "."):
                        val = ctx.states[s]
                        break
                if val is not None:
                    arg_map[fid] = val
        values = {k: v for k, v in arg_map.items()}
        success = save_filter_template(template_name, values)
        status = f"Template '{template_name}' saved." if success else f"Error saving template '{template_name}'."
        options = [{'label': name, 'value': name} for name in list_filter_templates()]
        return [status, options, template_name] + [dash.no_update]*21
    # Load template (auto-include all filter fields)
    elif trigger == 'load-template':
        if not load_clicks or not selected_template:
            raise PreventUpdate
        template = load_filter_template(selected_template)
        if not template:
            return [f"Template '{selected_template}' not found.", options, selected_template] + [dash.no_update]*19
        # List of all filter output order (must match callback Outputs)
        output_order = [
            'forecast_strategy', 'forecast_sell_time', 'min_price', 'max_price', 'min_avg_daily_volume', 'max_avg_trade_time',  'min_profit',
            'min_potential_profit', 'min_volume', 'min_roi', 'forecast_hours', 'volume_power', 'max_qty_factor', 'trend_filter',
            'top_n', 'sort_attribute', 'risk_filter', 'status_filter', 'buy_price_type', 'forecast_price_type', 'template_name'
        ]
        # Build result list in output order, using template values or defaults
        defaults = {
            'forecast_strategy': 'Weighted Moving Avg (WMA)',
            'forecast_sell_time': 60,
            'min_price': 0,
            'max_price': 0,
            'min_avg_daily_volume': 0,
            'max_avg_trade_time': 0,
            'min_profit': 1,
            'min_potential_profit': 10000,
            'min_volume': 6,
            'min_roi': 5,
            'forecast_hours': 72,
            'volume_power': 1,
            'max_qty_factor': 1,
            'trend_filter': '',
            'top_n': 20,
            'sort_attribute': 'potential_profit',
            'risk_filter': 'all',
            'status_filter': [],
            'buy_price_type': 'low',
            'forecast_price_type': 'avg',
            'template_name': dash.no_update
        }
        result = [f"Template '{selected_template}' loaded.", options, selected_template]
        for key in output_order:
            result.append(template.get(key, defaults.get(key, dash.no_update)))
        # Ensure result is exactly the right length (3 fixed + 21 filter values)
        while len(result) < 24:
            result.append(dash.no_update)
        return result[:24]
    # Delete template
    elif trigger == 'delete-template':
        if not delete_clicks or not selected_template:
            raise PreventUpdate
        success = delete_filter_template(selected_template)
        options = [{'label': name, 'value': name} for name in list_filter_templates()]
        status = f"Template '{selected_template}' deleted." if success else f"Error deleting template '{selected_template}'."
        return [status, options, None] + [dash.no_update]*19
    raise PreventUpdate

# --- Watchlist Add/Remove Callback ---
@app.callback(
    # [Output('remove-selected-status', 'children'), Output('watchlist-table', 'data')],
    [Input('remove-selected-from-watchlist-btn', 'n_clicks')],
    [State('watchlist-table', 'selected_rows'), State('watchlist-table', 'data')]
)
def remove_selected_from_watchlist(n_clicks, selected_rows, table_data):
    print('[DEBUG] remove_selected_from_watchlist called')
    if n_clicks and n_clicks > 0:
        if not selected_rows or not table_data:
            print('[DEBUG] No selected row or table data')
            return 'No row selected.', table_data
        row_idx = selected_rows[0] if isinstance(selected_rows, list) and selected_rows else None
        if row_idx is None or row_idx < 0 or row_idx >= len(table_data):
            print(f'[DEBUG] Invalid row index: {row_idx}')
            return 'Invalid row selected.', table_data
        item = table_data[row_idx]
        item_id = item.get('item_id')
        item_name = item.get('item_name', '')
        watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
        try:
            with open(watchlist_path, 'r', encoding='utf-8') as f:
                watchlist = json.load(f)
        except Exception:
            watchlist = []
        # Remove item
        new_watchlist = [w for w in watchlist if str(w.get('item_id')) != str(item_id)]
        if len(new_watchlist) == len(watchlist):
            print(f'[DEBUG] Item {item_id} not found in watchlist')
            return f'Item {item_name} not found in watchlist.', table_data
        try:
            with open(watchlist_path, 'w', encoding='utf-8') as f:
                json.dump(new_watchlist, f, ensure_ascii=False, indent=2)
            print(f'[DEBUG] Removed {item_name} from watchlist')
            return f'Removed {item_name} from watchlist.', new_watchlist
        except Exception as e:
            print(f'[ERROR] Failed to save watchlist: {e}')
            return f'Error saving watchlist: {e}', table_data
    print('[DEBUG] Button not clicked yet')
    return dash.no_update, table_data
@app.callback(
    Output('add-selected-status', 'children'),
    [Input('add-selected-to-watchlist-btn', 'n_clicks')],
    [State('forecast-table', 'selected_rows'), State('forecast-table', 'data')]
)
def add_selected_to_watchlist(n_clicks, selected_rows, table_data):
    print('[DEBUG] add_selected_to_watchlist called')
    if n_clicks and n_clicks > 0:
        if not selected_rows or not table_data:
            print('[DEBUG] No selected row or table data')
            return 'No row selected.'
        row_idx = selected_rows[0] if isinstance(selected_rows, list) and selected_rows else None
        if row_idx is None or row_idx < 0 or row_idx >= len(table_data):
            print(f'[DEBUG] Invalid row index: {row_idx}')
            return 'Invalid row selected.'
        item = table_data[row_idx]
        item_id = item.get('id')
        item_name = item.get('name', '')
        entry_price = item.get('lowP', 0)
        watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
        try:
            with open(watchlist_path, 'r', encoding='utf-8') as f:
                watchlist = json.load(f)
        except Exception:
            watchlist = []
        # Check for existing item
        existing_item = next((w for w in watchlist if str(w.get('item_id')) == str(item_id)), None)
        if existing_item:
            print(f'[DEBUG] Item {item_id} already in watchlist')
            return f'Item {item_name} already in watchlist.'
        new_entry = {
            'item_id': item_id,
            'item_name': item_name,
            'entry_price': entry_price,
            'quantity': 1,
            'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        watchlist.append(new_entry)
        try:
            with open(watchlist_path, 'w', encoding='utf-8') as f:
                json.dump(watchlist, f, ensure_ascii=False, indent=2)
            print(f'[DEBUG] Added {item_name} to watchlist')
            return f'Added {item_name} to watchlist.'
        except Exception as e:
            print(f'[ERROR] Failed to save watchlist: {e}')
            return f'Error saving watchlist: {e}'
    print('[DEBUG] Button not clicked yet')
    return dash.no_update

# --- Watchlist Table Update Callback ---

## Removed all callbacks referencing watchlist-table (pagination, sorting, data, etc.)


# --- Unified debug callback for add-watchlist-status.children ---
@app.callback(
    # ...existing code...
    [
        Input('forecast-table', 'active_cell'),
        Input('forecast-table', 'data'),
        Input('item-name-search', 'value'),
        Input('filter-by-watchlist', 'value'),
        Input('apply-filters', 'n_clicks'),
        Input('refresh-interval', 'n_intervals'),
        Input('save-template', 'n_clicks'),
        Input('load-template', 'n_clicks'),
        Input('delete-template', 'n_clicks'),
        # Input('watchlist-table', 'page_current'),
        # Input('watchlist-table', 'page_size'),
        # Input('watchlist-table', 'sort_by'),
        # Input('watchlist-table', 'data'),
        # Input('watchlist-table', 'active_cell'),
    ],
    [
        State('forecast-table', 'data'),
        # State('watchlist-table', 'data')
    ]
)
def unified_watchlist_status(
    forecast_active_cell, forecast_table_data, item_name_search, filter_by_watchlist, apply_filters_clicks,
    refresh_interval, save_template_clicks, load_template_clicks, delete_template_clicks,
    # watchlist_page_current, watchlist_page_size, 
    # watchlist_sort_by, watchlist_table_data, watchlist_active_cell,
    state_forecast_table_data
):
    ctx = dash.callback_context
    # All branches removed; do not return anything for no-output callback
    pass



# # --- Debug callback for apply-filters button ---
# @app.callback(
#     # ...existing code...
#     [Input('apply-filters', 'n_clicks')]
# )
# def debug_apply_filters_button(n_clicks):
#     return f"Apply filters button clicked {n_clicks} times."

# # --- Debug callback for refresh-interval component ---
# @app.callback(
#     # ...existing code...
#     [Input('refresh-interval', 'n_intervals')]
# )
# def debug_refresh_interval_component(n_intervals):
#     return f"Refresh interval: {n_intervals}"

# # --- Debug callback for save-template button ---
# @app.callback(
#     # ...existing code...
#     [Input('save-template', 'n_clicks')]
# )
# def debug_save_template_button(n_clicks):
#     return f"Save template button clicked {n_clicks} times."

# # --- Debug callback for load-template button ---
# @app.callback(
#     # ...existing code...
#     [Input('load-template', 'n_clicks')]
# )
# def debug_load_template_button(n_clicks):
#     return f"Load template button clicked {n_clicks} times."

# # --- Debug callback for delete-template button ---
# @app.callback(
#     # ...existing code...
#     [Input('delete-template', 'n_clicks')]
# )
# def debug_delete_template_button(n_clicks):
#     return f"Delete template button clicked {n_clicks} times."

# # --- Debug callback for watchlist-table pagination ---
# @app.callback(
#     # ...existing code...
#     [Input('watchlist-table', 'page_current'),
#      Input('watchlist-table', 'page_size')]
# )
# def debug_watchlist_table_pagination(page_current, page_size):
#     return f"Watchlist table pagination: page {page_current}, size {page_size}"

# # --- Debug callback for watchlist-table sorting ---
# @app.callback(
#     # ...existing code...
#     [Input('watchlist-table', 'sort_by')]
# )
# def debug_watchlist_table_sorting(sort_by):
#     return f"Watchlist table sorting: {sort_by}"


# # --- Debug callbacks for add-watchlist-status removed (lines 1372-1540) ---

# --- Main block to launch Dash app and update 5m data on startup ---
if __name__ == '__main__':
    import json
    import os
    import traceback
    
    try:
        print(f"[DEBUG] APP_DIR: {APP_DIR}")
        print(f"[DEBUG] sys.frozen: {getattr(sys, 'frozen', False)}")
        print(f"[DEBUG] sys.executable: {sys.executable}")
        
        DATA_DIR = os.path.join(APP_DIR, 'Data')
        print(f"[DEBUG] DATA_DIR: {DATA_DIR}")
        print(f"[DEBUG] DATA_DIR exists: {os.path.exists(DATA_DIR)}")
        
        FIVE_M_FILE = os.path.join(DATA_DIR, '5m_data_cache.json')
        
        def latest_chunk_is_current():
            try:
                with open(FIVE_M_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                chunks = cache.get('chunks', [])
                if not chunks:
                    return False
                latest_ts = max(chunk['timestamp'] for chunk in chunks)
                # Get current 5-min mark (UTC)
                import datetime
                now_utc = datetime.datetime.now(datetime.timezone.utc).replace(second=0, microsecond=0)
                current_5m = now_utc.replace(minute=(now_utc.minute // 5) * 5)
                current_5m_str = current_5m.strftime('%Y-%m-%dT%H:%M:%S')
                return latest_ts == current_5m_str
            except Exception:
                return False

        try:
            from fetch_5m_data import fetch_and_save_5m_data    
            if not latest_chunk_is_current():
                fetch_and_save_5m_data()
            else:   
                print("5m data is already up to date on startup.")
        except Exception as e:
            print(f"Error fetching 5m data on startup: {e}")
            traceback.print_exc()
        
        print("[DEBUG] Starting Dash app...")
        app.run(
            debug=False,
            use_reloader=False,
            dev_tools_hot_reload=False,
            dev_tools_ui=False,
            dev_tools_props_check=False,
            host='127.0.0.1',
            port=8050
        )
    except Exception as e:
        print(f">>[FATAL ERROR]<< {e}")
        traceback.print_exc()
        import time
        time.sleep(5)
        raise