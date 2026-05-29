# TO DO:
# - Better spike detection and handling in evaluate_forecast_for_item
# - Forecast validation
# - Status filtering in analysis tab
# - Descriptive status messages
# - Filter template should use all filter fields


import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import os
import sys
import math
import re
import importlib
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
    get_latest_prices,
    get_5m_prices,
    GE_TAX
)
from forecast_helpers import forecast_avg_price


def fetch_and_save_5m_data_safe(log_context=''):
    """Best-effort 5m refresh. Skip silently when optional module is unavailable."""
    context_suffix = f" ({log_context})" if log_context else ''
    try:
        fetch_module = importlib.import_module('fetch_5m_data')
        fetch_and_save_5m_data = getattr(fetch_module, 'fetch_and_save_5m_data', None)
        if fetch_and_save_5m_data is None:
            return False
    except ModuleNotFoundError:
        return False
    except Exception as e:
        return False

    try:
        fetch_and_save_5m_data()
        return True
    except Exception as e:
        print(f"Error fetching 5m data{context_suffix}: {e}")
        return False


def _coerce_positive_float(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value <= 0:
        return None
    return numeric_value


def _normalize_dump_alert_metrics(metric_values):
    allowed_metrics = {'avg_price', 'avg_low_price', 'avg_high_price', 'median_price'}
    # RadioItems returns a single string; stored settings may still be a list
    if isinstance(metric_values, str):
        return [metric_values] if metric_values in allowed_metrics else ['avg_price']
    if not metric_values:
        return ['avg_price']
    normalized = [m for m in metric_values if m in allowed_metrics]
    return [normalized[0]] if normalized else ['avg_price']


def _normalize_dump_alert_windows(window_values):
    if isinstance(window_values, (int, float, str)):
        window_values = [window_values]

    if not window_values:
        return [30]

    normalized_windows = []
    for value in window_values:
        try:
            window_minutes = int(value)
        except (TypeError, ValueError):
            continue
        if window_minutes >= 5:
            normalized_windows.append(window_minutes)

    if not normalized_windows:
        return [30]

    return sorted(dict.fromkeys(normalized_windows))


def _parse_dump_alert_custom_windows(custom_windows_text):
    if not isinstance(custom_windows_text, str):
        return []

    tokens = [token.strip().lower() for token in re.split(r'[\s,;]+', custom_windows_text) if token.strip()]
    parsed_windows = []
    for token in tokens:
        match = re.fullmatch(r'(\d+)([mhd]?)', token)
        if not match:
            continue

        numeric_value = int(match.group(1))
        unit = match.group(2) or 'm'
        if numeric_value <= 0:
            continue

        if unit == 'h':
            numeric_value *= 60
        elif unit == 'd':
            numeric_value *= 1440

        parsed_windows.append(numeric_value)

    return _normalize_dump_alert_windows(parsed_windows) if parsed_windows else []


def _merged_dump_alert_windows(window_values, custom_windows_text):
    if isinstance(window_values, (list, tuple, set)) and len(window_values) > 0:
        preset_windows = _normalize_dump_alert_windows(window_values)
    elif isinstance(window_values, (int, float, str)):
        preset_windows = _normalize_dump_alert_windows([window_values])
    else:
        preset_windows = []

    custom_windows = _parse_dump_alert_custom_windows(custom_windows_text)
    if not custom_windows and not preset_windows:
        return [30]
    if not custom_windows:
        return preset_windows
    return _normalize_dump_alert_windows(preset_windows + custom_windows)


def _extract_dump_alert_price(low_price, high_price, metric_key):
    normalized_low = _coerce_positive_float(low_price)
    normalized_high = _coerce_positive_float(high_price)

    if metric_key == 'avg_low_price':
        return normalized_low
    if metric_key == 'avg_high_price':
        return normalized_high

    available_prices = [price for price in (normalized_low, normalized_high) if price is not None]
    if not available_prices:
        return None
    if metric_key == 'median_price':
        return float(np.median(available_prices))
    if len(available_prices) == 1:
        return available_prices[0]
    return sum(available_prices) / len(available_prices)


def _dump_alert_price_from_history_entry(entry, metric_key):
    return _extract_dump_alert_price(entry.get('avgLowPrice'), entry.get('avgHighPrice'), metric_key)


def _dump_alert_price_from_latest_entry(entry, metric_key):
    return _extract_dump_alert_price(entry.get('low'), entry.get('high'), metric_key)


def _dump_alert_current_price_from_latest_entry(entry):
    if not isinstance(entry, dict):
        return None

    low_price = _coerce_positive_float(entry.get('low'))
    high_price = _coerce_positive_float(entry.get('high'))
    low_time = entry.get('lowTime')
    high_time = entry.get('highTime')

    if low_price is None and high_price is None:
        return None
    if low_price is None:
        return high_price
    if high_price is None:
        return low_price

    try:
        low_time_value = float(low_time) if low_time is not None else None
    except (TypeError, ValueError):
        low_time_value = None
    try:
        high_time_value = float(high_time) if high_time is not None else None
    except (TypeError, ValueError):
        high_time_value = None

    if low_time_value is not None and high_time_value is not None:
        if low_time_value > high_time_value:
            return low_price
        if high_time_value > low_time_value:
            return high_price

    if high_time_value is not None and low_time_value is None:
        return high_price
    if low_time_value is not None and high_time_value is None:
        return low_price

    return high_price


def _dump_alert_metric_label(metric_key):
    metric_labels = {
        'avg_price': 'Avg Price',
        'avg_low_price': 'Avg of Low Prices',
        'avg_high_price': 'Avg of High Prices',
        'median_price': 'Median Price',
    }
    return metric_labels.get(metric_key, metric_key)


def _dump_alert_chart_row_from_result(row):
    if not isinstance(row, dict):
        return None

    item_id = str(row.get('item_id') or row.get('id') or '')
    if not item_id:
        return None

    latest_low = row.get('latest_low')
    latest_high = row.get('latest_high')
    return {
        'id': item_id,
        'name': row.get('item_name') or f'Item {item_id}',
        'lowP': latest_low,
        'highP': latest_high,
        'buy_price': latest_low,
        'sell_price': latest_high,
        'forecast_price': row.get('reference_price'),
    }


def _dump_alert_y_fit_range_from_lines(low_line, high_line, y_fit_pct=15):
    low_value = _coerce_positive_float(low_line)
    high_value = _coerce_positive_float(high_line)

    if low_value is None or high_value is None:
        return None

    # Ensure correct order regardless of which price happens to be larger.
    low_value, high_value = min(low_value, high_value), max(low_value, high_value)

    try:
        pct_value = float(y_fit_pct) if y_fit_pct is not None else 15.0
    except (TypeError, ValueError):
        pct_value = 15.0
    pct_value = max(0.0, pct_value) / 100.0

    lower_bound = low_value * (1 - pct_value)
    upper_bound = high_value * (1 + pct_value)

    if upper_bound <= lower_bound:
        return None

    return [lower_bound, upper_bound]


def _dump_alert_chart_control_theme():
    return {
        'bgcolor': '#e5ecf6',
        'activecolor': '#c7d7ef',
        'bordercolor': '#b8c8df',
        'borderwidth': 1,
        'font_color': '#2a3f5f',
        'font_size': 12,
    }


def _dump_alert_chart_button_style():
    theme = _dump_alert_chart_control_theme()
    return {
        'height': '22px',
        'padding': '0 8px',
        'fontSize': '11px',
        'lineHeight': '20px',
        'border': f"1px solid {theme['bordercolor']}",
        'borderRadius': '3px',
        'backgroundColor': theme['bgcolor'],
        'color': theme['font_color'],
        'cursor': 'pointer',
    }


def _load_item_mapping_cache():
    mapping_path = os.path.join(APP_DIR, 'Data', 'mapping_cache.json')
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_json = json.load(f)
    except Exception:
        return {}

    mapping = mapping_json.get('data', {})
    return mapping if isinstance(mapping, dict) else {}


def _format_utc_datetime_for_display(dt_value):
    return dt_value.astimezone().strftime('%Y-%m-%d %H:%M')


def _dump_alert_base_table_styles():
    return [
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
        {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
        {'if': {'filter_query': '{drop_pct} >= 10', 'column_id': 'drop_pct'}, 'color': '#c62828', 'fontWeight': 'bold'},
        {'if': {'filter_query': '{drop_pct} >= 15'}, 'backgroundColor': '#fff1f1'},
    ]


def _default_dump_alert_settings():
    return {
        'threshold_pct': 5,
        'min_gp_drop': 250,
        'min_avg_daily_volume': 0,
        'min_potential_profit': 0,
        'min_volume_potential': 0,
        'volume_power': 1,
        'max_qty_factor': 1,
        'metrics': ['avg_price'],
        'windows': [30],
        'custom_windows_text': '',
        'alert_options': ['sound', 'highlight'],
        'new_options': [],
        'auto_refresh': ['enabled'],
        'auto_refresh_minutes': 5,
    }


def _build_dump_alert_rows(threshold_pct, min_gp_drop, min_avg_daily_volume, min_potential_profit, min_volume_potential, volume_power, max_qty_factor, metric_values, window_values):
    try:
        threshold_pct = float(threshold_pct) if threshold_pct is not None else 5.0
    except (TypeError, ValueError):
        threshold_pct = 5.0
    threshold_pct = max(0.1, threshold_pct)

    try:
        min_gp_drop = float(min_gp_drop) if min_gp_drop is not None else 0.0
    except (TypeError, ValueError):
        min_gp_drop = 0.0
    min_gp_drop = max(0.0, min_gp_drop)

    try:
        min_avg_daily_volume = float(min_avg_daily_volume) if min_avg_daily_volume is not None else 0.0
    except (TypeError, ValueError):
        min_avg_daily_volume = 0.0
    min_avg_daily_volume = max(0.0, min_avg_daily_volume)

    try:
        min_potential_profit = float(min_potential_profit) if min_potential_profit is not None else 0.0
    except (TypeError, ValueError):
        min_potential_profit = 0.0
    min_potential_profit = max(0.0, min_potential_profit)

    try:
        min_volume_potential = float(min_volume_potential) if min_volume_potential is not None else 0.0
    except (TypeError, ValueError):
        min_volume_potential = 0.0
    min_volume_potential = max(0.0, min_volume_potential)

    try:
        volume_power = float(volume_power) if volume_power is not None else 1.0
    except (TypeError, ValueError):
        volume_power = 1.0
    volume_power = max(0.0, volume_power)

    try:
        max_qty_factor = float(max_qty_factor) if max_qty_factor is not None else 1.0
    except (TypeError, ValueError):
        max_qty_factor = 1.0
    max_qty_factor = max(0.0, max_qty_factor)

    metric_values = _normalize_dump_alert_metrics(metric_values)
    window_values = _normalize_dump_alert_windows(window_values)

    refreshed_5m = fetch_and_save_5m_data_safe('dump-alert')
    five_m_prices = get_5m_prices()
    latest_prices = get_latest_prices()
    mapping = _load_item_mapping_cache()

    rows = []
    items_scanned = 0
    for raw_item_id, history in five_m_prices.items():
        if not history:
            continue

        item_id = str(raw_item_id)
        item_info = mapping.get(item_id, {})
        item_name = item_info.get('name') or f'Item {item_id}'
        buy_limit = item_info.get('buy_limit') or item_info.get('limit') or ''
        try:
            buy_limit_numeric = int(float(buy_limit)) if buy_limit not in ('', None) else 0
        except (TypeError, ValueError):
            buy_limit_numeric = 0
        latest_live_entry = latest_prices.get(item_id, {})

        total_hist_vol = sum(
            int(_coerce_positive_float(entry.get('lowPriceVolume')) or 0)
            + int(_coerce_positive_float(entry.get('highPriceVolume')) or 0)
            for entry in history
        )
        history_days = max(1.0, len(history) / (24.0 * 12.0))
        avg_daily_volume = int(total_hist_vol / history_days) if total_hist_vol > 0 else 0

        low_vol_recent = sum(int(_coerce_positive_float(entry.get('lowPriceVolume')) or 0) for entry in history[-6:])
        high_vol_recent = sum(int(_coerce_positive_float(entry.get('highPriceVolume')) or 0) for entry in history[-6:])

        for metric_key in metric_values:
            valid_points = []
            for entry in history:
                raw_timestamp = entry.get('timestamp')
                if not raw_timestamp:
                    continue
                try:
                    entry_ts = datetime.strptime(raw_timestamp, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                metric_price = _dump_alert_price_from_history_entry(entry, metric_key)
                if metric_price is None:
                    continue
                valid_points.append((entry_ts, metric_price, entry))

            if len(valid_points) < 2:
                continue

            valid_points.sort(key=lambda point: point[0])
            latest_ts, latest_hist_price, latest_hist_entry = valid_points[-1]
            current_price = _dump_alert_current_price_from_latest_entry(latest_live_entry)
            if current_price is None:
                current_price = _dump_alert_price_from_history_entry(latest_hist_entry, 'avg_price')

            if current_price is None:
                continue

            items_scanned += 1

            for window_minutes in window_values:
                cutoff_ts = latest_ts - timedelta(minutes=window_minutes)
                window_points = [point for point in valid_points if point[0] >= cutoff_ts]
                if len(window_points) < 2:
                    continue

                window_metric_values = [point[1] for point in window_points if point[1] is not None]
                if len(window_metric_values) < 2:
                    continue

                if metric_key == 'median_price':
                    reference_price = float(np.median(window_metric_values))
                    reference_seen = 'Window Median'
                else:
                    reference_price = float(np.mean(window_metric_values))
                    reference_seen = 'Window Average'

                current_metric_price = current_price

                if reference_price is None or reference_price <= 0 or current_metric_price is None:
                    continue

                reference_price_int = int(round(reference_price))
                current_metric_price_int = int(round(current_metric_price))
                if reference_price_int <= 0:
                    continue

                gp_drop = reference_price_int - current_metric_price_int
                drop_pct = (gp_drop / reference_price_int) * 100.0
                if drop_pct < threshold_pct or gp_drop < min_gp_drop:
                    continue

                profit = 0
                if current_metric_price_int > 0:
                    taxed_sell_price = int(math.floor(reference_price_int * (1 - GE_TAX)))
                    profit = taxed_sell_price - current_metric_price_int
                if window_minutes < 60:
                    max_qty = min(
                        buy_limit_numeric,
                        int(avg_daily_volume * 0.1 * (30 / 1440)) if avg_daily_volume > 0 else 0,
                        low_vol_recent * 2 if low_vol_recent > 0 else 0,
                    )
                else:
                    max_qty = min(
                        buy_limit_numeric,
                        int(avg_daily_volume * 0.6 * (window_minutes / 1440)) if avg_daily_volume > 0 else 0,
                        low_vol_recent * 2 if low_vol_recent > 0 else 0,
                    )

                if window_minutes < 1440:
                    max_qty = min(max_qty, int(avg_daily_volume * 0.05))

                max_qty = max(0, int(max_qty))
                max_qty = int(max_qty * max_qty_factor) if max_qty > 0 else 0
                max_qty = min(max_qty, buy_limit_numeric)
                potential_profit = profit * max_qty if profit > 0 else 0
                volume_potential = (
                    math.floor(math.log10(abs(potential_profit) * (avg_daily_volume ** volume_power)) * 100) / 100
                    if avg_daily_volume > 0 and potential_profit is not None and abs(potential_profit) * (avg_daily_volume ** volume_power) > 1
                    else 0
                )

                if avg_daily_volume < min_avg_daily_volume:
                    continue
                if potential_profit < min_potential_profit:
                    continue
                if volume_potential < min_volume_potential:
                    continue

                low_volume_window = sum(int(_coerce_positive_float(point[2].get('lowPriceVolume')) or 0) for point in window_points)
                high_volume_window = sum(int(_coerce_positive_float(point[2].get('highPriceVolume')) or 0) for point in window_points)
                latest_seen_ts = latest_ts
                latest_low = latest_live_entry.get('low') if latest_live_entry else latest_hist_entry.get('avgLowPrice')
                latest_high = latest_live_entry.get('high') if latest_live_entry else latest_hist_entry.get('avgHighPrice')
                alert_key = f'{item_id}|{metric_key}|{window_minutes}'

                rows.append({
                    'alert_key': alert_key,
                    'is_new': 0,
                    'item_id': item_id,
                    'item_name': item_name,
                    'metric': _dump_alert_metric_label(metric_key),
                    'metric_key': metric_key,
                    'drop_pct': round(drop_pct, 2),
                    'gp_drop': int(round(gp_drop)),
                    'reference_price': reference_price_int,
                    'current_price': current_metric_price_int,
                    'window_minutes': window_minutes,
                    'reference_seen': reference_seen,
                    'latest_seen': _format_utc_datetime_for_display(latest_seen_ts),
                    'latest_low': int(round(float(latest_low))) if latest_low is not None else '',
                    'latest_high': int(round(float(latest_high))) if latest_high is not None else '',
                    'window_low_volume': low_volume_window,
                    'window_high_volume': high_volume_window,
                    'avg_daily_volume': avg_daily_volume,
                    'profit_per_item': profit,
                    'buy_qty': max_qty,
                    'potential_profit': int(round(potential_profit)),
                    'volume_potential': round(volume_potential, 2),
                    'buy_limit': buy_limit,
                })

    rows.sort(key=lambda row: (-row['drop_pct'], -row['gp_drop'], row['window_minutes'], row['item_name']))
    return rows, items_scanned, refreshed_5m, threshold_pct, min_gp_drop, min_avg_daily_volume, min_potential_profit, min_volume_potential, volume_power, max_qty_factor, metric_values, window_values


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
    'FORECAST_HOURS': 168,
    'FORECAST_RECENCY_MINUTES': 0
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
        dcc.Tab(label='Dump Alert', value='dump_alert'),
    ]),
    html.Div(id='tab-content'),
    dcc.Store(id='forecasted-prices-store'),
    dcc.Store(id='dark-mode-store', data={'dark_mode': False}),
    dcc.Store(id='dump-alert-settings-store', storage_type='local', data=_default_dump_alert_settings()),
    dcc.Store(id='dump-alert-refresh-state-store', data={'keys': [], 'sticky_new_keys': []}),
    dcc.Store(id='dump-alert-sound-signal', data={'play': False, 'nonce': 0}),
], id='app-container', style={'minHeight': '100vh'})

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


@app.callback(
    [
        Output('dump-alert-threshold-pct', 'value'),
        Output('dump-alert-min-gp-drop', 'value'),
        Output('dump-alert-min-avg-daily-volume', 'value'),
        Output('dump-alert-min-potential-profit', 'value'),
        Output('dump-alert-min-volume-potential', 'value'),
        Output('dump-alert-volume-power', 'value'),
        Output('dump-alert-max-qty-factor', 'value'),
        Output('dump-alert-metrics', 'value'),
        Output('dump-alert-windows', 'value'),
        Output('dump-alert-custom-windows', 'value'),
        Output('dump-alert-options', 'value'),
        Output('dump-alert-new-options', 'value'),
        Output('dump-alert-auto-refresh', 'value'),
        Output('dump-alert-auto-refresh-minutes', 'value'),
    ],
    Input('main-tabs', 'value'),
    State('dump-alert-settings-store', 'data'),
    prevent_initial_call=False
)
def load_dump_alert_settings(active_tab, settings_store):
    if active_tab != 'dump_alert':
        raise PreventUpdate

    settings = _default_dump_alert_settings()
    if isinstance(settings_store, dict):
        settings.update(settings_store)

    return (
        settings.get('threshold_pct', 5),
        settings.get('min_gp_drop', 250),
        settings.get('min_avg_daily_volume', 0),
        settings.get('min_potential_profit', 0),
        settings.get('min_volume_potential', 0),
        settings.get('volume_power', 1),
        settings.get('max_qty_factor', 1),
        _normalize_dump_alert_metrics(settings.get('metrics'))[0],
        _normalize_dump_alert_windows(settings.get('windows')),
        settings.get('custom_windows_text', ''),
        settings.get('alert_options', ['sound', 'highlight']) or [],
        settings.get('new_options', []) or [],
        settings.get('auto_refresh', ['enabled']) or [],
        settings.get('auto_refresh_minutes', 5),
    )


@app.callback(
    Output('dump-alert-settings-store', 'data'),
    [
        Input('dump-alert-threshold-pct', 'value'),
        Input('dump-alert-min-gp-drop', 'value'),
        Input('dump-alert-min-avg-daily-volume', 'value'),
        Input('dump-alert-min-potential-profit', 'value'),
        Input('dump-alert-min-volume-potential', 'value'),
        Input('dump-alert-volume-power', 'value'),
        Input('dump-alert-max-qty-factor', 'value'),
        Input('dump-alert-metrics', 'value'),
        Input('dump-alert-windows', 'value'),
        Input('dump-alert-custom-windows', 'value'),
        Input('dump-alert-options', 'value'),
        Input('dump-alert-new-options', 'value'),
        Input('dump-alert-auto-refresh', 'value'),
        Input('dump-alert-auto-refresh-minutes', 'value'),
    ],
    prevent_initial_call=True
)
def persist_dump_alert_settings(threshold_pct, min_gp_drop, min_avg_daily_volume, min_potential_profit, min_volume_potential, volume_power, max_qty_factor, metric_values, window_values, custom_windows_text, alert_options, new_options, auto_refresh_value, auto_refresh_minutes):
    defaults = _default_dump_alert_settings()
    try:
        threshold_pct = float(threshold_pct) if threshold_pct is not None else defaults['threshold_pct']
    except (TypeError, ValueError):
        threshold_pct = defaults['threshold_pct']
    try:
        min_gp_drop = float(min_gp_drop) if min_gp_drop is not None else defaults['min_gp_drop']
    except (TypeError, ValueError):
        min_gp_drop = defaults['min_gp_drop']
    try:
        min_avg_daily_volume = float(min_avg_daily_volume) if min_avg_daily_volume is not None else defaults['min_avg_daily_volume']
    except (TypeError, ValueError):
        min_avg_daily_volume = defaults['min_avg_daily_volume']
    try:
        min_potential_profit = float(min_potential_profit) if min_potential_profit is not None else defaults['min_potential_profit']
    except (TypeError, ValueError):
        min_potential_profit = defaults['min_potential_profit']
    try:
        min_volume_potential = float(min_volume_potential) if min_volume_potential is not None else defaults['min_volume_potential']
    except (TypeError, ValueError):
        min_volume_potential = defaults['min_volume_potential']
    try:
        volume_power = float(volume_power) if volume_power is not None else defaults['volume_power']
    except (TypeError, ValueError):
        volume_power = defaults['volume_power']
    try:
        max_qty_factor = float(max_qty_factor) if max_qty_factor is not None else defaults['max_qty_factor']
    except (TypeError, ValueError):
        max_qty_factor = defaults['max_qty_factor']
    try:
        auto_refresh_minutes = int(float(auto_refresh_minutes)) if auto_refresh_minutes is not None else defaults['auto_refresh_minutes']
    except (TypeError, ValueError):
        auto_refresh_minutes = defaults['auto_refresh_minutes']

    return {
        'threshold_pct': max(0.1, threshold_pct),
        'min_gp_drop': max(0.0, min_gp_drop),
        'min_avg_daily_volume': max(0.0, min_avg_daily_volume),
        'min_potential_profit': max(0.0, min_potential_profit),
        'min_volume_potential': max(0.0, min_volume_potential),
        'volume_power': max(0.0, volume_power),
        'max_qty_factor': max(0.0, max_qty_factor),
        'metrics': _normalize_dump_alert_metrics(metric_values),
        'windows': _normalize_dump_alert_windows(window_values),
        'custom_windows_text': custom_windows_text or '',
        'alert_options': alert_options or [],
        'new_options': new_options or [],
        'auto_refresh': auto_refresh_value or [],
        'auto_refresh_minutes': min(1440, max(1, auto_refresh_minutes)),
    }


@app.callback(
    [Output('dump-alert-interval', 'disabled'), Output('dump-alert-interval', 'interval')],
    [Input('dump-alert-auto-refresh', 'value'), Input('dump-alert-auto-refresh-minutes', 'value')],
    prevent_initial_call=False
)
def toggle_dump_alert_auto_refresh(auto_refresh_value, auto_refresh_minutes):
    try:
        refresh_minutes = int(float(auto_refresh_minutes)) if auto_refresh_minutes is not None else 5
    except (TypeError, ValueError):
        refresh_minutes = 5
    refresh_minutes = min(1440, max(1, refresh_minutes))
    return 'enabled' not in (auto_refresh_value or []), refresh_minutes * 60 * 1000







## Remove the separate toggle_watchlist_cell callback and merge its logic into update_table_and_times below



# --- Tab content callback ---
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
    State('dump-alert-settings-store', 'data')
)
def render_tab_content(tab, dump_alert_settings_store):
    dump_alert_settings = _default_dump_alert_settings()
    if isinstance(dump_alert_settings_store, dict):
        dump_alert_settings.update(dump_alert_settings_store)

    if tab == 'analysis':
        # Render the original analysis layout
        return html.Div([
            html.H2('OSRS GE Forecast Analysis (Dash)'),
            html.Div([
                html.H4('1) Market Scope', className='filter-section-title'),
                html.Div([
                    html.Div([
                        html.Label('Item Search', style={'marginRight': '10px'}),
                        dcc.Input(id='item-name-search', type='text', placeholder='Type item name...', debounce=True, style={'width': '220px', 'marginRight': '30px'}),
                    ], className='filter-field'),
                    html.Div([
                        dcc.Checklist(
                            id='filter-by-watchlist',
                            options=[{'label': 'Watchlist Only', 'value': 'watchlist'}],
                            value=[],
                            style={'marginRight': '30px'}
                        ),
                    ], className='filter-field'),
                    html.Div([
                        html.Label('Risk Level', style={'marginLeft': '40px', 'marginRight': '10px'}),
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
                            searchable=False,
                            style={'width': '180px', 'display': 'inline-block', 'marginRight': '30px'}
                        ),
                    ], className='filter-field'),
                    html.Div([
                        html.Label('Status / Warnings', style={'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='status-filter',
                            options=['Normal', 'Volume imbalance', 'Possible manipulation', 'Abnormally high', 'Abnormally low', 'Spread unusually high'
                                     ,'Low price dump', 'High price spike', 'Low-Volume spike', 'High-Volume spike'
                                     ,'Low-Volume low', 'Low-Volume high', 'High-Volume low', 'High-Volume high'],  # Will be set dynamically

                            value=[],
                            multi=True,
                            clearable=True,
                            searchable=False,
                            style={'width': 'auto', 'minWidth': '250px', 'display': 'inline-block', 'marginRight': '30px'},
                            maxHeight=300,
                            optionHeight=36,
                            placeholder='Choose warnings...'
                        ),
                        dcc.Checklist(
                            id='status-filter-exclude',
                            options=[{'label': 'Exclude Chosen Statuses', 'value': 'exclude'}],
                            value=[],
                            style={'display': 'inline-block'}
                        ),
                    ], className='filter-field'),
                ], className='filter-row')
            ], className='filter-section-card'),
            html.Div(id='update-times', style={'marginBottom': '12px', 'fontSize': '16px'}),
            dcc.Interval(id='refresh-interval', interval=24*60*60*1000, n_intervals=0, disabled=True),
            html.Div([
                html.Div([
                    html.H4('2) Forecast Inputs', className='filter-section-title'),
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
                                {'label': 'Volume Percentile Median (Top 15%)', 'value': 'volume_percentile_median'},

                                # ---- RECOMMENDED DEFAULT ----
                                {'label': 'Auto (Horizon-Aware Mix)', 'value': 'mix'},
                            ],
                            value='wma',
                            clearable=False,
                            searchable=False,
                            style={'display': 'inline-block'}
                        ),], className='filter-row', style={'display': 'flex', 'width': 'auto', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label('Target Hold Time (min)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='forecast-sell-time', type='number', value=60, min=5, step=5, debounce=True, style={'width': '40px', 'marginRight': '30px'}),
                        html.Label('Min Price', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-price', type='number', value=0, debounce=True, style={'width': '70px', 'marginRight': '30px'}),
                        html.Label('Max Price', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='max-price', type='number', value=0, debounce=True, style={'width': '70px', 'marginRight': '30px'}),
                        html.Label('Min Daily Volume', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-avg-daily-volume', type='number', value=0, debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Avg Trade Recency (min)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='max-avg-trade-time', type='number', value=0, debounce=True, style={'width': '120px', 'marginRight': '30px'}),
                    ], className='filter-row', style={'display': 'flex', 'width': 'auto', 'marginBottom': '12px'}),
                    html.Div([
                        html.Label('Buy Price Source', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='buy-price-type',
                            options=[
                                {'label': 'Low Price', 'value': 'low'},
                                {'label': 'High Price', 'value': 'high'},
                                {'label': 'Avg Low Price (30min)', 'value': 'avg_low_30min'},
                            ],
                            value='low',
                            clearable=False,
                            searchable=False,
                            style={'display': 'inline-block', 'marginRight': '30px'}
                        ),
                        html.Label('Forecast Target Price', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='forecast-price-type',
                            options=[
                                {'label': 'Low Price', 'value': 'low'},
                                {'label': 'High Price', 'value': 'high'},
                                {'label': 'Average Price', 'value': 'avg'},
                            ],
                            value='avg',
                            clearable=False,
                            searchable=False,
                            style={'display': 'inline-block', 'marginRight': '30px'}
                        ),
                    ], className='filter-row', style={'display': 'flex', 'width': 'auto', 'marginBottom': '12px'}),
                    html.H4('3) Profit, Liquidity, and Risk Thresholds', className='filter-section-title'),
                    html.Div([
                        html.Label('Min Forecast Profit (gp)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-profit', type='number', value=filter_defaults['MIN_FORECAST_PROFIT'], debounce=True, style={'width': '70px', 'marginRight': '30px'}),
                        html.Label('Min Potential Profit (gp)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-potential-profit', type='number', value=filter_defaults['MIN_POTENTIAL_PROFIT'], debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Min Volume Potential (gp)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-volume', type='number', value=filter_defaults['MIN_VOLUME_POTENTIAL'], debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Min ROI (%)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='min-roi', type='number', value=filter_defaults['MIN_FORECAST_ROI'], debounce=True, style={'width': '60px', 'marginRight': '30px'}),
                        html.Label('History Window (hrs)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='forecast-hours', type='number', value=filter_defaults['FORECAST_HOURS'], debounce=True, style={'width': '60px', 'marginRight': '30px'}),
                        html.Label('Ignore Recent Data (min)', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='forecast-recency-minutes', type='number', value=filter_defaults['FORECAST_RECENCY_MINUTES'], min=0, debounce=True, style={'width': '80px', 'marginRight': '30px'}),
                        html.Label('Volume Weight', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='volume-power', type='number', value=1, debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                        html.Label('Amount Multiplier', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                        dcc.Input(id='max-qty-factor', type='number', value=1, min=0, step=0.1, debounce=True, style={'width': '90px', 'marginRight': '30px'}),
                    ], className='filter-row compact-threshold-row', style={'display': 'flex', 'width': 'auto', 'marginBottom': '16px'}),
                ], style={'display': 'block', 'flexWrap': 'wrap', 'gap': '6px', 'alignItems': 'center', 'marginBottom': '12px'}),
                html.H4('4) Ranking and Output', className='filter-section-title'),
                html.Div([
                    html.Label('Trend Direction', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
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
                        searchable=False,
                        style={'width': '80px', 'marginRight': '30px'}
                    ),
                    html.Label('Result Limit', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                    dcc.Input(id='top-n', type='number', value=10, min=1, debounce=True, style={'width': '60px', 'marginRight': '30px'}),
                    html.Label('Sort By', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
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
                        searchable=False,
                        style={'display': 'inline-block', 'marginRight': '30px'}
                    ),
                    html.Button('Apply Filters and Refresh Results', id='apply-filters', n_clicks=0, style={'height': '40px'}),
                ], className='filter-row compact-ranking-row', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '6px', 'alignItems': 'center', 'marginBottom': '12px'}),
                html.H4('5) Saved Presets', className='filter-section-title'),
                html.Div([
                    html.Label('Saved Filter:', style={'display': 'flex', 'align-items': 'center', 'line-height': 'normal', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='template-dropdown',
                        options=[{'label': name, 'value': name} for name in list_filter_templates()],
                        value=list_filter_templates()[0] if list_filter_templates() else None,
                        placeholder='Select template',
                        style={'width': '180px', 'marginRight': '10px'}
                    ),
                    dcc.Input(id='template-name', type='text', placeholder='New preset name', style={'marginRight': '10px'}),
                    html.Button('Save Preset', id='save-template', n_clicks=0, style={'marginRight': '6px'}),
                    html.Button('Load Preset', id='load-template', n_clicks=0, style={'marginRight': '6px'}),
                    html.Button('Delete Preset', id='delete-template', n_clicks=0, style={'marginRight': '6px'}),
                    html.Span(id='template-status', style={'marginLeft': '10px', 'color': '#888'}),
                ], className='filter-row', style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '8px'}),
                # Insert Add selected to watchlist button and store
                html.Button('Add Selected Rows to Watchlist', id='add-selected-to-watchlist-btn', n_clicks=0, style={'marginBottom': '10px', 'marginRight': '10px'}),
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
                            style_cell={'textAlign': 'center', 'fontSize': '11px', 'padding': '0px 6px', 'height': '24px', 'maxHeight': '20px', 'minHeight': '0px', 'lineHeight': '20px'},
                            style_header={'fontWeight': 'bold', 'fontSize': '11px', 'padding': '0px 6px', 'height': '24px', 'maxHeight': '20px', 'minHeight': '0px', 'lineHeight': '20px'},
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
                html.Div([
                    html.Button('Y Fit', id='chart-y-fit-btn', n_clicks=0, style=_dump_alert_chart_button_style()),
                    dcc.Input(id='chart-y-fit-pct', type='number', value=15, min=0, step=1, debounce=True, style={'width': '70px', 'marginLeft': '8px'}),
                ], style={'marginTop': '10px'}),
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
                style_cell={'textAlign': 'center', 'fontSize': '11px', 'padding': '0px 6px', 'height': '20px', 'maxHeight': '20px', 'minHeight': '0px', 'lineHeight': '20px'},
                style_header={'fontWeight': 'bold', 'fontSize': '11px', 'padding': '0px 6px', 'height': '20px', 'maxHeight': '20px', 'minHeight': '0px', 'lineHeight': '20px'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                    {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                ],
                row_selectable='single',
                selected_rows=[],
                editable=True,
            )
        ], className='tab-page')

    elif tab == 'dump_alert':
        return html.Div([
            html.H2('Dump Alert'),
            html.P('Flags items whose 5-minute price has fallen sharply from the peak seen inside your selected time window.'),
            html.Div([
                html.Div([
                    html.Label('Drop Threshold (%)'),
                    dcc.Input(id='dump-alert-threshold-pct', type='number', value=dump_alert_settings.get('threshold_pct', 5), min=0.1, step=0.5, style={'width': '120px'})
                ], className='filter-field'),
                html.Div([
                    html.Label('Min GP Drop'),
                    dcc.Input(id='dump-alert-min-gp-drop', type='number', value=dump_alert_settings.get('min_gp_drop', 250), min=0, step=50, style={'width': '120px'})
                ], className='filter-field'),
                html.Div([
                    html.Label('Min Avg Daily Volume'),
                    dcc.Input(id='dump-alert-min-avg-daily-volume', type='number', value=dump_alert_settings.get('min_avg_daily_volume', 0), min=0, step=1, style={'width': '150px'})
                ], className='filter-field'),
                html.Div([
                    html.Label('Min Potential Profit (gp)'),
                    dcc.Input(id='dump-alert-min-potential-profit', type='number', value=dump_alert_settings.get('min_potential_profit', 0), min=0, step=1, style={'width': '150px'})
                ], className='filter-field'),
                html.Div([
                    html.Label('Min Volume Potential'),
                    dcc.Input(id='dump-alert-min-volume-potential', type='number', value=dump_alert_settings.get('min_volume_potential', 0), min=0, step=0.1, style={'width': '150px'})
                ], className='filter-field'),
                html.Div([
                    html.Label('Volume Weight'),
                    dcc.Input(id='dump-alert-volume-power', type='number', value=dump_alert_settings.get('volume_power', 1), min=0, step=0.1, style={'width': '120px'})
                ], className='filter-field'),
                html.Div([
                    html.Label('Amount Multiplier'),
                    dcc.Input(id='dump-alert-max-qty-factor', type='number', value=dump_alert_settings.get('max_qty_factor', 1), min=0, step=0.1, style={'width': '130px'})
                ], className='filter-field'),
                html.Button('Refresh Now', id='dump-alert-refresh-btn', n_clicks=0, style={'height': '38px'}),
            ], className='filter-row', style={'marginBottom': '16px'}),
            html.Div([
                html.Div([
                    html.Label('Drop Metric'),
                    dcc.RadioItems(
                        id='dump-alert-metrics',
                        options=[
                            {'label': 'Avg', 'value': 'avg_price'},
                            {'label': 'Avg of Low Prices', 'value': 'avg_low_price'},
                            {'label': 'Avg of High Prices', 'value': 'avg_high_price'},
                            {'label': 'Median Price', 'value': 'median_price'},
                        ],
                        value=_normalize_dump_alert_metrics(dump_alert_settings.get('metrics'))[0],
                        inline=True,
                    ),
                ], className='filter-field', style={'alignItems': 'flex-start'}),
            ], className='filter-row', style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label('Tracked Windows'),
                    dcc.Checklist(
                        id='dump-alert-windows',
                        options=[
                            {'label': '15m', 'value': 15},
                            {'label': '30m', 'value': 30},
                            {'label': '60m', 'value': 60},
                            {'label': '120m', 'value': 120},
                            {'label': '240m', 'value': 240},
                            {'label': '360m (6h)', 'value': 360},
                            {'label': '720m (12h)', 'value': 720},
                            {'label': '1440m (1d)', 'value': 1440},
                            {'label': '2880m (2d)', 'value': 2880},
                            {'label': '10080m (7d)', 'value': 10080},
                        ],
                        value=_normalize_dump_alert_windows(dump_alert_settings.get('windows')),
                        inline=True,
                    ),
                    dcc.Input(
                        id='dump-alert-custom-windows',
                        type='text',
                        value=dump_alert_settings.get('custom_windows_text', ''),
                        placeholder='Custom windows (e.g. 360, 12h, 2d)',
                        debounce=True,
                        style={'width': '280px', 'marginTop': '8px'}
                    ),
                ], className='filter-field', style={'alignItems': 'flex-start'}),
                html.Div([
                    html.Label('Alert Options'),
                    dcc.Checklist(
                        id='dump-alert-options',
                        options=[
                            {'label': 'Play sound for new alerts', 'value': 'sound'},
                            {'label': 'Highlight new alerts', 'value': 'highlight'},
                        ],
                        value=dump_alert_settings.get('alert_options', ['sound', 'highlight']) or [],
                        inline=True,
                    ),
                ], className='filter-field', style={'alignItems': 'flex-start'}),
            ], className='filter-row', style={'marginBottom': '16px'}),
            html.Div([
                html.Div([
                    html.Label('New Alert Options'),
                    dcc.Checklist(
                        id='dump-alert-new-options',
                        options=[
                            {'label': 'Show only new since last refresh', 'value': 'show_new_only'},
                            {'label': 'Keep new markers until manual refresh', 'value': 'sticky_new'},
                        ],
                        value=dump_alert_settings.get('new_options', []) or [],
                        inline=True,
                    ),
                ], className='filter-field', style={'alignItems': 'flex-start'}),
                html.Div([
                    html.Label('Refresh Mode'),
                    dcc.Checklist(
                        id='dump-alert-auto-refresh',
                        options=[
                            {'label': 'Enable auto-refresh', 'value': 'enabled'},
                        ],
                        value=dump_alert_settings.get('auto_refresh', ['enabled']) or [],
                        inline=True,
                    ),
                    html.Div([
                        html.Label('Every (min)', style={'marginRight': '8px'}),
                        dcc.Input(
                            id='dump-alert-auto-refresh-minutes',
                            type='number',
                            value=dump_alert_settings.get('auto_refresh_minutes', 5),
                            min=1,
                            max=1440,
                            step=1,
                            debounce=True,
                            style={'width': '80px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '8px'}),
                ], className='filter-field', style={'alignItems': 'flex-start'}),
            ], className='filter-row', style={'marginBottom': '14px'}),
            dcc.Interval(id='dump-alert-interval', interval=5 * 60 * 1000, n_intervals=0),
            html.Div(id='dump-alert-sound-dummy', style={'display': 'none'}),
            html.Div(id='dump-alert-formula-help', style={'marginBottom': '8px', 'fontSize': '13px', 'opacity': 0.85}),
            html.Div(id='dump-alert-status', style={'marginBottom': '6px', 'fontWeight': 'bold'}),
            html.Div(id='dump-alert-new-summary', style={'marginBottom': '8px', 'fontSize': '14px', 'color': '#8a5a00'}),
            html.Div(id='dump-alert-last-refresh', style={'marginBottom': '14px', 'fontSize': '14px', 'opacity': 0.85}),
            dash_table.DataTable(
                id='dump-alert-table',
                columns=[
                    {'name': 'Alert Key', 'id': 'alert_key', 'type': 'text'},
                    {'name': 'Is New', 'id': 'is_new', 'type': 'numeric'},
                    {'name': 'Item ID', 'id': 'item_id', 'type': 'text'},
                    {'name': 'Item Name', 'id': 'item_name', 'type': 'text'},
                    {'name': 'Metric', 'id': 'metric', 'type': 'text'},
                    {'name': 'Drop %', 'id': 'drop_pct', 'type': 'numeric'},
                    {'name': 'GP Drop', 'id': 'gp_drop', 'type': 'numeric'},
                    {'name': 'Reference Price', 'id': 'reference_price', 'type': 'numeric'},
                    {'name': 'Current Price', 'id': 'current_price', 'type': 'numeric'},
                    {'name': 'Window (m)', 'id': 'window_minutes', 'type': 'numeric'},
                    {'name': 'Reference Seen', 'id': 'reference_seen', 'type': 'text'},
                    {'name': 'Latest Seen', 'id': 'latest_seen', 'type': 'text'},
                    {'name': 'Latest Low', 'id': 'latest_low', 'type': 'numeric'},
                    {'name': 'Latest High', 'id': 'latest_high', 'type': 'numeric'},
                    {'name': 'Low Vol (Window)', 'id': 'window_low_volume', 'type': 'numeric'},
                    {'name': 'High Vol (Window)', 'id': 'window_high_volume', 'type': 'numeric'},
                    {'name': 'Avg Daily Volume', 'id': 'avg_daily_volume', 'type': 'numeric'},
                    {'name': 'Profit/Item', 'id': 'profit_per_item', 'type': 'numeric'},
                    {'name': 'Buy Qty', 'id': 'buy_qty', 'type': 'numeric'},
                    {'name': 'Potential Profit', 'id': 'potential_profit', 'type': 'numeric'},
                    {'name': 'Volume Potential', 'id': 'volume_potential', 'type': 'numeric'},
                    {'name': 'Buy Limit', 'id': 'buy_limit', 'type': 'numeric'},
                ],
                data=[],
                hidden_columns=['alert_key', 'is_new'],
                page_size=25,
                sort_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'fontSize': '11px', 'padding': '0px 6px', 'height': '20px', 'maxHeight': '20px', 'minHeight': '0px', 'lineHeight': '20px'},
                style_header={'fontWeight': 'bold', 'fontSize': '11px', 'padding': '0px 6px', 'height': '20px', 'maxHeight': '20px', 'minHeight': '0px', 'lineHeight': '20px'},
                style_cell_conditional=[
                    {'if': {'column_id': 'item_name'}, 'textAlign': 'left', 'minWidth': '180px', 'width': '180px', 'maxWidth': '240px'},
                    {'if': {'column_id': 'metric'}, 'minWidth': '110px', 'width': '110px', 'maxWidth': '140px'},
                ],
                style_data_conditional=_dump_alert_base_table_styles(),
            ),
            html.Div([
                html.Button('Y Fit', id='dump-alert-chart-y-fit-btn', n_clicks=0, style=_dump_alert_chart_button_style()),
                dcc.Input(id='dump-alert-chart-y-fit-pct', type='number', value=15, min=0, step=1, debounce=True, style={'width': '70px', 'marginLeft': '8px'}),
            ], style={'marginTop': '10px'}),
            html.Div(id='dump-alert-timeseries-container', style={'marginTop': '30px'}),
        ], className='tab-page')


    # NOTE: watchlist callback is defined at module scope below. Keep layout-only logic here.




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
    State('forecast-recency-minutes', 'value'),
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
def update_table_and_times(n_clicks, item_name_search_input, filter_by_watchlist_input, item_name_search, forecast_strategy, min_profit, min_potential_profit, min_volume, min_roi, forecast_hours, forecast_recency_minutes, volume_power, max_qty_factor, top_n, sort_attribute, trend_filter, forecast_sell_time, min_price, max_price, min_avg_daily_volume, max_avg_trade_time, table_data, template_selected, status_filter, risk_filter, buy_price_type, forecast_price_type):
    if forecast_sell_time is None:
        forecast_sell_time = 0
    
    t_debug_start = time.time()
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    # Forecast table cell-click watchlist logic is handled by dedicated watchlist callbacks.
    # Ensure filter_by_watchlist_input is always a list
    if filter_by_watchlist_input is None:
        filter_by_watchlist_input = []
    # import time module already at top, avoid shadowing
    t0 = time.time()
    t1 = time.time()
    forecast_start = time.time()


    # If item-name-search or watchlist filter is triggered, apply those filters
    if 'item-name-search' in triggered or 'filter-by-watchlist' in triggered or (item_name_search and isinstance(item_name_search, str) and item_name_search.strip()) or (filter_by_watchlist_input and 'watchlist' in filter_by_watchlist_input):
        
        search_term = (item_name_search or '').strip().lower()
        
        # if search_term == '' and not watchlist_filter_active:
        #     print('[DEBUG] No item-name-search or watchlist filter active, skipping forecast recalculation.')
        #     from dash import no_update
        #     return table_data, no_update, no_update, no_update
        
        t2 = time.time()
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
                if fetch_and_save_5m_data_safe('item-name/watchlist'):
                    pass
            except Exception as e:
                print(f"Error fetching 5m data: {e}")
        t2b = time.time()
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
                pass

        import os, json
        
        with open(os.path.join(APP_DIR, 'Data', 'mapping_cache.json'), 'r', encoding='utf-8') as f:
            mapping_cache = json.load(f)
        
        # --- Watchlist filter logic ---
        watchlist_ids = set()
        watchlist_filter_active = filter_by_watchlist_input and 'watchlist' in filter_by_watchlist_input
        if watchlist_filter_active:
            watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
            try:
                with open(watchlist_path, 'r', encoding='utf-8') as f:
                    watchlist = json.load(f)
                watchlist_ids = set(str(w['item_id']) for w in watchlist if 'item_id' in w)
            except Exception as e:
                watchlist_ids = set()
        if watchlist_filter_active:
            if not watchlist_ids:
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
            else:
                pass

        
        try:
            all_items, forecasted_prices, latest_prices = analyze_forecast_gui({
                'MIN_FORECAST_PROFIT': 0,
                'MIN_POTENTIAL_PROFIT': 0,
                'MIN_VOLUME_POTENTIAL': 0,
                'MIN_FORECAST_ROI': 0,
                'FORECAST_HOURS': forecast_hours,
                'FORECAST_RECENCY_MINUTES': forecast_recency_minutes,
                'STATUS_FILTER': status_filter_ctx,
                'STATUS_FILTER_MODE': status_filter_mode,
                'STATUS_EXCLUDE': status_exclude_ctx,
            }, forecast_sell_time, forecast_strategy, name_filter=name_filter, volume_power = volume_power, max_qty_factor=max_qty_factor, forecast_hours=forecast_hours, buy_price_type=buy_price_type, forecast_price_type=forecast_price_type)
        except Exception as e:
            all_items, forecasted_prices = [], []
        forecast_end = time.time()
        forecast_duration = forecast_end - forecast_start
        
        t4 = time.time()
        data = all_items
    elif 'apply-filters' in triggered:
        t2 = time.time()
        # Always fetch new 5m data on update
        try:
            if fetch_and_save_5m_data_safe('apply-filters'):
                pass
        except Exception as e:
            print(f"Error fetching 5m data: {e}")
        t2b = time.time()

        name_filter = []

        import os, json

        with open(os.path.join(APP_DIR, 'Data', 'mapping_cache.json'), 'r', encoding='utf-8') as f:
            mapping_cache = json.load(f)
        if item_name_search and isinstance(item_name_search, str) and item_name_search.strip():
            search_term = item_name_search.strip().lower()
            name_filter = [mapping_cache['data'][r]['name'] for r in mapping_cache['data'] if not search_term or search_term in r['name'].lower()]



        # Gather status filter from GUI dropdown parameter
        status_filter_ctx = []
        status_filter_mode = 'include'
        status_exclude_ctx = []
        
        # Use status filter from State parameter
        if status_filter and isinstance(status_filter, list) and len(status_filter) > 0:
            status_filter_ctx = status_filter
            # Check if exclude mode is enabled (you'd need to add this as a State too if needed)
        else:
            pass


        # Send no status filter to backend; apply UI status filter on the GUI-rendered rows
        all_items, forecasted_prices_raw, latest_prices = analyze_forecast_gui({
            'MIN_FORECAST_PROFIT': 0,
            'MIN_POTENTIAL_PROFIT': 0,
            'MIN_VOLUME_POTENTIAL': 0,
            'MIN_FORECAST_ROI': 0,
            'FORECAST_HOURS': forecast_hours,
            'FORECAST_RECENCY_MINUTES': forecast_recency_minutes,
            'STATUS_FILTER': [],
            'STATUS_FILTER_MODE': 'include',
            'STATUS_EXCLUDE': [],
        }, forecast_sell_time, forecast_strategy, name_filter=name_filter, max_avg_trade_time=max_avg_trade_time, volume_power = volume_power, max_qty_factor=max_qty_factor, forecast_hours=forecast_hours, buy_price_type=buy_price_type, forecast_price_type=forecast_price_type)
        forecast_end = time.time()
        forecast_duration = forecast_end - forecast_start
        t3 = time.time()
        # Build forecasted_prices map by id
        forecasted_prices_map = {entry['id']: entry for entry in forecasted_prices_raw if 'id' in entry}
        # If item name search is provided, filter by name and disregard all other filters
        if item_name_search and isinstance(item_name_search, str) and item_name_search.strip():
            pass
        else:
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
                    except Exception as e:
                            pass
            forecasted_prices = [forecasted_prices_map[item['id']] for item in filtered if item['id'] in forecasted_prices_map]
            # Validate forecast_price structure
            for fp in forecasted_prices:
                if not isinstance(fp.get('forecast_price'), list):
                    fp['forecast_price'] = []
            # Defer top_n slicing until after UI status filtering on rendered rows
            data = filtered
            t6 = time.time()


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
                    pass

    if not isinstance(data, list):
        # Defensive: if data is not a list, return empty table and log error
        print('[ERROR] Data is not a list in update_table_and_times, returning empty table.')
        columns = [
            {'name': 'ID', 'id': 'id'},
            {'name': 'Name', 'id': 'name'},
            {'name': 'Low Price', 'id': 'lowP'},
            {'name': 'High Price', 'id': 'highP'},
            {'name': 'Low Recency', 'id': 'lowTime'},
            {'name': 'High Recency', 'id': 'highTime'},
            {'name': 'Spread', 'id': 'rel_spread'},
            {'name': 'Forecast Price', 'id': 'forecast_price'},
            {'name': 'Profit', 'id': 'profit'},
            {'name': 'ROI (%)', 'id': 'roi'},
            {'name': 'Volume', 'id': 'avg_daily_volume'},
            {'name': 'Buy Limit', 'id': 'buy_limit'},
            {'name': 'Suggested Qty', 'id': 'max_qty'},
            {'name': 'Potential Profit', 'id': 'potential_profit'},
            {'name': 'Volume Potential', 'id': 'volume_potential'},
            {'name': 'Trend', 'id': 'trend_direction'},
            {'name': 'Risk Level', 'id': 'risk_level'},
            {'name': 'Status', 'id': 'status'},
            {'name': 'Recent Low Vol (%)', 'id': 'lowVol_recent'},
            {'name': 'Recent High Vol (%)', 'id': 'highVol_recent'},
            # {'name': 'Buy Price', 'id': 'buy_price', 'hideable': False, 'hidden': True},
            # {'name': 'Sell Price', 'id': 'sell_price', 'hideable': False, 'hidden': True},
        ]
        update_text = 'No data available.'
        forecasted_prices_dict = {}
        return [], columns, update_text, forecasted_prices_dict
    columns = [
        {'name': 'Item ID', 'id': 'id'},
        {'name': 'Item Name', 'id': 'name'},
        {'name': 'Low Price', 'id': 'lowP'},
        {'name': 'High Price', 'id': 'highP'},
        {'name': 'Low Updated', 'id': 'lowTime'},
        {'name': 'High Updated', 'id': 'highTime'},
        {'name': 'Spread (%)', 'id': 'rel_spread'},
        {'name': 'Forecast Price', 'id': 'forecast_price'},
        {'name': 'Forecast Profit', 'id': 'profit'},
        {'name': 'ROI (%)', 'id': 'roi'},
        {'name': 'Avg Daily Volume', 'id': 'avg_daily_volume'},
        {'name': 'Buy Limit', 'id': 'buy_limit'},
        {'name': 'Suggested Qty', 'id': 'max_qty'},
        {'name': 'Potential Profit', 'id': 'potential_profit'},
        {'name': 'Volume Potential', 'id': 'volume_potential'},
        {'name': 'Trend', 'id': 'trend_direction'},
        {'name': 'Risk Level', 'id': 'risk_level'},
        {'name': 'Status', 'id': 'status'},
        {'name': 'Recent Low Vol (%)', 'id': 'lowVol_recent'},
        {'name': 'Recent High Vol (%)', 'id': 'highVol_recent'},
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
            pass

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
            # For percentile strategy, highP is the top-15% forecast ceiling — preserve it
            if forecast_strategy == 'volume_percentile_median':
                new_row['highP'] = format_number(row.get('highP', 0), 0)
            else:
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
        status = str(row.get('status', '') or '').strip()
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
                pass

            if manipulation and warning_msgs:
                warning_text = '; '.join(warning_msgs)
                status = f"{status} | ⚠ Possible manipulation! {warning_text}" if status else f"⚠ Possible manipulation! {warning_text}"

            if status == '':
                status = 'Normal'
        except Exception as e:
            status = ''
        new_row['status'] = status
        return new_row
    try:
        if data:
            data = [clean_row(row) for row in data]
            # Apply status filter AFTER clean_row so we filter on GUI-rendered status
            status_filter_values = status_filter if isinstance(status_filter, list) else []
            if status_filter_values:
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
        pass
    # print('[DEBUG] Final data sample:', data[:3])
    # print('[DEBUG] Final columns:', columns)
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
        update_text = f"Refresh: {dt_cls.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
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
    # Keep RuneLite API cache in sync with the latest rendered forecast rows.
    global _last_forecast_rows
    _last_forecast_rows = data if isinstance(data, list) else []

    # Return only the four expected outputs - let DataTable manage selected_rows
    return data, columns, update_text, forecasted_prices_dict

@app.callback(
    Output('timeseries-container', 'children'),
    [
        Input('forecast-table', 'selected_rows'),
        Input('forecast-table', 'selected_row_ids'),
        Input('forecast-table', 'active_cell'),
        Input('forecast-table', 'derived_viewport_data'),
        Input('forecast-table', 'data'),
        Input('forecasted-prices-store', 'data'),
        Input('forecast-sell-time', 'value'),
        Input('chart-y-fit-btn', 'n_clicks'),
        Input('chart-y-fit-pct', 'value'),
        Input('dark-mode-store', 'data')
    ],
    prevent_initial_call=False
)
def show_timeseries(selected_rows, selected_row_ids, active_cell, viewport_data, table_data, forecasted_prices, forecast_sell_time=0, y_fit_clicks=0, y_fit_pct=15, dark_mode_data=None):
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
    print("=" * 80, flush=True)
    
    if forecast_sell_time is None:
        forecast_sell_time = 0
    y_fit_enabled = bool(y_fit_clicks and y_fit_clicks > 0)
    y_fit_revision = int(y_fit_clicks or 0)
    try:
        y_fit_pct = float(y_fit_pct) if y_fit_pct is not None else 15.0
    except (TypeError, ValueError):
        y_fit_pct = 15.0
    y_fit_pct = max(0.0, y_fit_pct)

    graph_config = {
        'scrollZoom': True,
        'doubleClick': 'reset',
        'modeBarButtonsToRemove': ['select2d', 'lasso2d']
    }

    def apply_default_window_and_zoom_controls(fig, time_values, default_window_days=0):
        if not time_values:
            return

        parsed_times = []
        for time_val in time_values:
            try:
                parsed = pandas_mod.to_datetime(time_val)
            except Exception:
                continue
            if pandas_mod.isna(parsed):
                continue
            if hasattr(parsed, 'to_pydatetime'):
                parsed = parsed.to_pydatetime()
            parsed_times.append(parsed)

        if not parsed_times:
            return

        range_start = min(parsed_times)
        range_end = max(parsed_times)

        if default_window_days > 0:
            requested_start = range_end - timedelta(days=default_window_days)
            if requested_start > range_start:
                range_start = requested_start

        fig.update_xaxes(
            range=[range_start.strftime('%Y-%m-%d %H:%M:%S'), range_end.strftime('%Y-%m-%d %H:%M:%S')],
            rangeslider=dict(visible=True),
            rangeselector=dict(
                x=0,
                y=1.08,
                xanchor='left',
                yanchor='top',
                bgcolor=_dump_alert_chart_control_theme()['bgcolor'],
                activecolor=_dump_alert_chart_control_theme()['activecolor'],
                bordercolor=_dump_alert_chart_control_theme()['bordercolor'],
                borderwidth=_dump_alert_chart_control_theme()['borderwidth'],
                buttons=[
                    dict(count=1, label='1d', step='day', stepmode='backward'),
                    dict(count=7, label='7d', step='day', stepmode='backward'),
                    dict(count=30, label='30d', step='day', stepmode='backward'),
                    dict(step='all', label='All')
                ]
            )
        )

        # Plotly sets y axes to fixedrange when x rangeslider is visible unless we override.
        fig.update_yaxes(fixedrange=False)
        fig.update_layout(dragmode='zoom')

    def compute_robust_axis_range(*series, lower_q=0.01, upper_q=0.99, pad_ratio=0.20):
        values = []
        for seq in series:
            if not seq:
                continue
            for v in seq:
                if isinstance(v, (int, float)) and not math.isnan(v) and math.isfinite(v):
                    values.append(float(v))

        if len(values) < 2:
            return None

        arr = np.array(values, dtype=float)
        lo = float(np.percentile(arr, lower_q * 100.0))
        hi = float(np.percentile(arr, upper_q * 100.0))

        if hi <= lo:
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            if hi <= lo:
                span = max(abs(hi) * 0.05, 1.0)
                lo -= span
                hi += span

        span = hi - lo
        lo -= span * pad_ratio
        hi += span * pad_ratio

        if min(values) >= 0:
            lo = max(0.0, lo)

        if hi <= lo:
            return None

        return [lo, hi]

    # Resolve selected item robustly across pagination/sorting.
    row = None
    resolved_item_id = None

    # Clicks are page-local; map through viewport rows to get the real item id.
    if isinstance(active_cell, dict) and active_cell.get('row') is not None and isinstance(viewport_data, list):
        viewport_row = active_cell.get('row')
        if isinstance(viewport_row, int) and 0 <= viewport_row < len(viewport_data):
            resolved_item_id = str(viewport_data[viewport_row].get('id'))

    # selected_row_ids are stable across pages if available.
    if resolved_item_id is None and selected_row_ids and len(selected_row_ids) > 0:
        resolved_item_id = str(selected_row_ids[0])

    # Fallback to legacy selected_rows index behavior.
    if resolved_item_id is None and selected_rows and len(selected_rows) > 0:
        row = selected_rows[0]
    elif resolved_item_id is not None and table_data and isinstance(table_data, list):
        for idx, candidate in enumerate(table_data):
            if str(candidate.get('id')) == resolved_item_id:
                row = idx
                break

    if row is None and table_data and len(table_data) > 0:
        row = 0  # Default to first row

    if row is not None and table_data and isinstance(table_data, list) and 0 <= row < len(table_data):
        item_id = table_data[row]['id']

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
                return result
            except Exception as e:
                return None

        line_low_price = parse_table_price(table_data[row].get('lowP'))
        line_high_price = parse_table_price(table_data[row].get('highP'))

        if line_low_price is None:
            line_low_price = parse_table_price(table_data[row].get('buy_price'))
        if line_high_price is None:
            line_high_price = parse_table_price(table_data[row].get('sell_price'))
        if line_high_price is None:
            line_high_price = parse_table_price(table_data[row].get('forecast_price'))

        t_parse_end = time.perf_counter()
        print(f"[PERF] Price parsing: {(t_parse_end - t_callback_start):.3f}s", flush=True)
        
        # --- 1h chart (existing) ---
        t_data_load_start = time.perf_counter()
        timestamps, avgLowPrice, avgHighPrice, lowVol, highVol = get_item_timeseries(item_id, hours_lookback=None)
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
            # Keep shared UI state stable, then control persistence per-axis below.
            uirevision='timeseries-1h-layout',
            yaxis=dict(
                title='Price',
                showgrid=True,
                zeroline=True,
                side='left',
                # Include y-fit revision so Y Fit clicks apply immediately on the active chart.
                uirevision=f'timeseries-1h-y-{item_id}-fit-{y_fit_revision}',
            ),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=False,
                uirevision=f'timeseries-1h-y2-{item_id}',
            ),
            xaxis=dict(
                # Preserve selected x-window buttons/range across item changes.
                uirevision='timeseries-1h-x-window'
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_dark' if dark_mode else 'plotly'
        )
        # Add forecasted_prices to the 1h chart if available
        forecasted = None
        forecast_y = []
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

        apply_default_window_and_zoom_controls(fig_1h, timestamps_shifted, 0)
        if y_fit_enabled:
            price_range_1h = _dump_alert_y_fit_range_from_lines(final_low, final_high, y_fit_pct)
            if price_range_1h is not None:
                fig_1h.update_yaxes(range=price_range_1h, fixedrange=False)

        volume_range_1h = compute_robust_axis_range(lowVol, highVol, lower_q=0.01, upper_q=0.99, pad_ratio=0.22)
        if volume_range_1h is not None:
            fig_1h.update_layout(yaxis2=dict(range=volume_range_1h, title='Volume', overlaying='y', side='right', showgrid=False, zeroline=False))
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
                    t_support_end = time.perf_counter()
                    print(f"[PERF] Support/resistance calculation (failed): {(t_support_end - t_support_start):.3f}s", flush=True)
                fig_5m.update_layout(
                    title=f"Price & Volume History (5m) for {table_data[row].get('name', item_id)}",
                    hovermode='closest',
                    # Keep shared UI state stable, then control persistence per-axis below.
                    uirevision='timeseries-5m-layout',
                    yaxis=dict(
                        title='Price',
                        showgrid=True,
                        zeroline=True,
                        side='left',
                        # Include y-fit revision so Y Fit clicks apply immediately on the active chart.
                        uirevision=f'timeseries-5m-y-{item_id}-fit-{y_fit_revision}',
                    ),
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        zeroline=False,
                        uirevision=f'timeseries-5m-y2-{item_id}',
                    ),
                    xaxis=dict(
                        # Preserve selected x-window buttons/range across item changes.
                        uirevision='timeseries-5m-x-window'
                    ),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    template='plotly_dark' if dark_mode else 'plotly'
                )
                apply_default_window_and_zoom_controls(fig_5m, ts_5m, 0)
                if y_fit_enabled:
                    price_range_5m = _dump_alert_y_fit_range_from_lines(final_low_5m, final_high_5m, y_fit_pct)
                    if price_range_5m is not None:
                        fig_5m.update_yaxes(range=price_range_5m, fixedrange=False)

                support_vals = list(price_df_5m['support']) if 'price_df_5m' in locals() and 'support' in price_df_5m else []
                resistance_vals = list(price_df_5m['resistance']) if 'price_df_5m' in locals() and 'resistance' in price_df_5m else []
                volume_range_5m = compute_robust_axis_range(lowVol_5m, highVol_5m, lower_q=0.01, upper_q=0.99, pad_ratio=0.22)
                if volume_range_5m is not None:
                    fig_5m.update_layout(yaxis2=dict(range=volume_range_5m, title='Volume', overlaying='y', side='right', showgrid=False, zeroline=False))
                # Return both charts stacked vertically
                t_callback_end = time.perf_counter()
                elapsed = t_callback_end - t_callback_start
                t_data_load_dur = t_fig1h_start - t_data_load_start
                t_fig1h_dur = t_fig1h_end - t_fig1h_start
                t_5m_all_dur = t_callback_end - t_5m_load_start
                print(f"[PERF] Breakdown: data_load={t_data_load_dur:.3f}s, fig1h={t_fig1h_dur:.3f}s, 5m_all={t_5m_all_dur:.3f}s", flush=True)
                print(f"[PERF] show_timeseries callback (both charts) in {elapsed:.3f}s", flush=True)
                return html.Div([
                    dcc.Graph(figure=fig_1h, config=graph_config),
                    html.Hr(),
                    dcc.Graph(figure=fig_5m, config=graph_config)
                ])
            else:
                # No 5m data, just show 1h chart
                t_callback_end = time.perf_counter()
                elapsed = t_callback_end - t_callback_start
                print(f"[PERF] Breakdown: data_load={(t_fig1h_start - t_data_load_start):.3f}s, fig1h={(t_fig1h_end - t_fig1h_start):.3f}s, 5m_load=0s, 5m_construct=0s", flush=True)
                print(f"[PERF] show_timeseries callback (1h only) in {elapsed:.3f}s", flush=True)
                return html.Div([
                    dcc.Graph(figure=fig_1h, config=graph_config),
                    html.Div('No 5m data available for this item.', style={'color': 'gray', 'marginTop': '10px'})
                ])
        except Exception as e:
            if 't_fig1h_end' not in locals():
                t_fig1h_end = time.perf_counter()
            t_callback_end = time.perf_counter()
            elapsed = t_callback_end - t_callback_start
            t_data_load_dur = (t_fig1h_start - t_data_load_start) if 't_fig1h_start' in locals() else 0
            t_fig1h_dur = (t_fig1h_end - t_fig1h_start) if 't_fig1h_start' in locals() else 0
            print(f"[PERF] Breakdown: data_load={t_data_load_dur:.3f}s, fig1h={t_fig1h_dur:.3f}s, 5m_error", flush=True)
            print(f"[PERF] show_timeseries callback (error) in {elapsed:.3f}s", flush=True)
            return html.Div([
                dcc.Graph(figure=fig_1h, config=graph_config),
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
        Output('forecast-recency-minutes', 'value'),
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
    State('forecast-recency-minutes', 'value'),
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
    min_potential_profit, min_volume, min_roi, forecast_hours, forecast_recency_minutes, volume_power, max_qty_factor, trend_filter,
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
            'min_potential_profit', 'min_volume', 'min_roi', 'forecast_hours', 'forecast_recency_minutes', 'volume_power', 'max_qty_factor', 'trend_filter',
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
            'forecast_recency_minutes': forecast_recency_minutes,
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
        return [status, options, template_name] + [dash.no_update]*22
    # Load template (auto-include all filter fields)
    elif trigger == 'load-template':
        if not load_clicks or not selected_template:
            raise PreventUpdate
        template = load_filter_template(selected_template)
        if not template:
            return [f"Template '{selected_template}' not found.", options, selected_template] + [dash.no_update]*22
        # List of all filter output order (must match callback Outputs)
        output_order = [
            'forecast_strategy', 'forecast_sell_time', 'min_price', 'max_price', 'min_avg_daily_volume', 'max_avg_trade_time',  'min_profit',
            'min_potential_profit', 'min_volume', 'min_roi', 'forecast_hours', 'forecast_recency_minutes', 'volume_power', 'max_qty_factor', 'trend_filter',
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
            'forecast_recency_minutes': 0,
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
        # Ensure result is exactly the right length (3 fixed + 22 filter values)
        while len(result) < 25:
            result.append(dash.no_update)
        return result[:25]
    # Delete template
    elif trigger == 'delete-template':
        if not delete_clicks or not selected_template:
            raise PreventUpdate
        success = delete_filter_template(selected_template)
        options = [{'label': name, 'value': name} for name in list_filter_templates()]
        status = f"Template '{selected_template}' deleted." if success else f"Error deleting template '{selected_template}'."
        return [status, options, None] + [dash.no_update]*22
    raise PreventUpdate

# --- Watchlist Add/Remove Callback ---
@app.callback(
    Output('add-selected-status', 'children'),
    [Input('add-selected-to-watchlist-btn', 'n_clicks')],
    [State('forecast-table', 'selected_rows'), State('forecast-table', 'data')]
)
def add_selected_to_watchlist(n_clicks, selected_rows, table_data):
    if n_clicks and n_clicks > 0:
        if not selected_rows or not table_data:
            return 'No row selected.'
        row_idx = selected_rows[0] if isinstance(selected_rows, list) and selected_rows else None
        if row_idx is None or row_idx < 0 or row_idx >= len(table_data):
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
        existing_item = next((w for w in watchlist if str(w.get('item_id')) == str(item_id)), None)
        if existing_item:
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
            return f'Added {item_name} to watchlist.'
        except Exception as e:
            print(f'[ERROR] Failed to save watchlist: {e}')
            return f'Error saving watchlist: {e}'
    return dash.no_update


@app.callback(
    [
        Output('dump-alert-table', 'data'),
        Output('dump-alert-table', 'style_data_conditional'),
        Output('dump-alert-formula-help', 'children'),
        Output('dump-alert-status', 'children'),
        Output('dump-alert-new-summary', 'children'),
        Output('dump-alert-last-refresh', 'children'),
        Output('dump-alert-refresh-state-store', 'data'),
        Output('dump-alert-sound-signal', 'data'),
    ],
    [
        Input('dump-alert-refresh-btn', 'n_clicks'),
        Input('dump-alert-interval', 'n_intervals'),
        Input('dump-alert-threshold-pct', 'value'),
        Input('dump-alert-min-gp-drop', 'value'),
        Input('dump-alert-min-avg-daily-volume', 'value'),
        Input('dump-alert-min-potential-profit', 'value'),
        Input('dump-alert-min-volume-potential', 'value'),
        Input('dump-alert-volume-power', 'value'),
        Input('dump-alert-max-qty-factor', 'value'),
        Input('dump-alert-metrics', 'value'),
        Input('dump-alert-windows', 'value'),
        Input('dump-alert-custom-windows', 'value'),
        Input('dump-alert-options', 'value'),
        Input('dump-alert-new-options', 'value'),
        Input('dump-alert-auto-refresh', 'value'),
        Input('dump-alert-auto-refresh-minutes', 'value'),
    ],
    State('dump-alert-refresh-state-store', 'data'),
    prevent_initial_call=False
)
def update_dump_alert_table(_refresh_clicks, _n_intervals, threshold_pct, min_gp_drop, min_avg_daily_volume, min_potential_profit, min_volume_potential, volume_power, max_qty_factor, metric_values, window_values, custom_windows_text, alert_options, new_options, auto_refresh_value, auto_refresh_minutes, refresh_state):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial-load'
    effective_windows = _merged_dump_alert_windows(window_values, custom_windows_text)
    rows, items_scanned, refreshed_5m, threshold_pct, min_gp_drop, min_avg_daily_volume, min_potential_profit, min_volume_potential, volume_power, max_qty_factor, metric_values, window_values = _build_dump_alert_rows(
        threshold_pct,
        min_gp_drop,
        min_avg_daily_volume,
        min_potential_profit,
        min_volume_potential,
        volume_power,
        max_qty_factor,
        metric_values,
        effective_windows,
    )
    refresh_state = refresh_state or {}
    previous_keys = set(refresh_state.get('keys') or [])
    sticky_previous_keys = set(refresh_state.get('sticky_new_keys') or [])
    current_keys = [row['alert_key'] for row in rows]
    current_key_set = set(current_keys)
    is_first_load = not refresh_state.get('keys') and not refresh_state.get('sticky_new_keys')
    new_alert_keys = [] if is_first_load else [key for key in current_keys if key not in previous_keys]
    new_alert_key_set = set(new_alert_keys)
    highlight_enabled = 'highlight' in (alert_options or [])
    sound_enabled = 'sound' in (alert_options or [])
    show_new_only = 'show_new_only' in (new_options or [])
    sticky_new_enabled = 'sticky_new' in (new_options or [])
    auto_refresh_enabled = 'enabled' in (auto_refresh_value or [])

    if sticky_new_enabled:
        if trigger == 'dump-alert-refresh-btn':
            effective_new_key_set = set(new_alert_keys)
        else:
            effective_new_key_set = (sticky_previous_keys | new_alert_key_set) & current_key_set
    else:
        effective_new_key_set = new_alert_key_set

    for row in rows:
        row['is_new'] = 1 if row['alert_key'] in effective_new_key_set else 0

    if show_new_only:
        rows = [row for row in rows if row['is_new'] == 1]

    rows.sort(key=lambda row: (-row['is_new'], -row['drop_pct'], -row['gp_drop'], row['window_minutes'], row['item_name']))

    table_styles = _dump_alert_base_table_styles()
    if highlight_enabled:
        table_styles.insert(0, {
            'if': {'filter_query': '{is_new} = 1'},
            'backgroundColor': '#fff6d8',
            'boxShadow': 'inset 4px 0 0 #e3a008',
        })
        table_styles.insert(1, {
            'if': {'filter_query': '{is_new} = 1', 'column_id': 'item_name'},
            'fontWeight': 'bold',
        })

    formula_help_text = (
        f'Formula (live): max_qty = min(int(base_qty * {max_qty_factor:.2f}), buy_limit); '
        f'potential_profit = max(0, (reference_price * (1 - {GE_TAX:.2f}) - current_price) * max_qty); '
        f'volume_potential = floor(log10(|potential_profit| * avg_daily_volume^{volume_power:.2f}) * 100) / 100 when value > 1, else 0.'
    )

    metric_labels = ', '.join(_dump_alert_metric_label(metric) for metric in metric_values)
    window_label = ', '.join(f'{window}m' for window in window_values)
    status_text = (
        f'{len(rows)} dump alerts triggered across {items_scanned} scanned items '
        f'using {metric_labels} over {window_label} with thresholds of {threshold_pct:.1f}% drop, '
        f'{min_gp_drop:.0f} gp drop, {min_avg_daily_volume:.0f} daily volume, '
        f'{min_potential_profit:.0f} potential profit, {min_volume_potential:.2f} volume potential, '
        f'volume weight {volume_power:.2f}, and amount multiplier {max_qty_factor:.2f}.'
    )
    if new_alert_keys:
        status_text += f' {len(new_alert_keys)} new alert(s) were just triggered.'
    if show_new_only:
        status_text += ' Showing only new alerts.'
    refresh_source = 'fresh 5m data' if refreshed_5m else 'cached 5m data'
    refreshed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        refresh_minutes = int(float(auto_refresh_minutes)) if auto_refresh_minutes is not None else 5
    except (TypeError, ValueError):
        refresh_minutes = 5
    refresh_minutes = min(1440, max(1, refresh_minutes))
    auto_refresh_label = f'enabled ({refresh_minutes} min)' if auto_refresh_enabled else 'disabled'
    last_refresh_text = f'Last refresh: {refreshed_at} using {refresh_source}. Auto-refresh is {auto_refresh_label}.'
    new_summary_text = (
        f'New since last refresh: {len(new_alert_keys)}. '
        f'Marked as new right now: {len(effective_new_key_set)}.'
    )
    if sticky_new_enabled:
        new_summary_text += ' New markers persist across auto-refreshes until you press Refresh Now.'
    refresh_state_payload = {
        'keys': list(current_key_set),
        'sticky_new_keys': list(effective_new_key_set) if sticky_new_enabled else [],
    }
    sound_signal = {
        'play': bool(sound_enabled and new_alert_keys),
        'new_alert_count': len(new_alert_keys),
        'nonce': int(time.time() * 1000),
    }
    return rows, table_styles, formula_help_text, status_text, new_summary_text, last_refresh_text, refresh_state_payload, sound_signal


@app.callback(
    Output('dump-alert-timeseries-container', 'children'),
    [
        Input('dump-alert-table', 'selected_rows'),
        Input('dump-alert-table', 'selected_row_ids'),
        Input('dump-alert-table', 'active_cell'),
        Input('dump-alert-table', 'derived_viewport_data'),
        Input('dump-alert-table', 'data'),
        Input('dump-alert-chart-y-fit-btn', 'n_clicks'),
        Input('dump-alert-chart-y-fit-pct', 'value'),
        Input('dark-mode-store', 'data'),
    ],
    prevent_initial_call=False
)
def show_dump_alert_timeseries(selected_rows, selected_row_ids, active_cell, viewport_data, table_data, y_fit_clicks=0, y_fit_pct=15, dark_mode_data=None):
    adapted_table_data = []
    source_rows = table_data if isinstance(table_data, list) else []
    for row in source_rows:
        adapted_row = _dump_alert_chart_row_from_result(row)
        if adapted_row is not None:
            adapted_table_data.append(adapted_row)

    adapted_viewport_data = []
    source_viewport = viewport_data if isinstance(viewport_data, list) else []
    for row in source_viewport:
        adapted_row = _dump_alert_chart_row_from_result(row)
        if adapted_row is not None:
            adapted_viewport_data.append(adapted_row)

    if not adapted_table_data:
        return html.Div('No dump alert data available for charting.')

    if not selected_row_ids:
        selected_row_ids = [adapted_table_data[0]['id']]

    if not selected_rows:
        selected_rows = [0]

    return show_timeseries(
        selected_rows,
        selected_row_ids,
        active_cell,
        adapted_viewport_data,
        adapted_table_data,
        {},
        0,
        y_fit_clicks,
        y_fit_pct,
        dark_mode_data,
    )


app.clientside_callback(
    """
    function(signal) {
        if (!signal || !signal.play || !signal.new_alert_count) {
            return window.dash_clientside.no_update;
        }
        try {
            const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextCtor) {
                return 'audio-unsupported';
            }
            const audioContext = new AudioContextCtor();
            const tones = Math.min(signal.new_alert_count, 3);
            for (let index = 0; index < tones; index += 1) {
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                oscillator.type = 'triangle';
                oscillator.frequency.value = 880 + (index * 110);
                gainNode.gain.setValueAtTime(0.0001, audioContext.currentTime + (index * 0.16));
                gainNode.gain.exponentialRampToValueAtTime(0.12, audioContext.currentTime + 0.02 + (index * 0.16));
                gainNode.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.13 + (index * 0.16));
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                oscillator.start(audioContext.currentTime + (index * 0.16));
                oscillator.stop(audioContext.currentTime + 0.14 + (index * 0.16));
            }
            return 'played-' + signal.nonce;
        } catch (error) {
            return 'audio-error';
        }
    }
    """,
    Output('dump-alert-sound-dummy', 'children'),
    Input('dump-alert-sound-signal', 'data'),
    prevent_initial_call=True,
)

# --- Watchlist Table Update Callback ---

## Removed all callbacks referencing watchlist-table (pagination, sorting, data, etc.)


# Removed no-output debug callback that could break callback registration.



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

# ---------------------------------------------------------------------------
# REST API for RuneLite plugin  (served on app.server — same Flask instance)
# Endpoints:
#   GET /api/watchlist          → watchlist items with latest prices
#   GET /api/forecast           → last computed forecast table rows (cached)
# ---------------------------------------------------------------------------
from flask import jsonify as _jsonify

# Simple in-process store so the plugin can read the last forecast result
# without triggering a full recompute.  Updated by the Dash callback.
_last_forecast_rows: list = []


@app.server.route('/api/watchlist')
def api_watchlist():
    """Return watchlist items enriched with the latest RS Wiki prices."""
    watchlist_path = os.path.join(APP_DIR, 'Data', 'watchlist.json')
    try:
        with open(watchlist_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except Exception:
        items = []
    latest_prices = get_latest_prices()
    result = []
    for item in items:
        item_id = str(item.get('item_id', ''))
        price_data = latest_prices.get(item_id, {})
        low_p = price_data.get('low', 0)
        high_p = price_data.get('high', 0)
        entry_price = float(item.get('entry_price', 0) or 0)
        quantity = int(item.get('quantity', 1) or 1)
        tax = high_p * 0.02
        potential_profit = round((high_p - tax - entry_price) * quantity, 2) if high_p > 0 and entry_price > 0 else 0
        result.append({
            'item_id': item_id,
            'item_name': item.get('item_name', ''),
            'entry_price': entry_price,
            'quantity': quantity,
            'low_price': low_p,
            'high_price': high_p,
            'potential_profit': potential_profit,
            'added_time': item.get('added_time', ''),
        })
    return _jsonify(result)


@app.server.route('/api/forecast')
def api_forecast():
    """Return the last set of forecast rows computed by the Dash callback."""
    # _last_forecast_rows is populated by update_table_and_times via the store below.
    # If empty, return an empty list — the plugin will show a 'no data' message.
    safe_rows = []
    for row in _last_forecast_rows:
        safe_rows.append({
            'item_id': str(row.get('id', '')),
            'item_name': row.get('name', ''),
            'low_price': row.get('lowP', ''),
            'high_price': row.get('highP', ''),
            'forecast_price': row.get('forecast_price', ''),
            'profit': row.get('profit', ''),
            'roi': row.get('roi', ''),
            'potential_profit': row.get('potential_profit', ''),
            'trend': row.get('trend_direction', ''),
            'risk_level': row.get('risk_level', ''),
            'status': row.get('status', ''),
        })
    return _jsonify(safe_rows)


# --- Main block to launch Dash app and update 5m data on startup ---
if __name__ == '__main__':
    import json
    import os
    import traceback
    
    try:
        
        DATA_DIR = os.path.join(APP_DIR, 'Data')
        
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
            if not latest_chunk_is_current():
                fetch_and_save_5m_data_safe('startup')
            else:
                print("5m data is already up to date on startup.")
        except Exception as e:
            print(f"Error fetching 5m data on startup: {e}")
        
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