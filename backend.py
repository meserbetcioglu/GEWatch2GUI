import math

def save_filter_template(name, values):
    """Save a filter template with the given name and values (dict)."""
    try:
        if os.path.exists(TEMPLATE_FILE):
            with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
                templates = json.load(f)
        else:
            templates = {}
        templates[name] = values
        with open(TEMPLATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=2)
        return True
    except Exception as e:
        print(f"[TEMPLATE ERROR] Could not save template '{name}': {e}")
        return False

def load_filter_template(name):
    """Load a filter template by name. Returns dict or None."""
    try:
        if not os.path.exists(TEMPLATE_FILE):
            return None
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        return templates.get(name)
    except Exception as e:
        print(f"[TEMPLATE ERROR] Could not load template '{name}': {e}")
        return None

def list_filter_templates():
    """Return a list of saved filter template names."""
    try:
        if not os.path.exists(TEMPLATE_FILE):
            return []
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        return list(templates.keys())
    except Exception as e:
        print(f"[TEMPLATE ERROR] Could not list templates: {e}")
        return []

def delete_filter_template(name):
    """Delete a filter template by name."""
    try:
        if not os.path.exists(TEMPLATE_FILE):
            return False
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        if name in templates:
            del templates[name]
            with open(TEMPLATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(templates, f, indent=2)
            return True
        return False
    except Exception as e:
        print(f"[TEMPLATE ERROR] Could not delete template '{name}': {e}")
        return False
import os
import json
import math
import sys
from datetime import datetime, timedelta
from forecast_helpers import forecast_avg_price
import numpy as np

APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, 'Data')
TEMPLATE_FILE = os.path.join(DATA_DIR, 'filter_templates.json')
MAPPING_FILE = os.path.join(DATA_DIR, 'mapping_cache.json')
FIVE_M_FILE = os.path.join(DATA_DIR, '5m_data_cache.json')
PRICE_HISTORY_FILE = os.path.join(DATA_DIR, 'Price_History.json')
GE_TAX = 0.02

def calculate_risk_score(hist_prices, lookback=12):
    """
    Calculate risk metrics for an item based on recent price history.
    Returns dict with risk indicators or None if insufficient data.
    """
    if not hist_prices or len(hist_prices) < 3:
        return None
    
    # Use recent data (last lookback points)
    prices = hist_prices[-lookback:] if len(hist_prices) >= lookback else hist_prices
    prices = [p for p in prices if p is not None and p > 0]
    
    if len(prices) < 3:
        return None
    
    try:
        # 1. Coefficient of Variation (volatility measure)
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        coefficient_of_variation = (price_std / price_mean * 100) if price_mean > 0 else 0
        
        # 2. Price Range (percentage)
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range_pct = ((price_max - price_min) / price_mean * 100) if price_mean > 0 else 0
        
        # 3. Trend Change (recent vs older periods)
        trend_change_pct = 0
        if len(prices) >= 6:
            recent_mean = np.mean(prices[-3:])
            older_mean = np.mean(prices[-6:-3])
            trend_change_pct = ((recent_mean - older_mean) / older_mean * 100) if older_mean > 0 else 0
        
        # 4. Average Price Jump (volatility between consecutive points)
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                pct_change = abs((prices[i] - prices[i-1]) / prices[i-1] * 100)
                price_changes.append(pct_change)
        avg_price_jump = np.mean(price_changes) if price_changes else 0
        
        # 5. Stability Score (composite metric, lower = more stable)
        stability_score = coefficient_of_variation + (price_range_pct / 2) + (avg_price_jump / 2)
        
        return {
            'coefficient_of_variation': coefficient_of_variation,
            'price_range_pct': price_range_pct,
            'trend_change_pct': trend_change_pct,
            'avg_price_jump': avg_price_jump,
            'stability_score': stability_score
        }
    except Exception as e:
        print(f"[RISK CALC ERROR] {e}")
        return None

def assess_risk_level(risk_metrics):
    """
    Assess risk level based on statistical thresholds.
    Returns 'High Risk', 'Medium Risk', or 'Low Risk'
    """
    if not risk_metrics:
        return 'Unknown Risk'
    
    # High risk criteria (based on validation analysis)
    high_risk_flags = 0
    
    if risk_metrics['coefficient_of_variation'] > 30:
        high_risk_flags += 1
    
    if risk_metrics['price_range_pct'] > 50:
        high_risk_flags += 1
    
    if abs(risk_metrics['trend_change_pct']) > 20:
        high_risk_flags += 1
    
    if risk_metrics['stability_score'] > 60:
        high_risk_flags += 1
    
    # Determine risk level
    if high_risk_flags >= 2:
        return 'High Risk'
    elif high_risk_flags == 1 or risk_metrics['stability_score'] > 40:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

_PRICE_HISTORY_CACHE = {
    'mtime': None,
    'data': None,
}
_ITEM_PRICE_HISTORY_CACHE = {
    'mtime': None,
    'lookback_hours': None,
    'data': None,
}


def _get_price_history_file_mtime():
    try:
        return os.path.getmtime(PRICE_HISTORY_FILE)
    except OSError:
        return None


def _get_price_history_cached():
    current_mtime = _get_price_history_file_mtime()
    if _PRICE_HISTORY_CACHE['data'] is None or _PRICE_HISTORY_CACHE['mtime'] != current_mtime:
        _PRICE_HISTORY_CACHE['data'] = load_json(PRICE_HISTORY_FILE)
        _PRICE_HISTORY_CACHE['mtime'] = current_mtime

        _ITEM_PRICE_HISTORY_CACHE['data'] = None
        _ITEM_PRICE_HISTORY_CACHE['mtime'] = current_mtime
        _ITEM_PRICE_HISTORY_CACHE['lookback_hours'] = None

    return _PRICE_HISTORY_CACHE['data']


def _get_item_price_history_cached(hours_lookback=168):
    price_history = _get_price_history_cached()
    current_mtime = _PRICE_HISTORY_CACHE['mtime']

    if (
        _ITEM_PRICE_HISTORY_CACHE['data'] is None
        or _ITEM_PRICE_HISTORY_CACHE['mtime'] != current_mtime
        or _ITEM_PRICE_HISTORY_CACHE['lookback_hours'] != hours_lookback
    ):
        _ITEM_PRICE_HISTORY_CACHE['data'] = get_item_price_history(price_history, hours_lookback)
        _ITEM_PRICE_HISTORY_CACHE['mtime'] = current_mtime
        _ITEM_PRICE_HISTORY_CACHE['lookback_hours'] = hours_lookback

    return _ITEM_PRICE_HISTORY_CACHE['data']

def get_update_times():
    try:
        with open(FIVE_M_FILE, 'r', encoding='utf-8') as f:
            five_m_data = json.load(f)
        chunks = five_m_data.get('chunks')
        if chunks and len(chunks) > 0:
            last_chunk = max(chunks, key=lambda c: c['timestamp'])
            five_m_dt = datetime.strptime(last_chunk['timestamp'], '%Y-%m-%dT%H:%M:%S')
        else:
            five_m_dt = None
    except Exception:
        five_m_dt = None
    try:
        with open(PRICE_HISTORY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if '"' in line:
                    break
            f.seek(0)
            data = json.load(f)
        if data:
            ts_keys = [k for k in data.keys() if k[0].isdigit()]
            if ts_keys:
                one_h_dt = datetime.strptime(max(ts_keys), '%Y-%m-%d %H:%M:%S')
            else:
                one_h_dt = None
        else:
            one_h_dt = None
    except Exception:
        one_h_dt = None
    return one_h_dt, five_m_dt

def get_latest_prices():
    try:
        import requests
        resp = requests.get("https://prices.runescape.wiki/api/v1/osrs/latest", timeout=10)
        if resp.status_code == 200:
            return resp.json()['data']
        print("[ERROR] Could not fetch latest prices from RS Wiki API: HTTP", resp.status_code)
        return {}
    except Exception as e:
        print("[ERROR] Could not fetch latest prices from RS Wiki API:", e)
        return {}


def get_5m_prices():
    
    data = load_json(FIVE_M_FILE)
    chunks = data.get('chunks', [])

    import datetime
    import time
        

    prices = {}

    for chunk in chunks:
        if chunk is None:
            continue
        ts = chunk['timestamp']
        data_chunk = chunk.get('data', {}).get('data', {})
        for item_id, entry in data_chunk.items():
            if item_id not in prices:
                prices[item_id] = []
            entry_copy = entry.copy()
            entry_copy['timestamp'] = ts
            prices[item_id].append(entry_copy)

    return prices
        


def get_item_price_history(price_history, hours_lookback=168):
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=hours_lookback)
    item_prices = {}
    for ts_str, ts_data in price_history.items():
        try:
            ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except Exception:
            continue
        if ts < cutoff:
            continue
        for item_id, pdata in ts_data.get('data', {}).items():
            if item_id not in item_prices:
                item_prices[item_id] = []
            entry = {
                'timestamp': ts_str,
                'avgLowPrice': pdata.get('avgLowPrice'),
                'lowPriceVolume': pdata.get('lowPriceVolume'),
                'avgHighPrice': pdata.get('avgHighPrice'),
                'highPriceVolume': pdata.get('highPriceVolume')
            }
            item_prices[item_id].append(entry)
    return item_prices

def analyze_forecast_gui(filters, forecast_sell_time, forecast_strategy='wma', name_filter = [], max_avg_trade_time=None, volume_power = 1, max_qty_factor=1, forecast_hours = 48, buy_price_type='low', forecast_price_type='avg'):

    import warnings

    warnings.simplefilter("error", RuntimeWarning)

    import time

    t_start = time.time()


    forecast_hours = int(filters.get('FORECAST_HOURS', 168))
    try:
        forecast_recency_minutes = int(filters.get('FORECAST_RECENCY_MINUTES', 0) or 0)
    except (TypeError, ValueError):
        forecast_recency_minutes = 0
    if forecast_recency_minutes < 0:
        forecast_recency_minutes = 0
    try:
        max_qty_factor = float(max_qty_factor) if max_qty_factor is not None else 1.0
    except (TypeError, ValueError):
        max_qty_factor = 1.0
    if max_qty_factor < 0:
        max_qty_factor = 0.0
    forecast_sell_time_val = filters.get('FORECAST_SELL_TIME', forecast_sell_time)
    if forecast_sell_time_val is None:
        forecast_sell_time = 0
    else:
        forecast_sell_time = int(forecast_sell_time_val)
    try:
        mapping_json = load_json(MAPPING_FILE)
        if 'data' not in mapping_json:
            return []
        mapping = mapping_json['data']
        if not isinstance(mapping, dict) or not mapping:
            return []
    except Exception as e:
        return []
    if name_filter != []:
        filtered_mapping = {item_id: info for item_id, info in mapping.items() if any(nf.lower() in info['name'].lower() for nf in name_filter)}
        mapping = filtered_mapping
    price_history = load_json(PRICE_HISTORY_FILE)

    item_history_chart = get_item_price_history(price_history, forecast_hours)
    
    item_history_5m = get_5m_prices()

    if forecast_sell_time_val <= 180:
        item_history = item_history_5m
    else:
        item_history = item_history_chart

    latest_prices = get_latest_prices()

    # Fetch live prices from RS Wiki API
    results_for_table = []
    forecast_results = []
    all_item_ids = set(mapping.keys())



    # Deduplicate all_item_ids before processing
    unique_item_ids = list(dict.fromkeys(all_item_ids))
    
    # unique_item_ids = [item_id for item_id in unique_item_ids if item_id == '7946']
    
    print('[DEBUG] Starting forecast with ')

    now = time.time()

    t1d = 0.0
    t2d = 0.0
    t3d = 0.0
    t4d = 0.0
    t5d = 0.0
    t6d = 0.0
    t7d = 0.0
    t8d = 0.0
    
    for idx, item_id in enumerate(unique_item_ids):
        try:

            t1 = time.time()

            if max_avg_trade_time and max_avg_trade_time > 0:
                latest = latest_prices.get(item_id)
                if not latest:
                    continue

                low_time = latest.get('lowTime')
                high_time = latest.get('highTime')

                if low_time is None or high_time is None:
                    continue

                low_diff = (now - low_time) // 60
                high_diff = (now - high_time) // 60

                if (low_diff + high_diff) / 2 > max_avg_trade_time:
                    continue

            hist = item_history.get(item_id)

            if not hist: # or len(hist) < forecast_hours / 2:
                item_history.pop(item_id, None)
                continue

            # Exclude most recent points from forecast training data when requested.
            interval_minutes = 5 if forecast_sell_time_val <= 180 else 60
            skip_points = int(math.ceil(forecast_recency_minutes / interval_minutes)) if forecast_recency_minutes > 0 else 0
            hist_for_forecast = hist[:-skip_points] if skip_points > 0 else hist
            if not hist_for_forecast:
                continue

            
            hist_5m = item_history_5m.get(item_id, [])
            latest = latest_prices.get(item_id, {})

            for e in hist:
                e['avgLowPrice'] = float(e.get('avgLowPrice') or 0)
                e['avgHighPrice'] = float(e.get('avgHighPrice') or 0)
                e['lowPriceVolume'] = float(e.get('lowPriceVolume') or 0)
                e['highPriceVolume'] = float(e.get('highPriceVolume') or 0)


            # if item_id == '31235':
            #     pass


            t2 = time.time()
            t1d += t2 - t1

            name = mapping[item_id]['name'] if item_id in mapping else ''
            last_entry = hist[-1] if hist else {}
            avg_low = last_entry.get('avgLowPrice', 0)
            avg_high = last_entry.get('avgHighPrice', 0)
            low_vol = last_entry.get('lowPriceVolume', 0)
            high_vol = last_entry.get('highPriceVolume', 0)
            buy_limit = mapping[item_id].get('buy_limit') or mapping[item_id].get('limit') if item_id in mapping else None
            constants = filters.get('CONSTANTS', {})
            # Calculate avg_daily_volume
            avg_daily_volume = 0
            if hist:
                total_hist_vol = sum((e.get('lowPriceVolume',0) or 0) + (e.get('highPriceVolume',0) or 0) for e in hist)
                days = max(1, (len(hist) / 24))
                avg_daily_volume = int(total_hist_vol / days)
            # Forecast price

            lowVol_recent = sum(e.get('lowPriceVolume', 0) for e in hist_5m[-6:]) #Last 30 Min
            highVol_recent = sum(e.get('highPriceVolume', 0) for e in hist_5m[-6:]) #Last 30 Min
            
            lowVol_recentPerc = lowVol_recent/avg_daily_volume if lowVol_recent > 0 and avg_daily_volume > 0 else 0
            highVol_recentPerc = highVol_recent/avg_daily_volume if highVol_recent > 0 and avg_daily_volume > 0 else 0

            
            hist_prices = [
                (e.get('avgHighPrice',0)*e.get('highPriceVolume',0) + e.get('avgLowPrice',0)*e.get('lowPriceVolume'))/(e.get('highPriceVolume',0) + e.get('lowPriceVolume',0)) for e in hist
                if e.get('highPriceVolume') > 0 or e.get('lowPriceVolume') > 0
            ]

            hist_prices_high = [
                e.get('avgHighPrice',0) for e in hist
            ]

            hist_prices_low = [
                e.get('avgLowPrice',0) for e in hist
            ]

            clean_hist_prices = [p for p in hist_prices.copy() if p is not None and p > 0] 

            avg_stds = []

            t3 = time.time()
            t2d += t3 - t2

            if hist_prices_high and hist_prices_low and clean_hist_prices and len(hist_prices_high) >= 3 and len(hist_prices_low) >= 3 and len(clean_hist_prices) >= 12 :
                N = min(12, len(clean_hist_prices))
                xs = np.arange(N)
                ys = np.array(clean_hist_prices[-N:])
                ysH = np.array(hist_prices_high[-N:])
                ysL = np.array(hist_prices_low[-N:])

                if N > 1:
                    coeffs = np.polyfit(xs, ys, 1)
                    slope = coeffs[0]
                    # Threshold for trend: 1% of mean per step
                    mean_y = np.mean(ys)
                    threshold = abs(mean_y) * 0.01
                    if forecast_sell_time < 180:
                        threshold = threshold / 12
                    if slope > threshold:
                        trend_direction = 'UP'
                    elif slope < -threshold:
                        trend_direction = 'DOWN'
                    else:
                        trend_direction = 'FLAT'
                else:
                    trend_direction = 'FLAT'

                detrend_prices = ys - (coeffs[0] * xs + coeffs[1])
                detrend_pricesH = []
                detrend_pricesL = []
                for i in range(N):
                    if ysH[i] > 0:
                        t1 = ysH[i] - (coeffs[0] * xs[i] + coeffs[1])
                        detrend_pricesH.append(ysH[i] - (coeffs[0] * xs[i] + coeffs[1]))
                    if ysL[i] > 0:
                        t2 = ysL[i] - (coeffs[0] * xs[i] + coeffs[1])
                        detrend_pricesL.append(ysL[i] - (coeffs[0] * xs[i] + coeffs[1]))

                detrend_std_prices = detrend_pricesH + detrend_pricesL

                avg_std = np.std(detrend_std_prices) if len(detrend_std_prices) > 0 else 0
                norm_std = (avg_std / np.mean(ys)) * 100 if np.mean(ys) != 0 and avg_std > 0 else 0

                support_high = ys + 1.2 * avg_std
                support_low = ys - 1.2 * avg_std

                avg_stds.append(avg_std)


            avg_std_med = np.median(avg_stds) if len(avg_stds) > 0 else 0

            
            t4 = time.time()
            t3d += t4 - t3

            forecast_hist = [entry.copy() for entry in hist_for_forecast]
            limit = 1.2 * avg_std_med

            for e in forecast_hist:
                hpv = e.get('highPriceVolume', 0)
                lpv = e.get('lowPriceVolume', 0)

                total_vol = hpv + lpv
                if total_vol <= 0:
                    continue

                ahp = e.get('avgHighPrice', 0)
                alp = e.get('avgLowPrice', 0)

                mean = (ahp * hpv + alp * lpv) / total_vol

                lower = mean - limit
                upper = mean + limit

                if alp > 0 and alp < lower:
                    e['avgLowPrice'] = lower

                if ahp > 0 and ahp > upper:
                    e['avgHighPrice'] = upper

            t5 = time.time()
            t4d += t5 - t4

            try:
                forecasted_prices = forecast_avg_price(forecast_hist, forecast_sell_time, strategy=forecast_strategy, forecast_price_type=forecast_price_type)
            except Exception as e:
                forecasted_prices = []
                print(f"[FORECAST ERROR] Item ID {item_id} - {e}")
            
            t6 = time.time()
            t5d += t6 - t5
            
            
            if isinstance(forecasted_prices, list) and forecasted_prices:
                forecast_price = forecasted_prices[-1]
            else:
                forecast_price = 0
            if forecast_sell_time == 0:
                forecast_price = latest.get('high', 0)
            # Defensive: ensure forecast_price and latest['low'] are valid numbers

            try:
                forecast_price = float(forecast_price)
            except (TypeError, ValueError):
                forecast_price = 0
            forecast_price = int(math.floor(forecast_price)) if forecast_price is not None else 0
            
            # Use selected buy price type
            if buy_price_type == 'high':
                buy_price = latest.get('high', 0)
            elif buy_price_type == 'avg_low_30min':
                # Calculate volume-weighted average low price from last 30 minutes
                # Use 5m data if available (last 6 data points = 30 minutes)
                if hist_5m and len(hist_5m) > 0:
                    # Get last 6 entries (30 minutes worth of 5-minute data)
                    recent_entries = hist_5m[-6:]
                    total_weighted_price = 0
                    total_volume = 0
                    valid_entries_count = 0
                    debug_entries = []
                    
                    for entry in recent_entries:
                        # Handle potential None or missing values
                        low_price = entry.get('avgLowPrice', 0) or 0
                        low_volume = entry.get('lowPriceVolume', 0) or 0
                        
                        try:
                            low_price = float(low_price)
                            low_volume = float(low_volume)
                        except (TypeError, ValueError):
                            low_price = 0
                            low_volume = 0
                        
                        debug_entries.append({'price': low_price, 'volume': low_volume})
                        
                        if low_price > 0 and low_volume > 0:
                            total_weighted_price += low_price * low_volume
                            total_volume += low_volume
                            valid_entries_count += 1
                    
                    if total_volume > 0:
                        # Use volume-weighted average
                        buy_price = total_weighted_price / total_volume
                        buy_price = int(round(buy_price))
                        print(f"[DEBUG] Item {item_id}: 30min weighted avg = {buy_price}")
                        print(f"[DEBUG]   Valid entries: {valid_entries_count}/{len(recent_entries)}, Total volume: {total_volume}")
                        print(f"[DEBUG]   All entries: {debug_entries}")
                        print(f"[DEBUG]   Current low price: {latest.get('low', 0)}")
                    else:
                        # Fallback to current low price if no volume data
                        buy_price = int(latest.get('low', 0))
                        print(f"[DEBUG] Item {item_id}: No volume data for 30min avg (entries: {debug_entries}), using current low = {buy_price}")
                else:
                    # Fallback to current low price if no 5m data available
                    buy_price = int(latest.get('low', 0))
                    print(f"[DEBUG] Item {item_id}: No 5m data available (len={len(hist_5m) if hist_5m else 0}), using current low = {buy_price}")
            else:  # default to 'low'
                buy_price = int(latest.get('low', 0))
            
            try:
                buy_price = int(float(buy_price))
            except (TypeError, ValueError):
                buy_price = 0
            
            # Calculate profit
            if buy_price <= 0 or forecast_price is None or forecast_price <= 0:
                profit = 0
            else:
                profit = forecast_price * (1-GE_TAX) - buy_price
            profit = math.ceil(profit) if profit > 0 else 0
            # Calculate ROI
            roi = profit / buy_price if buy_price else 0
            # Calculate rel_spread
            if avg_low and avg_high and (avg_high + avg_low) != 0:
                rel_spread = ((avg_high - avg_low) / ((avg_high + avg_low) / 2)) * 100
            else:
                rel_spread = 0

                
            t7 = time.time()
            t6d += t7 - t6

            # avg_std = 0
            # if len(hist) > 1:
            #     prices = [((e.get('avgHighPrice',0) or 0) + (e.get('avgLowPrice',0) or 0)) / 2 for e in hist if e.get('avgHighPrice') is not None and e.get('avgLowPrice') is not None]
            #     mean_price = sum(prices) / len(prices)
            #     variance = sum((p - mean_price) ** 2 for p in prices) / (len(prices) - 1)
            #     avg_std = math.sqrt(variance)
            #     norm_std = (avg_std / mean_price) * 100 if mean_price != 0 else 0
            
            # Calculate max_qty only if profit > 0
            if profit > 0:
                try:
                    if(forecast_sell_time < 60):
                        max_qty = min(
                            int(buy_limit) if buy_limit is not None else 0,
                            int(avg_daily_volume * 0.1 * (30/1440)) if avg_daily_volume is not None else 0,
                            lowVol_recent*2 if lowVol_recent else 0
                        )
                    else:
                        max_qty = min(
                        int(buy_limit) if buy_limit is not None else 0,
                        int(avg_daily_volume * 0.6 * (forecast_sell_time / 1440)) if avg_daily_volume is not None else 0,
                            lowVol_recent*2 if lowVol_recent else 0
                        )
                except Exception:
                    max_qty = 0
            else:
                max_qty = 0

            if forecast_sell_time < 1440:
                max_qty = min(max_qty, avg_daily_volume*0.05)

            buy_limit_cap = int(buy_limit) if buy_limit is not None else 0
            max_qty = int(max_qty * max_qty_factor) if max_qty > 0 else 0
            max_qty = min(max_qty, buy_limit_cap)
            # Calculate potential profit and volume potential
            potential_profit = profit * max_qty if profit > 0 else 0
            volume_potential = math.floor(math.log10(abs(potential_profit) * (avg_daily_volume ** volume_power))*100)/100 if avg_daily_volume is not None and potential_profit is not None and abs(potential_profit)*(avg_daily_volume**volume_power) > 1 else 0
            # Calculate trend direction
            # hist_prices = [
            #     (e.get('avgHighPrice',0) + e.get('avgLowPrice',0))/2 for e in hist
            #     if e.get('avgHighPrice') is not None and e.get('avgLowPrice') is not None
            # ]

            
            t8 = time.time()
            t7d += t8 - t7

            trend_direction = 'FLAT'
            status_msgs = []
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_score(clean_hist_prices, lookback=12)
            risk_level = assess_risk_level(risk_metrics)
            
            # Add risk level to status if high or medium risk
            if risk_level == 'High Risk':
                status_msgs.append('High Risk')
            elif risk_level == 'Medium Risk':
                status_msgs.append('Medium Risk')

            abnormally_high = False
            abnormally_low = False

            lowVol_high = False
            lowVol_low = False
            highVol_high = False
            highVol_low = False


            if clean_hist_prices and len(clean_hist_prices) >= 3:
                # Use linear regression slope over last 12 points (or all if less)
                N = min(12, len(clean_hist_prices))
                xs = np.arange(N)
                ys = np.array(clean_hist_prices[-N:])
                if N > 1:
                    coeffs = np.polyfit(xs, ys, 1)
                    slope = coeffs[0]
                    # Threshold for trend: 0.5% of mean per step
                    mean_y = np.mean(ys)
                    threshold = abs(mean_y) * 0.005
                    if forecast_sell_time < 180:
                        threshold = threshold / 12
                    if slope > threshold:
                        trend_direction = 'UP'
                    elif slope < -threshold:
                        trend_direction = 'DOWN'
                    else:
                        trend_direction = 'FLAT'
                else:
                    trend_direction = 'FLAT'

                detrend_prices = ys - (coeffs[0] * xs + coeffs[1])
                avg_std = np.std(detrend_prices) if len(detrend_prices) > 0 else 0
                norm_std = (avg_std / np.mean(ys)) * 100 if np.mean(ys) != 0 and avg_std > 0 else 0

                support_high = ys + 1.2 * avg_std
                support_low = ys - 1.2 * avg_std

                # Keep rapid fall and abnormal price logic
                # if len(hist_prices) >= 6:
                #     recent_max = max(hist_prices[-6:])
                #     if ys[-1] < recent_max * 0.9:
                #         rapid_fall = True
                #         status_msgs.append('Rapid fall!')
                # if len(hist_prices) >= 24:
                #     mean24 = sum(hist_prices[-24:]) / 24
                #     if ys[-1] > mean24 * 1.2:
                #         abnormal_price = True
                #         status_msgs.append('Abnormally high!')

                # Compare last detrended price against detrended bounds
                recent_min_med = np.min(detrend_prices) if len(detrend_prices) > 0 else 0
                recent_max_med = np.max(detrend_prices) if len(detrend_prices) > 0 else 0
                detrend_last = detrend_prices[-1] if len(detrend_prices) > 0 else 0

                # Use a more conservative threshold to avoid over-flagging
                threshold_k = 1.2
                if detrend_last > recent_max_med + threshold_k * avg_std:
                    abnormally_high = True
                    status_msgs.append('Abnormally high!')
                elif detrend_last < recent_min_med - threshold_k * avg_std:
                    abnormally_low = True
                    status_msgs.append('Abnormally low!')

                if lowVol_recentPerc < 0.021/5:
                    lowVol_low = True
                    status_msgs.append('Low-Volume low!')
                elif lowVol_recentPerc > 0.021*5:
                    lowVol_high = True
                    status_msgs.append('Low-Volume high!')
                    
                if highVol_recentPerc < 0.021/5:
                    highVol_low = True
                    status_msgs.append('High-Volume low!')
                elif highVol_recentPerc > 0.021*5:
                    highVol_high = True
                    status_msgs.append('High-Volume high!')

                # if ys[-1] < recent_min - 1.2 * avg_std:
                #     abnormally_low = True
                #     status_msgs.append('Abnormally low!') 
                # elif ys[-1] > recent_max + 1.2 * avg_std:
                #     abnormally_high = True
                #     status_msgs.append('Abnormally high!')
            
            row_color = ''
            if profit > 0 and roi > 0.05 and not abnormally_low and not abnormally_high:
                row_color = 'lightgreen'
            elif abnormally_low:
                row_color = 'orange'
            elif abnormally_high:
                row_color = 'red'
            else:
                row_color = 'white'
            # if trend_direction == 'UP':
            #     status_msgs.append('Trending up')
            # elif trend_direction == 'DOWN':
            #     status_msgs.append('Trending down')
            # elif trend_direction == 'FLAT':
            #     status_msgs.append('Flat trend')

            status = '; '.join(status_msgs) if status_msgs else 'Normal'
            
            # Debug: log first few status values
            if len(results_for_table) < 5:
                print(f"[DEBUG-BACKEND] Item {item_id} ({name}): status_msgs={status_msgs}, status={status}")
            
            # Use buy_price for lowP only when avg_low_30min is selected
            display_low_price = buy_price if buy_price_type == 'avg_low_30min' else latest.get('low', 0)
            
            row = {
                'id': item_id,
                'name': name,
                'lowP': display_low_price,
                'highP': latest.get('high', 0),  # avg_high,
                'buy_price': buy_price,  # Actual buy price used in profit calculation
                'sell_price': forecast_price,  # Actual sell price (forecast)
                'lowVol': low_vol,
                'highVol': high_vol,
                'rel_spread': rel_spread,
                'forecast_price': forecast_price,
                'profit': profit,
                'roi': roi,
                'avg_daily_volume': avg_daily_volume,
                'buy_limit': buy_limit,
                'max_qty': max_qty,
                'potential_profit': potential_profit,
                'volume_potential': volume_potential,
                'trend_direction': trend_direction,
                'row_color': row_color,
                'status': status,
                'risk_level': risk_level,
                'risk_metrics': risk_metrics,
                'best_sell_time': None,
                'max_profit': None,
                'avg_std': avg_std if 'avg_std' in locals() else 0,
                'norm_std': norm_std if 'norm_std' in locals() else 0,
                'support_high': support_high.tolist() if 'support_high' in locals() else [],
                'support_low': support_low.tolist() if 'support_low' in locals() else [],
                'support_high_latest': support_high[-1] if 'support_high' in locals() else 0,
                'support_low_latest': support_low[-1] if 'support_low' in locals() else 0,
                'lowVol_recent': lowVol_recent if lowVol_recent > 0 else 0,
                'highVol_recent': highVol_recent if highVol_recent > 0 else 0
            }
            results_for_table.append(row)
            forecast_result = {
                'id': item_id,
                'forecast_price': forecasted_prices
            }
            forecast_results.append(forecast_result)
            # Timing code removed
            # print timing info removed

            t9 = time.time()
            t8d += t9 - t8

        except Exception as e:
            print(f"[ERROR] Could not process item ID {item_id}: {e}")
            pass

    # Optional backend-side status filtering (mirrors UI include/exclude behavior)
    status_filter = filters.get('STATUS_FILTER') if isinstance(filters, dict) else None
    status_filter_mode = filters.get('STATUS_FILTER_MODE', 'include') if isinstance(filters, dict) else 'include'
    
    print(f"[DEBUG-BACKEND] status_filter from filters: {status_filter}")
    print(f"[DEBUG-BACKEND] status_filter_mode: {status_filter_mode}")
    print(f"[DEBUG-BACKEND] Results before filter: {len(results_for_table)} items")
    
    if status_filter and isinstance(status_filter, list) and len(status_filter) > 0:
        # Normalize filter terms (strip whitespace)
        normalized_filter = [sf.strip() for sf in status_filter if sf and isinstance(sf, str)]
        
        print(f"[DEBUG-BACKEND] Filter terms: {normalized_filter}")
        
        # Debug: Show sample of actual status values
        sample_statuses = [r.get('status', '') for r in results_for_table[:10]]
        print(f"[DEBUG-BACKEND] Sample status values (first 10): {sample_statuses}")
        
        if status_filter_mode == 'exclude':
            results_before = len(results_for_table)
            results_for_table = [r for r in results_for_table if not any(sf in r.get('status', '') for sf in normalized_filter)]
            print(f"[DEBUG-BACKEND] Exclude mode: {results_before} -> {len(results_for_table)} items")
        else:
            # Include mode: handle "Normal" with exact match, others with substring match
            results_before = len(results_for_table)
            filtered = []
            for r in results_for_table:
                item_status = r.get('status', '')
                # Check if any filter term matches
                matched = False
                for sf in normalized_filter:
                    if sf == 'Normal':
                        # Exact match for Normal (no warning flags)
                        if item_status == 'Normal':
                            matched = True
                            break
                    else:
                        # Substring match for other terms
                        if sf in item_status:
                            matched = True
                            break
                if matched:
                    filtered.append(r)
            results_for_table = filtered
            print(f"[DEBUG-BACKEND] Include mode: {results_before} -> {len(results_for_table)} items")
            
        # Keep forecasts aligned with filtered rows
        allowed_ids = {str(r['id']) for r in results_for_table if 'id' in r}
        forecast_results = [fr for fr in forecast_results if str(fr.get('id')) in allowed_ids]
        print(f"[DEBUG-BACKEND] Forecasts aligned: {len(forecast_results)} items")
    
    # Second pass: exclude items matching exclusion terms
    status_exclude = filters.get('STATUS_EXCLUDE') if isinstance(filters, dict) else None
    if status_exclude and isinstance(status_exclude, list) and len(status_exclude) > 0:
        exclude_terms = [s.strip() for s in status_exclude if s and isinstance(s, str)]
        if exclude_terms:
            results_before = len(results_for_table)
            results_for_table = [r for r in results_for_table if not any(term in r.get('status', '') for term in exclude_terms)]
            print(f"[DEBUG-BACKEND] After exclude pass: {results_before} -> {len(results_for_table)} items")
            # Keep forecasts aligned
            allowed_ids = {str(r['id']) for r in results_for_table if 'id' in r}
            forecast_results = [fr for fr in forecast_results if str(fr.get('id')) in allowed_ids]

    # Remove duplicate items by ID before returning
    unique_results = list({str(row['id']): row for row in results_for_table if 'id' in row}.values())
    unique_forecasts = list({str(row['id']): row for row in forecast_results if 'id' in row}.values())


    print(
        "[DEBUG] Backend times: "
        f"t1d:{t1d:.2f}s | "
        f"t2d:{t2d:.2f}s | "
        f"t3d:{t3d:.2f}s | "
        f"t4d:{t4d:.2f}s | "
        f"t5d:{t5d:.2f}s | "
        f"t6d:{t6d:.2f}s | "
        f"t7d:{t7d:.2f}s | "
        f"t8d:{t8d:.2f}s"
    )
    
    return unique_results, unique_forecasts, latest_prices

def get_item_timeseries(item_id, hours_lookback=168):
    item_history = _get_item_price_history_cached(hours_lookback)

    hist = item_history.get(item_id)
    if hist is None:
        hist = item_history.get(str(item_id))
    if hist is None:
        try:
            hist = item_history.get(int(item_id), [])
        except Exception:
            hist = []

    timestamps = [e['timestamp'] for e in hist]
    avgLowPrice = [e.get('avgLowPrice') for e in hist]
    avgHighPrice = [e.get('avgHighPrice') for e in hist]
    lowVol = [e.get('lowPriceVolume') for e in hist]
    highVol = [e.get('highPriceVolume') for e in hist]
    return timestamps, avgLowPrice, avgHighPrice, lowVol, highVol


if __name__ == "__main__":
    import sys
    # Example filters, adjust as needed
    filters = {
        'MIN_FORECAST_PROFIT': 0,
        'MIN_POTENTIAL_PROFIT': 0,
        'MIN_VOLUME_POTENTIAL': 0,
        'MIN_FORECAST_ROI': 0,
        'FORECAST_HOURS': 24,
        'FORECAST_SELL_TIME': 60
    }
    forecast_sell_time = filters['FORECAST_SELL_TIME']
    forecast_strategy = 'wma'
    name_filter = []
    if len(sys.argv) > 1:
        name_filter = [sys.argv[1]]
    results, forecasts, _latest_prices = analyze_forecast_gui(filters, forecast_sell_time, forecast_strategy, name_filter=name_filter)
    print("Results:")
    print("\nForecast len: ", len(forecasts))
    