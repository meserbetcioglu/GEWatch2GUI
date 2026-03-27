import requests
import numpy as np
from datetime import datetime, timedelta
import math
from numpy.linalg import LinAlgError
import warnings

# Suppress convergence warnings from statsmodels
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
def forecast_avg_price(hist, minutes_ahead=60, strategy='wma', latest_prices=None, forecast_price_type='avg'):

    # --- UI alias handling ---
    if strategy == 'long_range':
        strategy = 'mix'



    prices = []

    for e in hist:
        highP = e.get('avgHighPrice') or 0
        highV = e.get('highPriceVolume') or 0
        lowP  = e.get('avgLowPrice') or 0
        lowV  = e.get('lowPriceVolume') or 0

        # Calculate price based on selected forecast price type
        if forecast_price_type == 'high':
            price = highP
        elif forecast_price_type == 'low':
            price = lowP
        else:  # 'avg' - volume-weighted average
            totalV = highV + lowV
            if totalV <= 0:
                continue
            price = (highP * highV + lowP * lowV) / totalV
        
        if price > 0:
            prices.append(price)

    if not prices:
        return []

    MAX_POINTS = {
        'wma': 200,
        'ewma': 200,
        'wma_trend': 200,
        'linear': 200,
        'poly': 200,
        'brown': 200,
        'holt_winters': 96,
        'arima': 80,
        'sarimax': 80,
        'prophet': 120,
        'lstm': 120,
        'rnn': 120,
        'cnn': 120,
        'nbeats': 120,
        'mix': 2016,  # allow 7d context
    }

    limit = MAX_POINTS.get(strategy)
    if limit:
        prices = prices[-limit:]

    interval_minutes = 60 if minutes_ahead > 180 else 5
    steps = max(1, int(minutes_ahead // interval_minutes))

    # --- flat market short-circuit ---
    if len(prices) >= 5:
        if max(prices) - min(prices) < 1:
            return [int(prices[-1])] * steps

    # ---- STRATEGIES ----
    STRATEGIES = {
        'wma': lambda: wma(prices, steps, preds=[]),
        'wma_trend': lambda: wma_trend(prices, steps),
        'ewma': lambda: ewma(prices, steps, alpha=0.5, preds=[]),
        'linear': lambda: linear_regression_forecast(prices, steps, preds=[]),
        'poly': lambda: polynomial_regression_forecast(prices, steps, degree=2, preds=[]),
        'brown': lambda: browns_double_moving_average(prices, steps),

        'arima': lambda: arima_forecast(prices, steps) or [None],
        'holt_winters': lambda: holt_winters_forecast(prices, steps) or [None],
        'sarimax': lambda: sarimax_forecast(prices, steps) or [None],
        'median_trend': lambda: median_trend(
            prices,
            steps=steps,
            window=24,
            max_slope_frac=0.02
        ),
        'robust_ewma': lambda: robust_ewma(
            prices, steps=steps, alpha=0.3, window=20, clip_sigma=1.5
        ),

        # ✅ NEW: weekly-aware mean reversion
        'median_reversion_weekly': lambda: median_reversion_forecast_weekly(
            prices,
            steps=steps,
            window_24h=288,
            window_7d=2016,
            slope_penalty=0.5,
            bound_frac=0.10
        ),
    }

    # ---- MIX LOGIC (THIS IS THE IMPORTANT PART) ----
    if strategy == 'mix':
        if minutes_ahead <= 180:
            return STRATEGIES['robust_ewma']()
        elif minutes_ahead <= 1440:
            return STRATEGIES['median_reversion_weekly']()
        else:
            return long_range_forecast(
                prices,
                steps,
                window=96,
                max_slope_frac=0.001,
                bound_frac=0.20
            )

    return STRATEGIES[strategy]()
import numpy as np

def median_reversion_forecast_weekly(
    prices,
    steps,
    window_24h=288,     # 24h @ 5m
    window_7d=2016,     # 7d @ 5m
    slope_penalty=0.5,  # reduce trend influence
    bound_frac=0.10     # max deviation from anchor
):
    if not prices or steps <= 0:
        return []

    prices = np.asarray(prices, dtype=float)

    # --- Windows ---
    recent_24h = prices[-window_24h:] if len(prices) >= window_24h else prices
    recent_7d  = prices[-window_7d:]  if len(prices) >= window_7d  else prices

    med_24h = np.median(recent_24h)
    med_7d  = np.median(recent_7d)

    # --- Anchor (weighted toward recent regime) ---
    anchor = 0.65 * med_24h + 0.35 * med_7d

    # --- Short-term slope (last ~2h) ---
    slope_window = min(24, len(prices))  # ~2h
    y = prices[-slope_window:]
    x = np.arange(len(y))

    try:
        slope, _ = np.polyfit(x, y, 1)
    except Exception:
        slope = 0.0

    slope *= slope_penalty  # dampen optimism

    # --- Reversion strength increases with horizon ---
    reversion_strength = min(1.0, steps / 288)  # full pull by 24h

    forecast = []
    current = prices[-1]

    max_dev = anchor * bound_frac

    for i in range(steps):
        drift = slope
        pull = (anchor - current) * reversion_strength / steps

        next_price = current + drift + pull

        # --- Hard bounds ---
        next_price = max(anchor - max_dev, min(anchor + max_dev, next_price))

        forecast.append(next_price)
        current = next_price

    return forecast

# --- Darts ML Forecasting Wrapper ---
def darts_forecast(prices, steps, model_name='LSTM'):
    try:
        from darts import TimeSeries
        from darts.models import RNNModel, TCNModel, NBEATSModel
        import torch
        import numpy as np
    except ImportError:
        print("darts[torch] is required for ML forecasting. Please install it via 'pip install u8darts[torch]'.")
        return None
    # Defensive: must have enough data
    if prices is None or len(prices) < 10:
        return None
    # Convert to TimeSeries
    series = TimeSeries.from_values(np.array(prices, dtype=np.float32))
    # Ensure input_chunk_length > output_chunk_length
    output_chunk_length = steps
    input_chunk_length = max(output_chunk_length + 1, min(20, len(prices)//2))
    # Model selection
    if model_name == 'LSTM':
        model = RNNModel(model='LSTM', input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_rnn_layers=1, random_state=42, batch_size=16, n_epochs=50, pl_trainer_kwargs={"enable_progress_bar": False})
    elif model_name == 'RNN':
        model = RNNModel(model='RNN', input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_rnn_layers=1, random_state=42, batch_size=16, n_epochs=50, pl_trainer_kwargs={"enable_progress_bar": False})
    elif model_name == 'TCN':
        model = TCNModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=50, random_state=42, batch_size=16, pl_trainer_kwargs={"enable_progress_bar": False})
    elif model_name == 'NBEATS':
        model = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=50, random_state=42, batch_size=16, pl_trainer_kwargs={"enable_progress_bar": False})
    else:
        print(f"Unknown Darts model: {model_name}")
        return None
    try:
        model.fit(series, verbose=False)
        forecast = model.predict(steps)
        return [float(f) for f in forecast.values().flatten()]
    except Exception as e:
        print(f"Darts {model_name} forecast error: {e}")
        return None
def prophet_forecast(hist, steps, interval_minutes):
    try:
        from prophet import Prophet
        import pandas as pd
    except ImportError:
        print("prophet is required for Prophet forecasting. Please install it via 'pip install prophet'.")
        return None
    # Prepare DataFrame with imputation for missing prices
    if not hist or len(hist) < 3:
        return None
    rows = []
    for e in hist:
        if e.get('timestamp'):
            high = e.get('avgHighPrice')
            low = e.get('avgLowPrice')
            # Impute missing prices with available value or skip if both missing
            if high is not None and low is not None:
                price = (high + low) / 2
            elif high is not None:
                price = high
            elif low is not None:
                price = low
            else:
                continue
            rows.append({'ds': e['timestamp'], 'y': price})
    if len(rows) < 3:
        return None
    df = pd.DataFrame(rows)
    df['ds'] = pd.to_datetime(df['ds'])
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq=f'{interval_minutes}min')
    forecast = model.predict(future)
    forecasted = forecast['yhat'][-steps:].values
    return [float(f) for f in forecasted]

def validate_forecast(hist, forecasted, interval_minutes=5):
    """
    Validate forecast accuracy using MAE and RMSE.
    hist: historical price dicts
    forecasted: list of forecasted prices
    interval_minutes: interval between forecast points
    Returns: MAE, RMSE
    """
    import numpy as np
    if not hist or not forecasted:
        return None, None
    n = len(forecasted)
    actual = []
    for e in hist[-n:]:
        high = e.get('avgHighPrice', 0)
        low = e.get('avgLowPrice', 0)
        if high is None:
            high = 0
        if low is None:
            low = 0
        actual.append((high + low) / 2)
    # Replace None in forecasted with 0
    forecasted_clean = [f if f is not None else 0 for f in forecasted]
    if len(actual) != n:
        return None, None
    actual = np.array(actual)
    forecasted_clean = np.array(forecasted_clean)
    mae = np.mean(np.abs(actual - forecasted_clean))
    rmse = np.sqrt(np.mean((actual - forecasted_clean)**2))
    return mae, rmse

def wma(prices, steps = 20, wma_factor = 20, preds = []):
    try:
        if prices is None or len(prices) == 0:
            return preds
        if (steps) == 0:
            return preds
        wma_factor = min(wma_factor, len(prices))
        weights = np.arange(1, wma_factor+1)
        preds.append(math.floor(np.sum(np.array(prices[-wma_factor:]) * weights) / np.sum(weights)))
        prices.append(preds[-1])
        return wma(prices, steps - 1, wma_factor, preds)

    except Exception as e:
        print("WMA forecast error:", e)
        return preds

def ewma(prices, steps = 20, alpha = 0.5, preds = []):
    if (steps) == 0:
        return preds
    ewma_value = prices[0]
    for y in prices[1:]:
        ewma_value = alpha * y + (1 - alpha) * ewma_value
    preds.append(math.floor(ewma_value))
    prices.append(preds[-1])   

    return ewma(prices, steps - 1, alpha, preds )

def linear_regression_forecast(prices, steps=20, preds=[]):
    if (steps) == 0:
        return preds
    xs = np.arange(len(prices))
    coeffs = np.polyfit(xs, prices, 1)
    forecast = math.floor(np.polyval(coeffs, len(prices)))
    preds.append(forecast)
    prices.append(preds[-1])

    return linear_regression_forecast(prices, steps - 1, preds)

def polynomial_regression_forecast(prices, steps=20, degree=2, preds=[]):
    if (steps) == 0:
        return preds
    xs = np.arange(len(prices))
    deg = min(degree, len(prices)-1)
    coeffs = np.polyfit(xs, prices, deg)
    forecast = math.floor(np.polyval(coeffs, len(prices)))
    preds.append(forecast)
    prices.append(preds[-1])

    return polynomial_regression_forecast(prices, steps - 1, degree, preds)


# --- WMA + Trend Forecast ---
def wma_trend(prices, steps=20, wma_factor=20):
    if prices is None or len(prices) == 0:
        return []
    wma_factor = min(wma_factor, len(prices))
    weights = np.arange(1, wma_factor+1)
    # Calculate WMA for last point
    wma_val = np.sum(np.array(prices[-wma_factor:]) * weights) / np.sum(weights)
    # Weighted linear regression for trend (recent points get higher weight)
    xs = np.arange(len(prices))
    ws = np.linspace(0.2, 1.0, len(prices))  # linearly increasing weights, oldest=0.2, newest=1.0
    # Use numpy's polyfit with weights for regression
    coeffs = np.polyfit(xs, prices, 1, w=ws)
    slope = coeffs[0]
    # Forecast: WMA + slope * step
    preds = []
    for i in range(1, steps+1):
        preds.append(float(wma_val + slope * i))
    return preds

# --- Brown's Double Moving Average Forecast ---
def browns_double_moving_average(prices, steps=20):
    if prices is None or len(prices) < 2:
        return []
    alpha = 0.4  # Smoothing parameter, can be tuned
    s1 = [prices[0]]
    s2 = [prices[0]]
    for p in prices[1:]:
        s1.append(alpha * p + (1 - alpha) * s1[-1])
        s2.append(alpha * s1[-1] + (1 - alpha) * s2[-1])
    a = 2 * np.array(s1) - np.array(s2)
    b = (alpha / (1 - alpha)) * (np.array(s1) - np.array(s2))
    last_a = a[-1]
    last_b = b[-1]
    preds = [float(last_a + last_b * i) for i in range(1, steps+1)]
    return preds


# def arima_forecast(prices, steps=20):
#     # Defensive: must have at least 2 data points
#     if prices is None or not hasattr(prices, '__len__') or len(prices) < 2:
#         return [0] * steps
#     try:
#         from statsmodels.tsa.arima.model import ARIMA
#     except ImportError:
#         print("statsmodels is required for ARIMA forecasting. Please install it via 'pip install statsmodels'.")
#         return [0] * steps

#     import warnings
#     from statsmodels.tools.sm_exceptions import ConvergenceWarning
#     try:
#         with warnings.catch_warnings(record=True) as wlist:
#             warnings.simplefilter('always', ConvergenceWarning)
#             model = ARIMA(prices, order=(2,1,2))
#             model_fit = model.fit()
#             # If any ConvergenceWarning, fallback
#             if any(isinstance(w.message, ConvergenceWarning) for w in wlist):
#                 if len(prices) > 0:
#                     return [prices[-1]] * steps
#                 else:
#                     return [0] * steps
#             forecasted = model_fit.forecast(steps=steps)
#             return list(forecasted)
#     except LinAlgError:
#         # Defensive fallback: repeat last value
#         if len(prices) > 0:
#             return [prices[-1]] * steps
#         else:
#             return [0] * steps
#     except Exception as e:
#         # General fallback
#         if len(prices) > 0:
#             return [prices[-1]] * steps
#         else:
#             return [0] * steps

def arima_forecast(prices, steps=20):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("statsmodels is required for ARIMA forecasting. Please install it via 'pip install statsmodels'.")
        return None

    try:
        if len(prices) < 3 or prices is None:
            return None

        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit(method_kwargs={'maxiter':100})
        forecasted = model_fit.forecast(steps=steps)
        return [math.floor(f) for f in forecasted]
    except LinAlgError:
        return None
    except Exception as e:
        return None


def holt_winters_forecast(prices_, steps=20):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import numpy as np
    import math

    prices = np.asarray(prices_, dtype=float)

    prices = prices[np.isfinite(prices)]

    if len(prices) < 6:
        return []
    prices = np.where(prices <= 0, 1e-6, prices)

    seasonal_period = cached_seasonality(tuple(prices))

    # seasonal_period = None

    if seasonal_period is None:
        model = ExponentialSmoothing(
            prices,
            trend='add',
            damped_trend=True,
            seasonal=None,
            initialization_method='estimated'
        )
    else:
        model = ExponentialSmoothing(
            prices,
            trend='add',
            damped_trend=True,
            seasonal='add',
            seasonal_periods=seasonal_period,
            initialization_method='estimated'
        )

        
    model_fit = model.fit(optimized=True)
    forecast = model_fit.forecast(steps)

    # --- ROBUST CLAMP BASELINE ---
    baseline = np.median(prices[-12:])   # spike-resistant

    MAX_UP = 1.20
    MAX_DOWN = 0.80

    upper = baseline * MAX_UP
    lower = baseline * MAX_DOWN

    forecast = np.clip(forecast, lower, upper)

    return np.floor(forecast).astype(int).tolist()

from functools import lru_cache

@lru_cache(maxsize=256)
def cached_seasonality(series):
    return find_seasonality(np.array(series))


def find_seasonality(y, min_period=2, max_period=60):
    from statsmodels.tsa.stattools import acf
    import numpy as np

    y = np.asarray(y)

    if len(y) < max_period * 2:
        return None

    acf_vals = acf(y, nlags=max_period, fft=True)

    # confidence threshold
    conf = 1.96 / np.sqrt(len(y))

    candidates = np.where(acf_vals > conf)[0]
    candidates = candidates[candidates >= min_period]

    if len(candidates) == 0:
        return None

    return int(candidates[0])

def sarimax_forecast(prices, steps=20):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("statsmodels is required for SARIMAX forecasting. Please install it via 'pip install statsmodels'.")
        return None

    try:
        if len(prices) < 4 or prices is None:
            return None
        model = SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        forecasted = model_fit.forecast(steps=steps)
        return [math.floor(f) for f in forecasted]
    except LinAlgError:
        return None
    except Exception as e:
        return None


def robust_ewma(prices, steps, alpha=0.3, window=20, clip_sigma=1.5):
    """
    Robust EWMA with MAD-based clipping and bounded recursion.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        return []

    window_prices = prices[-window:]

    # Robust center & scale
    med = np.median(window_prices)
    mad = np.median(np.abs(window_prices - med)) + 1e-6

    lower = med - clip_sigma * mad
    upper = med + clip_sigma * mad

    # Clip historical prices
    clipped = np.clip(prices, lower, upper)

    # EWMA
    y = clipped[0]
    for x in clipped[1:]:
        y = alpha * x + (1 - alpha) * y

    # Forecast (bounded)
    forecast = []
    for _ in range(steps):
        y = alpha * y + (1 - alpha) * y
        y = min(max(y, lower), upper)
        forecast.append(y)

    return forecast

def median_trend(prices, steps, window=24, max_slope_frac=0.02):
    """
    Median-anchored linear trend forecast with slope clamping.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return []

    window_prices = prices[-window:]
    anchor = np.median(window_prices)

    x = np.arange(len(window_prices))
    slope = np.polyfit(x, window_prices, 1)[0]

    # Clamp slope
    max_slope = max_slope_frac * anchor
    slope = np.clip(slope, -max_slope, max_slope)

    forecast = anchor + slope * np.arange(1, steps + 1)
    return forecast.tolist()


def long_range_forecast(
    prices,
    steps,
    window=96,
    max_slope_frac=0.002,
    bound_frac=0.25
):
    """
    Long-range median-anchored bounded trend forecast.
    Designed to converge and never explode.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return []

    window_prices = prices[-window:] if len(prices) > window else prices

    anchor = np.median(window_prices)

    x = np.arange(len(window_prices))
    slope = np.polyfit(x, window_prices, 1)[0]

    # Very conservative slope
    max_slope = max_slope_frac * anchor
    slope = np.clip(slope, -max_slope, max_slope)

    forecast = anchor + slope * np.arange(1, steps + 1)

    # Hard bounds
    lower = (1 - bound_frac) * anchor
    upper = (1 + bound_frac) * anchor
    forecast = np.clip(forecast, lower, upper)

    return forecast.tolist()

def mix_forecast(prices, minutes_ahead, interval_minutes=5):
    steps = max(1, int(minutes_ahead // interval_minutes))

    if len(prices) < 2:
        return []

    # ---- SHORT RANGE (<= 3h) ----
    if minutes_ahead <= 180:
        return robust_ewma(
            prices,
            steps=steps,
            alpha=0.3,
            window=20,
            clip_sigma=1.5
        )

    # ---- MID RANGE (<= 24h) ----
    elif minutes_ahead <= 1440:
        return median_reversion_forecast(
            prices,
            steps=steps,
            window=288,        # 24h @ 5m
            slope_penalty=0.5, # dampen, don't project
            bound_frac=0.10
        )

    # ---- LONG RANGE (> 24h) ----
    else:
        return long_range_forecast(
            prices,
            steps=steps,
            window=96,
            max_slope_frac=0.001,  # reduced
            bound_frac=0.20
        )

def median_reversion_forecast(
    prices,
    steps,
    window=288,          # 24h @ 5m
    alpha=0.25,          # EWMA smoothing
    slope_penalty=0.5,   # how hard downtrends reduce upside
    bound_frac=0.10      # +/- % bounds around median
):
    import numpy as np

    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]

    if len(prices) < 5:
        return []

    # --- rolling window ---
    w = min(window, len(prices))
    recent = prices[-w:]

    # --- median anchor ---
    median = np.median(recent)

    # --- robust EWMA baseline ---
    ewma = recent[0]
    for p in recent[1:]:
        ewma = alpha * p + (1 - alpha) * ewma

    base = 0.7 * ewma + 0.3 * median

    # --- slope (linear regression, normalized) ---
    x = np.arange(w)
    slope = np.polyfit(x, recent, 1)[0]
    slope_norm = slope / max(median, 1e-6)

    # --- slope penalty (never boost, only dampen) ---
    if slope_norm < 0:
        base *= max(0.85, 1 + slope_norm * slope_penalty)

    # --- bounds around median ---
    lower = median * (1 - bound_frac)
    upper = median * (1 + bound_frac)
    base = np.clip(base, lower, upper)

    # --- flat forecast path (mean reversion assumption) ---
    forecast = np.full(steps, base)

    return forecast.tolist()


if __name__ == "__main__":
    # Example usage
    hist = [
        {'avgHighPrice': 100, 'avgHighVolume': 10, 'avgLowPrice': 90, 'avgLowVolume': 10},
        {'avgHighPrice': 110, 'avgHighVolume': 10, 'avgLowPrice': 100, 'avgLowVolume': 10},
        {'avgHighPrice': 120, 'avgHighVolume': 10, 'avgLowPrice': 110, 'avgLowVolume': 10},
        {'avgHighPrice': 130, 'avgHighVolume': 10, 'avgLowPrice': 120, 'avgLowVolume': 10},
    ]
    forecasted_price = forecast_avg_price(hist, minutes_ahead=30, strategy='wma')
    print(f'Forecasted Price for WMA: {forecasted_price}')

    forecasted_price = forecast_avg_price(hist, minutes_ahead=30, strategy='ewma') 
    print(f'Forecasted Price for EWMA: {forecasted_price}')

    forecasted_price = forecast_avg_price(hist, minutes_ahead=30, strategy='linear')
    print(f'Forecasted Price for Linear Regression: {forecasted_price}')    

    forecasted_price = forecast_avg_price(hist, minutes_ahead=30, strategy='poly')
    print(f'Forecasted Price for Polynomial Regression: {forecasted_price}')