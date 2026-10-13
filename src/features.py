"""Direct next-day features: all load features stop at forecast origin."""
import numpy as np
import pandas as pd

HORIZON = 24
LOOKBACK = 168


def origin_features(s: pd.Series, origin: pd.Timestamp):
    history = s.loc[origin - pd.Timedelta(hours=LOOKBACK - 1):origin]
    if len(history) != LOOKBACK or history.isna().any():
        raise ValueError(f'Need 168 consecutive usable hours ending at {origin}.')
    x = {f'lag_{lag}': float(history.iloc[-1 - lag]) for lag in (0, 1, 2, 23, 24, 47, 167)}
    for window in (24, 168):
        x[f'mean_{window}'] = float(history.iloc[-window:].mean())
        x[f'std_{window}'] = float(history.iloc[-window:].std())
    # The next day's calendar is known when making a forecast.
    tomorrow = origin + pd.Timedelta(hours=1)
    x['dow_sin'] = np.sin(2 * np.pi * tomorrow.dayofweek / 7)
    x['dow_cos'] = np.cos(2 * np.pi * tomorrow.dayofweek / 7)
    x['year_sin'] = np.sin(2 * np.pi * tomorrow.dayofyear / 365.25)
    x['year_cos'] = np.cos(2 * np.pi * tomorrow.dayofyear / 365.25)
    x['weekend'] = float(tomorrow.dayofweek >= 5)
    return x


def supervised_days(clean: pd.DataFrame):
    xs, ys, origins = [], [], []
    s = clean['energy_kwh']
    # Forecast at 23:00, after that hour is measured; predict 00:00..23:00.
    for origin in s.index[s.index.hour == 23]:
        target_times = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq='h')
        target = s.reindex(target_times)
        observed = clean['observed'].reindex(target_times, fill_value=False)
        # Do not train or score on invented/imputed labels.
        if target.isna().any() or not observed.all():
            continue
        try:
            features = origin_features(s, origin)
        except ValueError:
            continue
        xs.append(features)
        ys.append(target.to_numpy())
        origins.append(origin)
    if len(xs) < 60:
        raise ValueError(f'Only {len(xs)} valid days; need >=60 (prefer >=180).')
    index = pd.DatetimeIndex(origins, name='origin')
    return pd.DataFrame(xs, index=index), np.asarray(ys)


def chronological_split(X, y):
    n = len(X)
    a, b = int(n * .6), int(n * .8)
    # Assert that training target windows finish before the next origin.
    assert X.index[a - 1] + pd.Timedelta(hours=24) <= X.index[a]
    assert X.index[b - 1] + pd.Timedelta(hours=24) <= X.index[b]
    return (X.iloc[:a], y[:a], X.iloc[a:b], y[a:b], X.iloc[b:], y[b:])


def seasonal_predictions(clean, origins, period):
    rows = []
    for origin in origins:
        future = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq='h')
        previous = future - pd.Timedelta(hours=period)
        rows.append(clean['energy_kwh'].reindex(previous).to_numpy())
    return np.maximum(np.asarray(rows), 0)
