"""Reproducible virtual facility and causal hourly data cleaning."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_data(path: Path, days: int = 365, seed: int = 42):
    """Hourly kWh; entirely synthetic, not corporate measurements."""
    if days < 90:
        raise ValueError('Use at least 90 synthetic days; 365 is recommended.')
    rng = np.random.default_rng(seed)
    t = pd.date_range('2025-01-01', periods=days * 24, freq='h')
    hour = t.hour.to_numpy()
    work = ((hour >= 8) & (hour < 18)).astype(float)
    weekday = (t.dayofweek < 5).astype(float)
    seasonal = 8 * np.cos(2 * np.pi * t.dayofyear.to_numpy() / 365)
    energy = (35 + 22 * work * (0.55 + 0.45 * weekday)
              + 6 * np.sin(2 * np.pi * (hour - 6) / 24)
              + seasonal + rng.normal(0, 2, len(t)))
    energy = np.maximum(energy, 0)
    # Deliberately introduce isolated sensor gaps and one duplicate.
    energy[rng.choice(np.arange(200, len(t) - 24), 20, replace=False)] = np.nan
    frame = pd.DataFrame({'timestamp': t, 'energy_kwh': energy})
    frame = pd.concat([frame, frame.iloc[[100]]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_hourly(path: Path, uci: bool = False):
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')
    if not uci:
        return pd.read_csv(path)
    # Global_active_power is minute-averaged kW, NOT kWh.
    raw = pd.read_csv(path, sep=';', usecols=['Date', 'Time', 'Global_active_power'],
                      na_values=['?', ''], low_memory=False)
    t = pd.to_datetime(raw['Date'] + ' ' + raw['Time'],
                       format='%d/%m/%Y %H:%M:%S', errors='coerce')
    power = pd.to_numeric(raw['Global_active_power'], errors='coerce')
    s = pd.Series(power.to_numpy(), index=t).loc[lambda x: x.index.notna()]
    s = s.where(s >= 0).groupby(level=0).mean().sort_index()
    # Require ALL 60 minutes. A missing/partial hour remains a gap.
    energy = (s.resample('h').sum(min_count=60) / 60).rename('energy_kwh')
    return energy.rename_axis('timestamp').reset_index()


def clean_data(raw: pd.DataFrame):
    required = {'timestamp', 'energy_kwh'}
    if not required.issubset(raw.columns):
        raise ValueError('CSV must contain timestamp and energy_kwh columns.')
    df = raw[['timestamp', 'energy_kwh']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if isinstance(df['timestamp'].dtype, pd.DatetimeTZDtype):
        raise ValueError('Use consistent timezone-naive hourly timestamps for this demo.')
    invalid_time = int(df['timestamp'].isna().sum())
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        raise ValueError('No valid timestamps.')
    if not (df['timestamp'] == df['timestamp'].dt.floor('h')).all():
        raise ValueError('Input must already be hourly. Use --uci-file for UCI minute data.')
    df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce')
    invalid_values = (~np.isfinite(df['energy_kwh']) | (df['energy_kwh'] < 0))
    df.loc[invalid_values, 'energy_kwh'] = np.nan
    duplicate_rows = int(df.duplicated('timestamp').sum())
    # Duplicate readings are averaged, not summed: avoids double billing.
    series = df.groupby('timestamp')['energy_kwh'].mean().sort_index()
    series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq='h'))
    observed = series.notna()
    before = int(series.isna().sum())
    # Only past readings fill gaps. Never interpolate across train/test boundaries.
    filled = series.ffill(limit=2)
    result = pd.DataFrame({'energy_kwh': filled, 'observed': observed})
    result.index.name = 'timestamp'
    report = {'input_rows': len(raw), 'hourly_rows': len(result),
              'invalid_timestamps': invalid_time, 'duplicate_rows': duplicate_rows,
              'missing_before_fill': before, 'imputed_hours': int((~observed & filled.notna()).sum()),
              'remaining_missing': int(filled.isna().sum()),
              'first_timestamp': str(result.index.min()), 'last_timestamp': str(result.index.max())}
    return result, report
