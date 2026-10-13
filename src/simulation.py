"""Illustrative time-of-use bill and idealized flexible-load scheduling."""
import numpy as np
import pandas as pd


def simulate(forecast, flexible_kwh=10.0, max_shift_per_hour=3.0):
    df = forecast.copy()
    # Currency is explicitly illustrative USD; NOT a real utility tariff.
    df['tariff_usd_per_kwh'] = np.where(df.timestamp.dt.hour.between(16, 21), .25, .10)
    df['before_kwh'] = df.predicted_kwh
    df['after_kwh'] = df.before_kwh.copy()
    peak = df.index[df.tariff_usd_per_kwh == .25]
    offpeak = df.index[df.tariff_usd_per_kwh == .10]
    remaining, moved = flexible_kwh, 0.0
    # Assumption: this much load is truly deferrable and can move within this day.
    for i in peak:
        amount = min(max_shift_per_hour, float(df.at[i, 'after_kwh']), remaining)
        df.at[i, 'after_kwh'] -= amount
        remaining -= amount
        moved += amount
    remaining = moved
    for i in offpeak:
        amount = min(max_shift_per_hour, remaining)
        df.at[i, 'after_kwh'] += amount
        remaining -= amount
    if remaining > 1e-8:
        raise ValueError('Insufficient receiving capacity for proposed load shift.')
    assert np.isclose(df.before_kwh.sum(), df.after_kwh.sum())
    before = float((df.before_kwh * df.tariff_usd_per_kwh).sum())
    after = float((df.after_kwh * df.tariff_usd_per_kwh).sum())
    report = {'scenario': 'Illustrative USD tariff, idealized deferrable-load scenario',
              'requested_shift_kwh': flexible_kwh, 'actual_shift_kwh': moved,
              'max_shift_per_hour_kwh': max_shift_per_hour,
              'total_energy_kwh': float(df.before_kwh.sum()),
              'forecast_bill_before_usd': before, 'forecast_bill_after_usd': after,
              'modeled_bill_difference_usd': before - after,
              'not_realized_savings': True,
              'limitations': 'No equipment schedules, losses, demand charges, capacity verification, or real business cost validation.'}
    return df, report
