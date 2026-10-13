"""Run with python -m unittest discover -s tests -v."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import generate_data, clean_data, load_hourly
from src.features import origin_features, supervised_days, chronological_split
from src.simulation import simulate


class PipelineTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = generate_data(Path(directory) / 'sample.csv', days=100)
        self.clean, self.quality = clean_data(raw)

    def test_duplicate_and_gaps(self):
        self.assertEqual(self.quality['duplicate_rows'], 1)
        self.assertEqual(len(self.clean), 2400)
        self.assertGreater(self.quality['imputed_hours'], 0)
        self.assertTrue((self.clean.energy_kwh.dropna() >= 0).all())

    def test_future_cannot_change_origin_features(self):
        origin = self.clean.index[500]
        original = origin_features(self.clean.energy_kwh, origin)
        altered = self.clean.energy_kwh.copy()
        altered.loc[altered.index > origin] = 999999
        self.assertEqual(original, origin_features(altered, origin))

    def test_chronology_and_shape(self):
        X, y = supervised_days(self.clean)
        Xt, yt, Xv, yv, Xtest, ytest = chronological_split(X, y)
        self.assertEqual(y.shape[1], 24)
        self.assertLessEqual(Xt.index[-1] + pd.Timedelta(hours=24), Xv.index[0])
        self.assertLessEqual(Xv.index[-1] + pd.Timedelta(hours=24), Xtest.index[0])
        for origin in X.index:
            times = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq='h')
            self.assertTrue(self.clean.observed.reindex(times).all())

    def test_fill_is_causal_and_limited(self):
        frame = pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=6, freq='h'),
                              'energy_kwh': [10, None, None, None, None, 90]})
        result, report = clean_data(frame)
        self.assertEqual(result.energy_kwh.iloc[1], 10)
        self.assertTrue(pd.isna(result.energy_kwh.iloc[3]))
        self.assertFalse(result.observed.iloc[1])

    def test_reproducibility(self):
        with tempfile.TemporaryDirectory() as d:
            a = generate_data(Path(d) / 'a.csv', 100)
            b = generate_data(Path(d) / 'b.csv', 100)
        pd.testing.assert_frame_equal(a, b)

    def test_invalid_values(self):
        raw = pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=4, freq='h'),
                            'energy_kwh': [10, -1, float('inf'), 11]})
        result, report = clean_data(raw)
        self.assertEqual(report['imputed_hours'], 2)
        self.assertEqual(result.energy_kwh.iloc[1], 10)

    def test_uci_units_and_partial_hour(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'uci.txt'
            timestamps = pd.date_range('2025-01-01', periods=90, freq='min')
            raw = pd.DataFrame({'Date': timestamps.strftime('%d/%m/%Y'),
                                'Time': timestamps.strftime('%H:%M:%S'),
                                'Global_active_power': 2.0})
            raw.to_csv(path, sep=';', index=False)
            result = load_hourly(path, uci=True)
        self.assertAlmostEqual(result.energy_kwh.iloc[0], 2.0)
        self.assertTrue(pd.isna(result.energy_kwh.iloc[1]))

    def test_simulation_conserves_energy(self):
        forecast = pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=24, freq='h'),
                                 'predicted_kwh': np.full(24, 40.0)})
        schedule, report = simulate(forecast)
        self.assertAlmostEqual(schedule.before_kwh.sum(), schedule.after_kwh.sum())
        self.assertAlmostEqual(report['modeled_bill_difference_usd'], 1.5)
        self.assertTrue((schedule.after_kwh >= 0).all())


if __name__ == '__main__':
    unittest.main()
