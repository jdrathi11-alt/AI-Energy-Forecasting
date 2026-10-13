"""Run: python main.py --mode all (or train / forecast)."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from src.data import generate_data, load_hourly, clean_data
from src.features import supervised_days, origin_features
from src.model import train_evaluate, predict_bundle
from src.simulation import simulate
from src.plots import save_plots

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run(args):
    for name in ('data/raw', 'data/processed', 'models', 'outputs', 'images'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    if args.csv and args.uci_file:
        raise ValueError('Choose --csv OR --uci-file, not both.')
    source = 'uci' if args.uci_file else ('csv' if args.csv else 'synthetic')
    path = Path(args.uci_file or args.csv or ROOT / 'data/raw/synthetic_energy.csv')
    if source == 'synthetic' and (args.mode != 'forecast' or not path.exists()):
        generate_data(path, args.days)
    raw = load_hourly(path, uci=(source == 'uci'))
    clean, quality = clean_data(raw)
    clean.to_csv(ROOT / 'data/processed/hourly_energy.csv')
    raw.head(12).to_csv(ROOT / 'outputs/dataset_preview.csv', index=False)
    write_json(ROOT / 'outputs/data_quality.json', quality)
    print('DATA QUALITY:', json.dumps(quality, indent=2))
    if args.mode in ('all', 'train'):
        X, y = supervised_days(clean)
        X.head(10).to_csv(ROOT / 'outputs/feature_preview.csv')
        production, evaluation, report, validation, test, predictions, horizons = train_evaluate(X, y, clean)
        provenance = {'source': source, 'dataset_path': str(path),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'seed': 42 if source == 'synthetic' else None,
                      'python': platform.python_version(), 'numpy': np.__version__,
                      'pandas': pd.__version__, 'sklearn': sklearn.__version__}
        report['provenance'] = provenance
        for bundle in (production, evaluation):
            bundle['source'] = source
            bundle['unit'] = 'hourly_kwh'
        joblib.dump(production, ROOT / 'models/production_model.joblib')
        joblib.dump(evaluation, ROOT / 'models/evaluation_model.joblib')
        validation.to_csv(ROOT / 'outputs/validation_metrics.csv', index=False)
        test.to_csv(ROOT / 'outputs/test_metrics.csv', index=False)
        predictions.to_csv(ROOT / 'outputs/test_predictions.csv', index=False)
        horizons.to_csv(ROOT / 'outputs/horizon_metrics.csv', index=False)
        write_json(ROOT / 'outputs/metrics.json', report)
        print('\nVALIDATION COMPARISON:\n', validation.to_string(index=False))
        print('\nLOCKED TEST REPORT:\n', json.dumps(report, indent=2))
    if args.mode in ('all', 'forecast'):
        model_path = ROOT / 'models/production_model.joblib'
        if not model_path.exists():
            raise ValueError('No model found. First run python main.py --mode train.')
        # Never load joblib files from untrusted people: deserialization can execute code.
        bundle = joblib.load(model_path)
        if bundle['source'] != source:
            raise ValueError('Model and input source differ. Train on this source first.')
        origin = clean.index[-1]
        if origin.hour != 23:
            raise ValueError('Forecast requires history ending at 23:00. Trim partial final day or use --mode train.')
        if origin + pd.Timedelta(hours=1) <= pd.Timestamp(bundle['training_last_target']):
            raise ValueError('Forecast target must follow the production training period.')
        Xfuture = pd.DataFrame([origin_features(clean.energy_kwh, origin)], index=[origin])
        values = predict_bundle(bundle, Xfuture, clean)[0]
        forecast = pd.DataFrame({'timestamp': pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq='h'),
                                 'predicted_kwh': values, 'model': bundle['name']})
        forecast.to_csv(ROOT / 'outputs/next_day_forecast.csv', index=False)
        scenario, summary = simulate(forecast)
        scenario.to_csv(ROOT / 'outputs/load_shift_scenario.csv', index=False)
        write_json(ROOT / 'outputs/simulation_summary.json', summary)
        predictions = pd.read_csv(ROOT / 'outputs/test_predictions.csv', parse_dates=['timestamp'])
        validation = pd.read_csv(ROOT / 'outputs/validation_metrics.csv')
        save_plots(clean, predictions, validation, forecast, scenario, ROOT / 'images')
        print('\nNEXT-DAY FORECAST:\n', forecast.head().to_string(index=False))
        print('\nSIMULATION:\n', json.dumps(summary, indent=2))
    print('\nArtifacts saved in outputs/, images/, and models/.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct 24-hour energy forecasting student project')
    parser.add_argument('--mode', choices=['all', 'train', 'forecast'], default='all')
    parser.add_argument('--days', type=int, default=365, help='Synthetic history length, >=90 recommended')
    parser.add_argument('--csv', help='Hourly CSV with timestamp,energy_kwh')
    parser.add_argument('--uci-file', help='Extracted household_power_consumption.txt')
    arguments = parser.parse_args()
    try:
        run(arguments)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
