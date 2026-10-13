"""Optional measured-data experiment; does not overwrite synthetic artifacts.
Run after extracting the official UCI TXT file:
python public_experiment.py --file data/raw/household_power_consumption.txt
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from src.data import load_hourly, clean_data
from src.features import supervised_days
from src.model import train_evaluate


def run(path):
    root = Path(__file__).resolve().parent
    folder = root / 'outputs/public_uci'
    folder.mkdir(parents=True, exist_ok=True)
    clean, quality = clean_data(load_hourly(path, uci=True))
    X, y = supervised_days(clean)
    _, _, report, validation, test, predictions, horizons = train_evaluate(X, y, clean)
    report['provenance'] = {'source': 'UCI measured household data',
                            'doi': '10.24432/C58K54', 'license': 'CC BY 4.0',
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'python': platform.python_version(), 'numpy': np.__version__,
                            'pandas': pd.__version__, 'sklearn': sklearn.__version__}
    for name, frame in [('validation_metrics', validation), ('test_metrics', test),
                         ('test_predictions', predictions), ('horizon_metrics', horizons)]:
        frame.to_csv(folder / f'{name}.csv', index=False)
    for name, value in [('metrics', report), ('data_quality', quality)]:
        (folder / f'{name}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({'quality': quality, 'report': report}, indent=2))
    print(f'Real-data experiment saved separately in {folder}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=Path, required=True)
    run(parser.parse_args().file)
