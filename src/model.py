"""Validation-only model selection, locked test, then production refit."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import chronological_split, seasonal_predictions


def metrics(actual, predicted):
    a, p = np.asarray(actual).ravel(), np.asarray(predicted).ravel()
    return {'rmse_kwh': float(np.sqrt(mean_squared_error(a, p))),
            'mae_kwh': float(mean_absolute_error(a, p)),
            'r2': float(r2_score(a, p))}


def predict_bundle(bundle, X, clean):
    if bundle['name'] in ('previous_day', 'previous_week'):
        period = 24 if bundle['name'] == 'previous_day' else 168
        return seasonal_predictions(clean, X.index, period)
    return np.maximum(bundle['estimator'].predict(X[bundle['features']]), 0)


def train_evaluate(X, y, clean):
    Xt, yt, Xv, yv, Xtest, ytest = chronological_split(X, y)
    candidates = {
        'previous_day': None,
        'previous_week': None,
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'random_forest': RandomForestRegressor(n_estimators=150, max_depth=10,
                                               min_samples_leaf=2, random_state=42, n_jobs=-1)
    }
    bundles, rows = {}, []
    for name, estimator in candidates.items():
        if estimator is not None:
            estimator.fit(Xt, yt)
        bundle = {'name': name, 'estimator': estimator, 'features': list(X.columns)}
        bundles[name] = bundle
        rows.append({'model': name, **metrics(yv, predict_bundle(bundle, Xv, clean))})
    # Test data has not been consulted at this point.
    validation = pd.DataFrame(rows).sort_values('rmse_kwh').reset_index(drop=True)
    winner = validation.iloc[0]['model']
    chosen = bundles[winner]
    predicted = predict_bundle(chosen, Xtest, clean)
    test_rows = [{'model': name, **metrics(ytest, predict_bundle(b, Xtest, clean))}
                 for name, b in bundles.items()]
    test = pd.DataFrame(test_rows)
    baseline_rmse = float(test.loc[test.model == 'previous_week', 'rmse_kwh'].iloc[0])
    chosen_rmse = metrics(ytest, predicted)['rmse_kwh']
    candidate_origins = clean.index[(clean.index.hour == 23)
                                    & (clean.index >= clean.index[0] + pd.Timedelta(hours=167))
                                    & (clean.index + pd.Timedelta(hours=24) <= clean.index[-1])]
    report = {'selected_model': winner, 'selection_rule': 'lowest validation RMSE',
              'candidate_forecast_days': len(candidate_origins),
              'eligible_forecast_days': len(X),
              'excluded_forecast_days': len(candidate_origins) - len(X),
              'train_days': len(Xt), 'validation_days': len(Xv), 'test_days': len(Xtest),
              'train_last_target': str(Xt.index[-1] + pd.Timedelta(hours=24)),
              'validation_first_origin': str(Xv.index[0]),
              'test_first_origin': str(Xtest.index[0]),
              'test_metrics': metrics(ytest, predicted),
              'rmse_improvement_vs_week_percent': (100 * (baseline_rmse - chosen_rmse) / baseline_rmse
                                                   if baseline_rmse > 0 else None),
              'evaluation_protocol': 'Daily rolling origin; measured history through 23:00; 24-hour direct forecast; fixed train-only estimators.',
              'target_unit': 'kWh per hour', 'forecast_horizon_hours': 24}
    # Only AFTER scoring the untouched test period: refit for future use.
    production = dict(chosen)
    if chosen['estimator'] is not None:
        production['estimator'] = clone(chosen['estimator']).fit(X, y)
    production['training_last_target'] = str(X.index[-1] + pd.Timedelta(hours=24))
    production['purpose'] = 'Future prediction only; do not use to reproduce holdout metrics.'
    records = []
    for i, origin in enumerate(Xtest.index):
        for h in range(24):
            records.append({'origin': origin, 'timestamp': origin + pd.Timedelta(hours=h + 1),
                            'horizon_hour': h + 1, 'actual_kwh': ytest[i, h],
                            'predicted_kwh': predicted[i, h], 'model': winner})
    by_horizon = pd.DataFrame([{'horizon_hour': h + 1, **metrics(ytest[:, h], predicted[:, h])}
                               for h in range(24)])
    return production, chosen, report, validation, test, pd.DataFrame(records), by_horizon
