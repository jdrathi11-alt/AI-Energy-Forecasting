# Model and evaluation card

## Task
Forecast all 24 next-day hourly energy values at the end of the last observed 23:00 hour. Units are kWh per hourly interval. A direct multi-output model receives one row per forecast day and outputs 24 values. No predictions are recursively fed back into lag features.

## Features
Latest known consumption and lags 1, 2, 23, 24, 47, 167 hours; 24-hour and 168-hour means/std; next-day cyclic day-of-week and annual calendar encodings; weekend indicator. All energy features are computed from at most the 168 hours ending at the origin. Because origins are always 23:00, a separate origin hour feature is unnecessary: each output position already identifies a forecast hour.

## Cleaning
Duplicate hourly readings are averaged. Invalid/negative/infinite values become missing. Forward fill uses only past values and is limited to two hours. Long gaps remain missing. Windows with missing features are excluded. Target days with any imputed/unobserved value are excluded, so reported accuracy is conditional on eligible complete days. Report dropped days/data coverage in any measured-data extension.

## Evaluation
Chronological 60% train / 20% validation / 20% test split of eligible daily origins. Target windows cannot overlap the next split's origin. Ridge scaling is fit inside the training pipeline only. Ridge and Random Forest are trained once on the training partition. Previous-day and previous-week seasonal-naive forecasts use only available historical measurements.

Select the winner using lowest VALIDATION RMSE. Do not choose a model or tune settings using test scores. Test is daily rolling-origin: earlier observations from the test interval may legitimately become known inputs on later test days. This is not a single year-long forecast issued before the test starts. Estimators remain fixed throughout evaluation.

The winner may be a naive baseline. That is a valid result, not a reason to hide the baseline. RMSE, MAE, R2, per-horizon metrics and RMSE improvement versus weekly naive are saved. Overall scores flatten hourly outputs; horizon-level scores are saved separately. Negative R2 or negative baseline improvement are possible and must be reported honestly. Avoid percentage 'accuracy' derived from R2.

`evaluation_model.joblib` contains train-only parameters used for reported scores. AFTER frozen test scoring, `production_model.joblib` refits the validation-selected algorithm on all eligible labeled days for future use. Production refit performance is not the reported holdout performance.

## Intended use and limitations
Educational offline decision support, not automatic equipment control. Synthetic-run accuracy does not demonstrate a real deployment. Real data requires longer history, season-specific audits, holidays, known-in-advance weather forecasts, uncertainty intervals and operational constraints. The tariff scenario assumes verified flexible load, zero losses, a per-hour transfer cap and sufficient receiving capacity. It does not model demand charges or verify equipment feasibility. Its bill difference is hypothetical, not measured savings.

No claim of a web service, online deployment, GPU training or corporate integration is made. Only load trusted joblib files; pickle-based model artifacts can execute arbitrary code.
