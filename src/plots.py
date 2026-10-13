"""Save portable proof assets; works without a desktop display."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_plots(clean, predictions, validation, forecast, scenario, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / name, dpi=160, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(clean.index[-168:], clean.energy_kwh.iloc[-168:], color='#16697a')
    ax.set(title='Historical consumption — final seven days', ylabel='Energy (kWh/hour)', xlabel='Timestamp')
    save('01_historical_consumption.png')

    fig, ax = plt.subplots(figsize=(11, 4))
    subset = predictions.tail(168)
    ax.plot(subset.timestamp, subset.actual_kwh, label='Actual')
    ax.plot(subset.timestamp, subset.predicted_kwh, label='24-hour forecast', alpha=.8)
    ax.set(title='Untouched test: final seven scored days', ylabel='Energy (kWh/hour)', xlabel='Timestamp')
    ax.legend()
    save('02_actual_vs_predicted.png')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(predictions.actual_kwh, predictions.predicted_kwh, s=6, alpha=.35)
    limits = [0, max(predictions.actual_kwh.max(), predictions.predicted_kwh.max()) * 1.05]
    ax.plot(limits, limits, '--', color='black')
    ax.set(title='Holdout prediction agreement', xlabel='Actual (kWh/hour)', ylabel='Predicted (kWh/hour)')
    save('03_prediction_scatter.png')

    fig, ax = plt.subplots(figsize=(8, 4))
    residual = predictions.actual_kwh - predictions.predicted_kwh
    ax.hist(residual, bins=35, color='#489fb5')
    ax.set(title='Holdout residuals', xlabel='Actual minus forecast (kWh/hour)', ylabel='Count')
    save('04_residual_distribution.png')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(validation.model, validation.rmse_kwh, color='#16697a')
    ax.set(title='Model selection — validation only', ylabel='RMSE (kWh/hour)')
    save('05_model_comparison.png')

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(forecast.timestamp, forecast.predicted_kwh, marker='o', color='#ef8354')
    ax.set(title='Next-day future forecast — not observed truth', ylabel='Energy (kWh/hour)', xlabel='Timestamp')
    save('06_next_day_forecast.png')

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(scenario.timestamp, scenario.before_kwh, label='Forecast load before')
    ax.plot(scenario.timestamp, scenario.after_kwh, label='Idealized shifted load')
    ax.set(title='Illustrative scheduling scenario — not realized savings', ylabel='Energy (kWh/hour)', xlabel='Timestamp')
    ax.legend()
    save('07_load_shift_simulation.png')
