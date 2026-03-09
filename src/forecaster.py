import pandas as pd
from prophet import Prophet
import os
import pickle
import numpy as np

# Paths relative to project root
MODEL_PATH = 'models/prophet_model.pkl'
DATA_PATH = 'data/processed/daily_load.parquet'


def train_model():
    """
    Trains a Prophet model on daily mean load data and saves it as a pickle file.

    Key decisions:
    - Train on 2005-2016 (11 years): enough cycles for Prophet to reliably
      estimate yearly seasonality without overfitting to recent behavior.
    - additive seasonality: PJM East load is relatively stable in level,
      so seasonal swings don't scale with trend — additive is appropriate.
    - fourier_order=6: captures the bimodal summer/winter peaks cleanly
      without overfitting to noise in a shorter window.
    - changepoint_range=0.80 (default): keeps late-training changepoints
      from extrapolating aggressively into 2017+.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run your data processing notebook first.")
        return

    df = pd.read_parquet(DATA_PATH)

    if 'date' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'date', 'Datetime': 'date'})

    prophet_df = df[['date', 'daily_mean_load']].rename(columns={
        'date': 'ds',
        'daily_mean_load': 'y'
    })

    # 11 years of training data — critical fix from 3-year window
    train = prophet_df[
    (prophet_df['ds'] >= '2002-01-01') &
    (prophet_df['ds'] < '2015-01-01')
    ]

    print(f"Training on {len(train)} days ({train['ds'].min().date()} to {train['ds'].max().date()})")

    model = Prophet(
    growth='flat',                      # eliminates trend extrapolation error
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,       # irrelevant for flat growth but harmless
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
    )

    model.add_seasonality(name='yearly', period=365.25, fourier_order=6)
    model.add_country_holidays(country_name='US')

    print("Training Prophet model (this may take ~30 seconds)...")
    model.fit(train)

    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved to {MODEL_PATH}")


def evaluate_model():
    """
    Computes MAPE and RMSE on the validation set (2017 onwards).
    Also prints a breakdown of the first 90 days vs the rest to spot
    any early-forecast drift from the final training changepoint.
    """
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run train_model() first.")
        return None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_parquet(DATA_PATH)

    if 'date' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'date', 'Datetime': 'date'})

    actuals = df[df['date'] >= '2015-01-01'][['date', 'daily_mean_load']].copy()
    actuals = actuals.rename(columns={'date': 'ds', 'daily_mean_load': 'y'})

    forecast_df = model.predict(actuals[['ds']])
    performance = forecast_df[['ds', 'yhat']].merge(actuals, on='ds')

    y_true = performance['y'].values
    y_pred = performance['yhat'].values

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"\n--- Model Evaluation (Post-2015 period) ---")
    print(f"MAPE : {mape:.2f}%")
    print(f"RMSE : {rmse:.2f} MW")

    # First 90 days breakdown — catches changepoint extrapolation drift
    early = performance.iloc[:90]
    y_true_e = early['y'].values
    y_pred_e = early['yhat'].values
    mape_early = np.mean(np.abs((y_true_e - y_pred_e) / y_true_e)) * 100
    print(f"\n--- First 90 days (Jan–Mar 2017) ---")
    print(f"MAPE : {mape_early:.2f}%")
    print("(If this is significantly higher than the full-period MAPE,")
    print(" the model is drifting at the training boundary — lower changepoint_prior_scale)")

    if mape < 5:
        print(f"\n✅ Target achieved: MAPE is {mape:.2f}%, under 5%")
    else:
        print(f"\n⚠️  MAPE is {mape:.2f}%. See troubleshooting notes below.")
        _print_troubleshooting(mape, mape_early)

    return mape


def _print_troubleshooting(mape: float, mape_early: float):
    """Prints targeted advice based on the failure mode observed."""
    print("\n--- Troubleshooting ---")
    if mape_early > mape * 1.5:
        print("• Early-period drift detected: the model is extrapolating a late")
        print("  changepoint past the training boundary. The changepoint_range is")
        print("  likely too high. Ensure it is at the default (0.80) — do not set")
        print("  it to 0.95 or higher.")
    if mape > 8:
        print("• MAPE above 8% usually means insufficient training data.")
        print("  Confirm your training window starts at 2005-01-01 and ends")
        print("  at 2016-12-31. Check len(train) — should be ~4,000 rows.")
    if 5 <= mape <= 8:
        print("• MAPE in 5–8% range: check your yearly seasonality component.")
        print("  Run model.plot_components(forecast_df) and inspect the yearly")
        print("  curve. It should show exactly two peaks (Jul/Aug and Jan) and")
        print("  two troughs (Apr/May and Oct/Nov). If it looks jagged, lower")
        print("  fourier_order from 6 to 4.")


def forecast(days_ahead: int) -> dict:
    """
    Loads the saved model and returns a structured dictionary for the LangGraph agent.

    Args:
        days_ahead: number of days to forecast from the end of training data

    Returns:
        dict with keys:
            forecast  — list of {date, predicted_mw, lower_mw, upper_mw}
            summary   — plain-English summary string
            peak_date — date string of highest predicted load
            trough_date — date string of lowest predicted load
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model() first."
        )

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    future = model.make_future_dataframe(periods=days_ahead)
    forecast_df = model.predict(future)

    predictions = forecast_df.tail(days_ahead).copy()

    records = []
    for _, row in predictions.iterrows():
        records.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_mw": round(row['yhat'], 2),
            "lower_mw": round(row['yhat_lower'], 2),
            "upper_mw": round(row['yhat_upper'], 2)
        })

    peak_row = predictions.loc[predictions['yhat'].idxmax()]
    trough_row = predictions.loc[predictions['yhat'].idxmin()]

    summary = (
        f"Over the next {days_ahead} days, load is forecast to average "
        f"{predictions['yhat'].mean():.0f} MW. "
        f"Peak demand of {peak_row['yhat']:.0f} MW is expected on "
        f"{peak_row['ds'].strftime('%A, %b %d')}. "
        f"Lowest demand of {trough_row['yhat']:.0f} MW is expected on "
        f"{trough_row['ds'].strftime('%A, %b %d')}."
    )

    return {
        "forecast": records,
        "summary": summary,
        "peak_date": peak_row['ds'].strftime('%Y-%m-%d'),
        "trough_date": trough_row['ds'].strftime('%Y-%m-%d')
    }

def diagnose_level():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_parquet(DATA_PATH)
    if 'date' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'date', 'Datetime': 'date'})

    actuals = df[df['date'] >= '2015-01-01'][['date', 'daily_mean_load']].head(10)
    actuals = actuals.rename(columns={'date': 'ds', 'daily_mean_load': 'y'})
    forecast_df = model.predict(actuals[['ds']])
    merged = forecast_df[['ds', 'yhat']].merge(actuals, on='ds')
    merged['error_mw'] = merged['yhat'] - merged['y']
    merged['error_pct'] = ((merged['yhat'] - merged['y']) / merged['y'] * 100).round(1)
    print(merged[['ds', 'y', 'yhat', 'error_mw', 'error_pct']].to_string(index=False))

    # Also print the mean load by year to see the trend
    df['year'] = pd.to_datetime(df['date']).dt.year
    print("\n--- Mean daily load by year ---")
    print(df.groupby('year')['daily_mean_load'].mean().round(0).to_string())


if __name__ == "__main__":
    train_model()
    mape = evaluate_model()

    print(f"\n--- 7-day forecast ---")
    results = forecast(7)
    print(pd.DataFrame(results['forecast']).to_string(index=False))
    print(f"\nSummary: {results['summary']}")

    diagnose_level()