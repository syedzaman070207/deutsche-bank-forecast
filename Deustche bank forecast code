import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from prophet import Prophet
from sklearn.metrics import mean_squared_error
import yfinance as yf  # type: ignore

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: MAPE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# Fetch daily stock data for Deutsche Bank
db_data = yf.Ticker("DBK.DE")
history = db_data.history(period="1y")

# Preview the data
print(history.head())

# Prepare data for Prophet
prophet_df = history.reset_index()[['Date', 'Close']].rename(
    columns={'Date': 'ds', 'Close': 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

# Train/test split: hold out the last 30 trading days
split_idx = len(prophet_df) - 30
train = prophet_df.iloc[:split_idx].copy()
test = prophet_df.iloc[split_idx:].copy()

# Fit the model on the training data only
model = Prophet()
model.fit(train)

# Forecast on the exact dates we have (train + test)
future = pd.concat([train[['ds']], test[['ds']]])
forecast = model.predict(future)

# Evaluate on the test set
merged = test.merge(
    forecast[['ds', 'yhat']].rename(columns={'yhat': 'predicted'}),
    on='ds',
    how='inner'
)

mape = mean_absolute_percentage_error(merged['y'], merged['predicted'])
rmse = np.sqrt(mean_squared_error(merged['y'], merged['predicted']))
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.2f}")
print(len(merged), "test days scored")
print(merged.tail())

# Plot the forecast with the test data as red dots
fig = model.plot(forecast)
ax = fig.gca()
ax.scatter(test['ds'], test['y'], color='r',
           label='Actual (held-out test data)')
ax.legend()
ax.set_title(f"Deutsche Bank (DBK.DE) Forecast, MAPE: {mape:.1f}%")
fig.savefig("forecast.png", dpi=200, bbox_inches="tight")
plt.show()
