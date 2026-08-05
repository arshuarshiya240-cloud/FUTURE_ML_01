import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# Model Training
# --------------------------------------------------

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=12
    )

    model.fit(X_train, y_train)

    return model


def train_gradient_boosting(X_train, y_train):

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

def get_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# --------------------------------------------------
# Future Forecast
# --------------------------------------------------

def forecast_future(
        model,
        last_date,
        history_sales,
        feature_cols,
        n_days=90
):

    history = list(history_sales)

    forecasts = []

    current_date = last_date

    for i in range(n_days):

        current_date = current_date + pd.Timedelta(days=1)

        lag1 = history[-1]
        lag7 = history[-7]
        lag30 = history[-30]

        rolling7 = np.mean(history[-7:])
        rolling30 = np.mean(history[-30:])
        rolling_std7 = np.std(history[-7:])

        row = {
            "day": current_date.day,
            "month": current_date.month,
            "year": current_date.year,
            "dayofweek": current_date.dayofweek,
            "quarter": current_date.quarter,
            "dayofyear": current_date.dayofyear,
            "weekofyear": int(current_date.isocalendar().week),
            "is_weekend": int(current_date.dayofweek >= 5),

            "month_sin":
                np.sin(
                    2 * np.pi * current_date.month / 12
                ),

            "month_cos":
                np.cos(
                    2 * np.pi * current_date.month / 12
                ),

            "lag_1": lag1,
            "lag_7": lag7,
            "lag_30": lag30,

            "rolling_mean_7": rolling7,
            "rolling_mean_30": rolling30,
            "rolling_std_7": rolling_std7,

            "trend": len(history)
        }

        X = pd.DataFrame([row])

        X = X[feature_cols]

        prediction = model.predict(X)[0]

        prediction = max(0, prediction)

        forecasts.append(prediction)

        history.append(prediction)

    return pd.DataFrame({

        "date":
            pd.date_range(
                last_date + pd.Timedelta(days=1),
                periods=n_days
            ),

        "forecasted_sales":
            forecasts
    })