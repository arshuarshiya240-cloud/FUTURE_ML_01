import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing import (
    load_and_clean,
    aggregate_daily,
    get_category_sales,
)

from src.feature_engineering import build_feature_matrix

from src.models import (
    train_linear,
    train_random_forest,
    train_gradient_boosting,
    get_metrics,
    forecast_future,
)

from src.visualization import (
    plot_sales_overview,
    plot_forecast,
    plot_category_sales,
    plot_model_comparison,
)


def main():

    print("=" * 60)
    print(" SALES & DEMAND FORECASTING PROJECT")
    print("=" * 60)

    # -----------------------------
    # Load Dataset
    # -----------------------------

    print("\nLoading dataset...")

    df = load_and_clean()

    daily = aggregate_daily(df)

    feature_df, FEATURE_COLS = build_feature_matrix(daily)

    # -----------------------------
    # Train Test Split
    # -----------------------------

    train_size = int(len(feature_df) * 0.8)

    train = feature_df.iloc[:train_size]
    test = feature_df.iloc[train_size:]

    X_train = train[FEATURE_COLS]
    y_train = train["total_sales"]

    X_test = test[FEATURE_COLS]
    y_test = test["total_sales"]

    # -----------------------------
    # Train Models
    # -----------------------------

    print("Training Linear Regression...")

    linear = train_linear(X_train, y_train)

    print("Training Random Forest...")

    rf = train_random_forest(X_train, y_train)

    print("Training Gradient Boosting...")

    gb = train_gradient_boosting(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------

    pred_linear = linear.predict(X_test)
    pred_rf = rf.predict(X_test)
    pred_gb = gb.predict(X_test)

    results = {

        "Linear Regression":
            get_metrics(y_test, pred_linear),

        "Random Forest":
            get_metrics(y_test, pred_rf),

        "Gradient Boosting":
            get_metrics(y_test, pred_gb),
    }

    print("\nModel Performance\n")

    for model, metrics in results.items():

        print(model)

        for k, v in metrics.items():
            print(f"   {k}: {v:.2f}")

        print()

    # -----------------------------
    # Select Best Model
    # -----------------------------

    best_model_name = min(
        results,
        key=lambda x: results[x]["RMSE"]
    )

    print(f"Best Model: {best_model_name}")

    if best_model_name == "Linear Regression":
        best_model = linear

    elif best_model_name == "Random Forest":
        best_model = rf

    else:
        best_model = gb

    # -----------------------------
    # Forecast
    # -----------------------------

    forecast = forecast_future(
        best_model,
        feature_df["date"].iloc[-1],
        feature_df["total_sales"],
        FEATURE_COLS,
        n_days=90,
    )

    os.makedirs(
        "outputs/forecasts",
        exist_ok=True,
    )

    forecast.to_csv(
        "outputs/forecasts/90_day_forecast.csv",
        index=False,
    )

    print("Forecast saved.")

    # -----------------------------
    # Charts
    # -----------------------------

    plot_sales_overview(daily)

    plot_forecast(
        feature_df[["date", "total_sales"]],
        forecast,
    )

    category = get_category_sales(df)

    plot_category_sales(category)

    plot_model_comparison(results)

    print("Charts saved.")

    # -----------------------------
    # Business Report
    # -----------------------------

    os.makedirs("reports", exist_ok=True)

    with open(
        "reports/business_insights.md",
        "w",
        encoding="utf-8",
    ) as f:

        f.write("# Business Insights\n\n")

        f.write(
            f"Best Performing Model: **{best_model_name}**\n\n"
        )

        f.write(
            f"Total Revenue: ${df['Sales'].sum():,.2f}\n\n"
        )

        f.write(
            f"Average Daily Sales: ${daily['total_sales'].mean():,.2f}\n\n"
        )

        top_category = (
            category.iloc[0]["Category"]
        )

        f.write(
            f"Top Selling Category: **{top_category}**\n\n"
        )

        f.write(
            "Recommendation:\n"
        )

        f.write(
            "- Increase inventory for high-performing categories.\n"
        )

        f.write(
            "- Monitor seasonal demand.\n"
        )

        f.write(
            "- Use forecast for procurement planning.\n"
        )

    print("Business report saved.")

    print("\nProject Completed Successfully!")


if __name__ == "__main__":
    main()