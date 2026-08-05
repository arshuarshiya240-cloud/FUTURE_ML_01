import os
import matplotlib.pyplot as plt


def create_output_folder():
    """Create output folder if it doesn't exist."""
    os.makedirs("outputs/plots", exist_ok=True)


# ---------------------------------------------------
# 1. Daily Sales Overview
# ---------------------------------------------------

def plot_sales_overview(daily_df):

    create_output_folder()

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily_df["date"],
        daily_df["total_sales"],
        linewidth=1.5
    )

    plt.title("Daily Sales Overview")
    plt.xlabel("Date")
    plt.ylabel("Sales")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/01_sales_overview.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------
# 2. Forecast Plot
# ---------------------------------------------------

def plot_forecast(history_df, forecast_df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        history_df["date"],
        history_df["total_sales"],
        label="Historical Sales"
    )

    plt.plot(
        forecast_df["date"],
        forecast_df["forecasted_sales"],
        label="Forecast",
        linewidth=2
    )

    plt.legend()

    plt.title("90-Day Sales Forecast")

    plt.xlabel("Date")

    plt.ylabel("Sales")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/02_forecast.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------
# 3. Category Analysis
# ---------------------------------------------------

def plot_category_sales(category_df):

    plt.figure(figsize=(10, 6))

    plt.bar(
        category_df["Category"],
        category_df["Sales"]
    )

    plt.title("Sales by Category")

    plt.xlabel("Category")

    plt.ylabel("Sales")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/03_category_analysis.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------
# 4. Model Comparison
# ---------------------------------------------------

def plot_model_comparison(results):

    names = list(results.keys())

    scores = [
        results[name]["RMSE"]
        for name in names
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(names, scores)

    plt.title("Model Comparison (RMSE)")

    plt.ylabel("RMSE")

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/04_model_comparison.png",
        dpi=300
    )

    plt.close()