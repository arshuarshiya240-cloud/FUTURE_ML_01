import pandas as pd


def load_and_clean(file_path="data/superstore.csv"):
    """
    Load the Superstore dataset and perform basic cleaning.
    """

    df = pd.read_csv(file_path, encoding="latin1")

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Convert date columns
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing sales or dates
    df = df.dropna(subset=["Order Date", "Sales"])

    # Ensure numeric columns
    numeric_cols = ["Sales", "Profit", "Quantity", "Discount"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    return df


def aggregate_daily(df):
    """
    Aggregate total sales per day.
    """

    daily_df = (
        df.groupby("Order Date")["Sales"]
        .sum()
        .reset_index()
    )

    daily_df.columns = ["date", "total_sales"]

    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    return daily_df


def get_category_sales(df):
    """
    Total sales by category.
    """

    return (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def get_region_sales(df):
    """
    Total sales by region.
    """

    return (
        df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


if __name__ == "__main__":

    df = load_and_clean()

    print(df.head())

    print("\nDataset Shape:", df.shape)

    daily = aggregate_daily(df)

    print("\nDaily Sales")

    print(daily.head())