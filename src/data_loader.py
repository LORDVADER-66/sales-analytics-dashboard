import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the Superstore CSV, parses dates, renames columns to snake_case,
    and drops duplicate rows.

    Args:
        filepath: path to the superstore CSV file

    Returns:
        Raw cleaned DataFrame
    """
    df = pd.read_csv(
        filepath,
        parse_dates=["Order Date", "Ship Date"],
        encoding="latin-1"
    )

    # Rename columns to snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    if before != after:
        print(f"Dropped {before - after} duplicate rows")

    # Summary
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isnull().sum()}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies transformations to the raw DataFrame:
    - Extracts time features from order_date
    - Creates profit_margin column
    - Validates region values

    Args:
        df: raw DataFrame from load_data()

    Returns:
        Cleaned and enriched DataFrame
    """
    df = df.copy()

    # Time features
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    df["day_of_week"] = df["order_date"].dt.dayofweek  # 0=Monday, 6=Sunday

    # Profit margin
    df["profit_margin"] = (df["profit"] / df["sales"] * 100).round(2)

    # Region validation
    expected_regions = {"East", "West", "Central", "South"}
    actual_regions = set(df["region"].unique())

    if actual_regions != expected_regions:
        print(f"WARNING: Unexpected regions found: {actual_regions}")
    else:
        print("Region validation passed:", actual_regions)

    return df