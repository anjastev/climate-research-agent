from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.climate_agent.data_loader import load_nasa_temperature_data


# Find the project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output files.
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "nasa_global_temperature.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    """Load, validate, summarize, and plot global temperature anomalies."""
    # Make sure the output folder exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the cleaned monthly records from the NOAA PSL CSV.
    rows = load_nasa_temperature_data(str(DATA_FILE))

    if not rows:
        raise ValueError(f"No valid temperature records found in {DATA_FILE}")

    # Convert the list of records into a pandas DataFrame.
    data = pd.DataFrame(rows)

    # Ensure dates are represented as pandas datetime values.
    data["date"] = pd.to_datetime(data["date"])

    # Add a year column so monthly values can be grouped by year.
    data["year"] = data["date"].dt.year

    # Count valid monthly readings for each year.
    monthly_counts = data.groupby("year")["temperature_anomaly"].count()

    # Calculate the mean anomaly for each year.
    annual_data = (
        data.groupby("year", as_index=False)["temperature_anomaly"]
        .mean()
    )

    # Add the number of available months and mark full years.
    annual_data["month_count"] = annual_data["year"].map(monthly_counts)
    annual_data["is_complete_year"] = annual_data["month_count"] == 12

    # Separate complete and partial years.
    complete_years = annual_data[annual_data["is_complete_year"]].copy()
    partial_years = annual_data[~annual_data["is_complete_year"]].copy()

    # Calculate a 10-year rolling average using complete years only.
    complete_years["rolling_avg_10y"] = (
        complete_years["temperature_anomaly"]
        .rolling(window=10, min_periods=10)
        .mean()
    )

    # Add the rolling average to the full annual table.
    rolling_averages = complete_years.set_index("year")["rolling_avg_10y"]
    annual_data["rolling_avg_10y"] = annual_data["year"].map(rolling_averages)

    # Show the data quality check and the most recent values.
    print("Incomplete years:")
    if partial_years.empty:
        print("None")
    else:
        print(partial_years[["year", "month_count"]].to_string(index=False))

    print("\nLatest annual values:")
    print(annual_data.tail().to_string(index=False))

    # Save the processed annual data as a CSV file.
    annual_csv_file = OUTPUT_DIR / "annual_temperature_summary.csv"
    annual_data.to_csv(annual_csv_file, index=False)

    # Create the chart.
    plt.figure(figsize=(12, 6))

    # Annual values for years with all 12 months.
    plt.plot(
        complete_years["year"],
        complete_years["temperature_anomaly"],
        color="firebrick",
        linewidth=1,
        label="Complete years",
    )

    # 10-year moving average, which emphasizes the long-term pattern.
    plt.plot(
        complete_years["year"],
        complete_years["rolling_avg_10y"],
        color="royalblue",
        linewidth=2,
        label="10-year rolling average",
    )

    # Show partial years separately.
    if not partial_years.empty:
        plt.scatter(
            partial_years["year"],
            partial_years["temperature_anomaly"],
            color="darkorange",
            label="Partial year",
            zorder=3,
        )

    # Add chart labels and styling.
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Annual Global Temperature Anomaly")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the chart into the outputs folder.
    chart_file = OUTPUT_DIR / "annual_temperature_anomaly.png"
    plt.savefig(chart_file, dpi=150)

    print(f"\nSaved annual data to: {annual_csv_file}")
    print(f"Saved chart to: {chart_file}")

    plt.show()


if __name__ == "__main__":
    main()