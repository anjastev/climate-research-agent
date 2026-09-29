import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.climate_agent.data_loader import load_nasa_temperature_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "nasa_global_temperature.csv"

rows = load_nasa_temperature_data(str(DATA_FILE))

# Make a DataFrame: a table-like structure that pandas can analyze.
data = pd.DataFrame(rows)

# Group the monthly records by year and calculate each year's mean.
data["year"] = pd.to_datetime(data["date"]).dt.year
annual_data = data.groupby("year", as_index=False)["temperature_anomaly"].mean()
# Keep only years with all 12 monthly records as complete years.
monthly_counts = data.groupby("year")["temperature_anomaly"].count()
annual_data["month_count"] = annual_data["year"].map(monthly_counts)

complete_years = annual_data[annual_data["month_count"] == 12]
partial_years = annual_data[annual_data["month_count"] < 12]

print("Latest annual values:")
print(annual_data.tail())
plt.figure(figsize=(12, 6))

# Draw complete years with a solid line.
plt.plot(
    complete_years["year"],
    complete_years["temperature_anomaly"],
    color="firebrick",
    label="Complete years",
)

# Draw incomplete years separately with a dashed line.
if not partial_years.empty:
    plt.plot(
        partial_years["year"],
        partial_years["temperature_anomaly"],
        color="darkorange",
        marker="o",
        linestyle="--",
        label="Partial year",
    )

plt.axhline(0, color="black", linewidth=0.8)
plt.title("Annual Global Temperature Anomaly")
plt.xlabel("Year")
plt.ylabel("Temperature anomaly (°C)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "annual_temperature_anomaly.png", dpi=150)

plt.show()