import csv
from pathlib import Path
from datetime import date

def load_csv(file_path: str) -> list[dict[str, str]]:
    """Read a CSV file and return its rows as dictionaries."""
    path = Path(file_path)

    with path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def convert_climate_rows(rows: list[dict[str, str]]) -> list[dict]:
    """Convert climate CSV values into appropriate Python types."""
    converted_rows = []

    for row in rows:
        converted_rows.append({
            "year": int(row["year"]),
            "region": row["region"],
            "temperature_anomaly": float(row["temperature_anomaly"]),
        })

    return converted_rows

def load_nasa_temperature_data(file_path: str) -> list[dict]:
    """Load monthly NASA temperature anomalies from the NOAA PSL CSV file."""
    rows = []

    with open(file_path, mode="r", encoding="utf-8", newline="") as csv_file:
        # Skip the descriptive first line; data starts on the next line.
        next(csv_file)

        for line in csv_file:
            date_text, value_text = line.strip().split(",")

            value = float(value_text)

            # NOAA marks missing values with 9999.
            if value == -9999:
                continue

            rows.append({
                "date": date.fromisoformat(date_text),
                "temperature_anomaly": value,
            })

    return rows