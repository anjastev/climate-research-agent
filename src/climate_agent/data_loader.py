import csv
from datetime import date


def load_csv(file_path: str) -> list[dict[str, str]]:
    """Read a standard CSV file and return its rows as dictionaries."""
    with open(file_path, mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def convert_climate_rows(rows: list[dict[str, str]]) -> list[dict]:
    """Convert the sample climate CSV values into Python types."""
    converted_rows = []

    for row in rows:
        converted_rows.append(
            {
                "year": int(row["year"]),
                "region": row["region"],
                "temperature_anomaly": float(row["temperature_anomaly"]),
            }
        )

    return converted_rows


def load_nasa_temperature_data(file_path: str) -> list[dict]:
    """Load monthly temperature anomalies from the NOAA PSL CSV file."""
    records = []

    with open(file_path, mode="r", encoding="utf-8", newline="") as csv_file:
        # The first line contains source information rather than column names.
        next(csv_file)

        for line_number, line in enumerate(csv_file, start=2):
            line = line.strip()

            if not line:
                continue

            columns = line.split(",")

            if len(columns) != 2:
                raise ValueError(
                    f"Unexpected format on line {line_number}: {line}"
                )

            date_text = columns[0].strip()
            value_text = columns[1].strip()

            try:
                anomaly = float(value_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid temperature value on line {line_number}: "
                    f"{value_text}"
                ) from error

            # NOAA PSL marks missing measurements with -9999.
            if anomaly == -9999:
                continue

            records.append(
                {
                    "date": date.fromisoformat(date_text),
                    "temperature_anomaly": anomaly,
                }
            )

    return records