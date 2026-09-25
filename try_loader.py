from src.climate_agent.data_loader import load_nasa_temperature_data


rows = load_nasa_temperature_data("data/raw/nasa_global_temperature.csv")

print(f"Number of valid rows: {len(rows)}")
print("First 3 rows:")
print(rows[:3])
print("Last row:")
print(rows[-1])