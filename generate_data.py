"""Generate a realistic Food_Delivery_Times.csv dataset for the DeliveryIQ app."""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

weather_options = ["Clear", "Rainy", "Snowy", "Foggy"]
traffic_options = ["Low", "Medium", "High"]
time_options = ["Morning", "Afternoon", "Evening", "Night"]
vehicle_options = ["Scooter", "Bike", "Car"]

data = {
    "Order_ID": range(1, n + 1),
    "Distance_km": np.round(np.random.uniform(1.0, 20.0, n), 1),
    "Weather": np.random.choice(weather_options, n),
    "Traffic_Level": np.random.choice(traffic_options, n),
    "Time_of_Day": np.random.choice(time_options, n),
    "Vehicle_Type": np.random.choice(vehicle_options, n),
    "Preparation_Time_min": np.random.randint(5, 31, n),
    "Courier_Experience_yrs": np.round(np.random.uniform(0.0, 9.0, n), 1),
}

df = pd.DataFrame(data)

# Realistic delivery time based on features
base_time = 10
distance_factor = df["Distance_km"] * 2.1
prep_factor = df["Preparation_Time_min"] * 0.45

weather_map = {"Clear": 0, "Rainy": 4, "Snowy": 6, "Foggy": 3}
weather_factor = df["Weather"].map(weather_map)

traffic_map = {"Low": 0, "Medium": 5, "High": 10}
traffic_factor = df["Traffic_Level"].map(traffic_map)

time_map = {"Morning": 2, "Afternoon": 3, "Evening": 5, "Night": 1}
time_factor = df["Time_of_Day"].map(time_map)

vehicle_map = {"Car": -3, "Scooter": 0, "Bike": 2}
vehicle_factor = df["Vehicle_Type"].map(vehicle_map)

experience_factor = -df["Courier_Experience_yrs"] * 0.8

noise = np.random.normal(0, 3, n)

df["Delivery_Time_min"] = np.round(
    base_time + distance_factor + prep_factor + weather_factor +
    traffic_factor + time_factor + vehicle_factor + experience_factor + noise,
    1
)
df["Delivery_Time_min"] = df["Delivery_Time_min"].clip(lower=5)

# Add ~3% nulls for realism
null_indices = np.random.choice(n, size=int(n * 0.03), replace=False)
for idx in null_indices:
    col = np.random.choice(["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"])
    df.loc[idx, col] = np.nan

df.to_csv("Food_Delivery_Times.csv", index=False)
print(f"Generated Food_Delivery_Times.csv with {len(df)} rows")
print(df.head())
