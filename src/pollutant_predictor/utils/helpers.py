import pandas as pd

def load_data(path):
    data_types = {
        "rideable_type": "category", 
        "start_station_name": "category", 
        "end_station_name": "category", 
        "member_casual": "category",
        "time_of_day": "category",
        "trip_type": "category"
    }
    return pd.read_csv(path, dtype=data_types, parse_dates=["started_at", "ended_at"], low_memory=False)
