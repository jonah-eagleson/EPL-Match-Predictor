import pandas as pd
from sklearn.ensemble import RandomForestClassifier

matches = pd.read_csv("matches.csv", index_col=0)

## predictors
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

## ML model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)