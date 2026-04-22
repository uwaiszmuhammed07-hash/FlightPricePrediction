import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("📦 Loading dataset...")
df = pd.read_excel("data/Flight_Fare.xlsx")
df.dropna(inplace=True)

print("⚙️  Preprocessing...")
df["Journey_Day"]   = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
df["Journey_Month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
df["Dep_Hour"]      = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_Min"]       = pd.to_datetime(df["Dep_Time"]).dt.minute
df["Arrival_Hour"]  = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arrival_Min"]   = pd.to_datetime(df["Arrival_Time"]).dt.minute

def convert_duration(duration):
    hours, minutes = 0, 0
    duration = str(duration).strip()
    if "h" in duration:
        hours = int(duration.split("h")[0].strip())
    if "m" in duration:
        minutes = int(duration.split("m")[0].split()[-1].strip())
    return hours * 60 + minutes

df["Duration_minutes"] = df["Duration"].apply(convert_duration)
stops_map = {"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4}
df["Total_Stops"] = df["Total_Stops"].map(stops_map)
df.drop(columns=["Date_of_Journey","Dep_Time","Arrival_Time","Duration","Route","Additional_Info"], inplace=True)
df = pd.get_dummies(df, columns=["Airline","Source","Destination"], drop_first=True)
df["Price"] = df["Price"].clip(upper=df["Price"].quantile(0.995))
df["is_short_flight"]   = (df["Duration_minutes"] < 120).astype(int)
df["is_long_flight"]    = (df["Duration_minutes"] > 360).astype(int)
df["is_peak_departure"] = (((df["Dep_Hour"]>=6)&(df["Dep_Hour"]<=9))|((df["Dep_Hour"]>=17)&(df["Dep_Hour"]<=20))).astype(int)
df["is_late_night"]     = ((df["Dep_Hour"]>=23)|(df["Dep_Hour"]<=5)).astype(int)
df["is_weekend"]        = (df["Journey_Day"]%7).isin([4,5,6]).astype(int)

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 Training Random Forest...")
model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"✅ R²  : {r2_score(y_test, preds):.4f}")
print(f"✅ MAE : ₹{mean_absolute_error(y_test, preds):.2f}")

joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("✅ model.pkl saved!")
print("✅ model_columns.pkl saved!")
print("🚀 Now run: streamlit run app.py")
