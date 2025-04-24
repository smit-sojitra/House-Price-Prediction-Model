import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
# Title
st.title("Delhi House Price Prediction")

# Load the dataset
df = pd.read_csv("MagicBricks.csv")

# Fill missing Per_Sqft if needed
df["Per_Sqft"] = df["Per_Sqft"].fillna(df["Price"] / df["Area"])

# Drop unnecessary column
df.drop(["Per_Sqft"], axis=1, inplace=True)

# Encode categorical columns
cat_cols = ["Furnishing", "Locality", "Status", "Transaction", "Type"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Scale numerical features separately
area_scaler = MinMaxScaler()
df["Area"] = area_scaler.fit_transform(df[["Area"]])

price_scaler = MinMaxScaler()
df["Price"] = price_scaler.fit_transform(df[["Price"]])

# Split data
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = joblib.load("delhi_price_model.pkl")
# model = joblib.load("delhi_price_model2.pkl")

# Input widgets
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, step=50)
bhk = st.selectbox("BHK", sorted(df["BHK"].unique()))
bathroom = st.selectbox("Bathroom", sorted(df["Bathroom"].unique()))
parking = st.selectbox("Parking", sorted(df["Parking"].unique()))
furnishing = st.selectbox("Furnishing", encoders["Furnishing"].classes_)
locality = st.selectbox("Locality", encoders["Locality"].classes_)
status = st.selectbox("Status", encoders["Status"].classes_)
transaction = st.selectbox("Transaction", encoders["Transaction"].classes_)
type_ = st.selectbox("Type", encoders["Type"].classes_)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame(
        {
            "Area": [area],
            "BHK": [bhk],
            "Bathroom": [bathroom],
            "Furnishing": [encoders["Furnishing"].transform([furnishing])[0]],
            "Locality": [encoders["Locality"].transform([locality])[0]],
            "Parking": [parking],
            "Status": [encoders["Status"].transform([status])[0]],
            "Transaction": [encoders["Transaction"].transform([transaction])[0]],
            "Type": [encoders["Type"].transform([type_])[0]],
        }
    )
    input_data["Area_Yards"] = input_data["Area"] / 9

    input_data["Area"] = area_scaler.transform(input_data[["Area"]])
    prediction = model.predict(input_data)[0]
    st.success(
        f"Predicted Price: ₹ {price_scaler.inverse_transform([[prediction]])[0][0]:,.2f}"
    )
