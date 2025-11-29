
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


# -----------------------------
# Load model and columns
# -----------------------------
model_path = "model.pkl"  # adjust path if needed
columns_path = "columns.json" # adjust path if needed

loaded_model = joblib.load(model_path)

with open(columns_path, "r") as f:
    model_columns = json.load(f)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("House Price Prediction")

st.write("Enter the property details below:")

# --- Inputs ---
# Numeric Inputs
LivingArea = st.number_input("Living Area (sq ft)", min_value=0.0, step=50.0)
BuildingAreaTotal = st.number_input("Building Area Total (sq ft)", min_value=0.0, step=50.0)
BedroomsTotal = st.number_input("Bedrooms Total", min_value=0, step=1)
BathroomsTotalInteger = st.number_input("Bathrooms Total", min_value=0, step=1)
GarageSpaces = st.number_input("Garage Spaces", min_value=0, step=1)
ParkingTotal = st.number_input("Parking Total", min_value=0, step=1)
LotSizeSquareFeet = st.number_input("Lot Size (sq ft)", min_value=0.0, step=50.0)
LotSizeAcres = st.number_input("Lot Size (acres)", min_value=0.0, step=0.01)
LotSizeArea = st.number_input("Lot Size Area", min_value=0.0, step=50.0)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2030, step=1)
Latitude = st.number_input("Latitude", value=0.0)
Longitude = st.number_input("Longitude", value=0.0)
StreetNumberNumeric = st.number_input("Street Number", min_value=0, step=1)
Levels = st.number_input("Levels", min_value=0, step=1)

# Binary / Yes-No Inputs
PoolPrivateYN = st.selectbox("Private Pool?", ["Yes", "No"])
NewConstructionYN = st.selectbox("New Construction?", ["Yes", "No"])
ViewYN = st.selectbox("View?", ["Yes", "No"])
BasementYN = st.selectbox("Basement?", ["Yes", "No"])
FireplaceYN = st.selectbox("Fireplace?", ["Yes", "No"])

# Categorical Inputs
City = st.text_input("City")
CountyOrParish = st.text_input("County/Parish")
StateOrProvince = st.text_input("State/Province")
PostalCode = st.text_input("Postal Code")

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Price"):
    # Build a single-row DataFrame
    input_dict = {
        "LivingArea": LivingArea,
        "BuildingAreaTotal": BuildingAreaTotal,
        "BedroomsTotal": BedroomsTotal,
        "BathroomsTotalInteger": BathroomsTotalInteger,
        "GarageSpaces": GarageSpaces,
        "ParkingTotal": ParkingTotal,
        "LotSizeSquareFeet": LotSizeSquareFeet,
        "LotSizeAcres": LotSizeAcres,
        "LotSizeArea": LotSizeArea,
        "YearBuilt": YearBuilt,
        "Latitude": Latitude,
        "Longitude": Longitude,
        "StreetNumberNumeric": StreetNumberNumeric,
        "Levels": Levels,
        "PoolPrivateYN": PoolPrivateYN,
        "NewConstructionYN": NewConstructionYN,
        "ViewYN": ViewYN,
        "BasementYN": BasementYN,
        "FireplaceYN": FireplaceYN,
        "City": City,
        "CountyOrParish": CountyOrParish,
        "StateOrProvince": StateOrProvince,
        "PostalCode": PostalCode
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure columns match the model's training columns
    input_encoded = pd.get_dummies(input_df)
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]  # reorder columns

    # Prediction
    prediction = loaded_model.predict(input_encoded)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
