# Streamlit turns a plain Python script into an interactive web app: every
# st.something() call below renders an actual widget or element in the
# browser. There is no HTML/JavaScript to write.
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------------------------------
# IMPORTANT: how Streamlit actually runs this file
# Streamlit re-runs this ENTIRE script, top to bottom, every single time the
# user interacts with anything, types a number, picks a dropdown option, or
# clicks a button. There's no separate "event handler" like in a typical
# web framework; the whole file just runs again, and Streamlit is smart
# enough to only redraw what changed. This is why the model is loaded inside
# a cached function below, rather than as a plain top-level line: without
# caching, it would be re-loaded from disk on every keystroke.
# --------------------------------------------------------------------------

# @st.cache_resource tells Streamlit: run this function once, keep its
# result in memory, and reuse that same result on every future re-run
# instead of calling the function again. Perfect for a model that doesn't
# change, and loading it from disk repeatedly would be slow and pointless.
@st.cache_resource
def load_bundle():
    return joblib.load("model/car_price_model_bundle.pkl")

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
selected_features = bundle["selected_features"]
continuous_cols = bundle["continuous_cols"]
log_transform_cols = bundle["log_transform_cols"]

# st.title() renders a large page heading. st.write() renders plain text
# (or, more generally, almost anything you pass it: text, numbers,
# dataframes, charts).
st.title("Used Car Price Predictor")
st.write(
    "Enter a vehicle's details below to get an estimated selling price, "
    "based on a Random Forest model trained on 15,240 used-car listings."
)

# --------------------------------------------------------------------------
# Input widgets. Each one renders on the page immediately when this line
# runs, and each returns whatever value the user currently has it set to.
# --------------------------------------------------------------------------

st.subheader("Vehicle details")

# st.number_input renders a numeric field with up/down arrows.
#   label       -> text shown above the field
#   min_value/max_value -> hard limits on what can be entered
#   value       -> the number shown when the app first loads
#   step        -> how much the up/down arrows change the value by
vehicle_age = st.number_input("Vehicle age (years)", min_value=0, max_value=30, value=5, step=1)
mileage = st.number_input("Mileage (km/l)", min_value=4.0, max_value=34.0, value=19.0, step=0.1)
engine = st.number_input("Engine size (cc)", min_value=600, max_value=6600, value=1200, step=10)
max_power = st.number_input("Max power (bhp)", min_value=30.0, max_value=630.0, value=90.0, step=0.1)

# st.selectbox renders a dropdown. The first argument is the label shown
# above it; the second is the list of options shown inside it.
#
# NOTE ON THESE THREE OPTION LISTS: the model was trained on only 10
# selected features (Section 9 of the notebook), which include dummy
# columns for just 3 brands (BMW, Maruti, Mercedes-Benz) and 2 fuel types
# (Diesel, Petrol). Every other brand or fuel type was dropped during
# feature selection, so the model structurally cannot tell them apart from
# the reference category it was trained against. Offering a full list of 32
# brands here would silently give identical, misleading predictions for
# most of them, so the dropdown only offers the categories the model can
# actually use, plus an honest "Other" option for everything else.
brand_choice = st.selectbox("Brand", ["BMW", "Maruti", "Mercedes-Benz", "Other"])
fuel_choice = st.selectbox("Fuel type", ["Diesel", "Petrol", "Other (CNG / LPG / Electric)"])
transmission_choice = st.selectbox("Transmission", ["Manual", "Automatic"])

if brand_choice == "Other" or fuel_choice.startswith("Other"):
    st.caption(
        "Note: the model was trained on a limited set of brand/fuel indicators. "
        "'Other' selections are predicted as the baseline category and will be "
        "less precise than the specific options listed."
    )

# st.button renders a clickable button and returns True only on the specific
# re-run where it was just clicked, and False on every other re-run, including
# the very first load and any later re-run triggered by changing a widget
# above. That's exactly the behaviour we want: don't predict until asked to.
if st.button("Predict price"):

    # Build a single-row DataFrame with every column the model expects,
    # defaulting the categorical dummy columns to 0.
    row = {col: 0 for col in selected_features}
    row["vehicle_age"] = vehicle_age
    row["mileage"] = mileage
    row["engine"] = engine
    row["max_power"] = max_power

    # Only set the ONE dummy column that matches the user's actual choice --
    # this is exactly how pd.get_dummies(..., drop_first=True) represented
    # these categories during training in the notebook's Section 8.
    if brand_choice == "BMW":
        row["brand_BMW"] = 1
    elif brand_choice == "Maruti":
        row["brand_Maruti"] = 1
    elif brand_choice == "Mercedes-Benz":
        row["brand_Mercedes-Benz"] = 1
    # "Other" -> leave all three brand columns at 0 (same as the dropped
    # reference category, Audi, from the notebook's Section 11)

    if fuel_choice == "Diesel":
        row["fuel_type_Diesel"] = 1
    elif fuel_choice == "Petrol":
        row["fuel_type_Petrol"] = 1
    # "Other" -> leave both fuel columns at 0

    if transmission_choice == "Manual":
        row["transmission_type_Manual"] = 1
    # "Automatic" -> leave at 0

    # pd.DataFrame([row]) builds a one-row table from the dictionary.
    # [selected_features] reorders its columns to EXACTLY match the order
    # the model was trained on. scikit-learn matches columns by position,
    # not by name, so getting this order right matters.
    X_new = pd.DataFrame([row])[selected_features]

    # Apply the SAME two preprocessing steps used in the notebook, in the
    # SAME order: log-transform first, then scale, using the scaler that
    # was FIT on training data, only ever TRANSFORMING here, never re-fit.
    for col in log_transform_cols:
        X_new[col] = np.log1p(X_new[col])
    X_new[continuous_cols] = scaler.transform(X_new[continuous_cols])

    # model.predict() returns the model's prediction in LOG space, because
    # that's what it was trained on (the notebook's log1p transform of
    # selling_price). np.expm1 is the exact inverse of that transform, so
    # this converts the prediction back into actual Rupees.
    pred_log = model.predict(X_new)[0]
    pred_price = np.expm1(pred_log)

    # st.success renders a green highlighted message, reserved for the
    # actual result, so it stands out from the rest of the page.
    st.success(f"Estimated selling price: ₹{pred_price:,.0f}")

    # The notebook's Section 12 found this model's typical test-set error
    # was about ₹113,500 (MAE), showing that alongside the prediction is
    # honest about the model's real-world precision, not just a bare number.
    st.caption("Typical error on unseen test data: about ₹113,500 (MAE, notebook Section 12).")
