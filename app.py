

# IMPORTS
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


# LOAD MODEL

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="vaijayanthimala07/tourism_package_prediction_model",
        filename="models/model_20260502_012720.joblib"
    )
    return joblib.load(model_path)

model = load_model()


# UI

st.title(" Tourism Package Prediction App")

st.write("""
Predict whether a customer is likely to purchase the ** Tourism Package**.
""")


# INPUTS

Age = st.number_input("Age", 1, 100, 40)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.number_input("City Tier", 1, 3, 3)
DurationOfPitch = st.number_input("Duration Of Pitch", 1, 200, 6)
Occupation = st.selectbox("Occupation", ["Salaried", "Large Business", "Small Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", 1, 10, 3)
NumberOfFollowups = st.number_input("Number Of Followups", 1, 10, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "King", "Deluxe", "Super Deluxe"])
PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number Of Trips", 0, 30, 2)
Passport = st.number_input("Passport (0=No, 1=Yes)", 0, 1, 1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.number_input("Own Car (0=No, 1=Yes)", 0, 1, 1)
NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", 1000, 200000, 30000)


# CREATE INPUT DATAFRAME

input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


# COLUMN ORDER

expected_cols = [
    'Age','CityTier','DurationOfPitch','NumberOfPersonVisiting',
    'NumberOfFollowups','PreferredPropertyStar','NumberOfTrips',
    'Passport','PitchSatisfactionScore','OwnCar',
    'NumberOfChildrenVisiting','MonthlyIncome',
    'TypeofContact','Occupation','Gender',
    'ProductPitched','MaritalStatus','Designation'
]

input_data = input_data[expected_cols]


# PREDICTION

if st.button(" Predict Package Booking"):
    try:
        probability = model.predict_proba(input_data)[0][1]

        # Adjustable threshold
        threshold = 0.45
        prediction = 1 if probability >= threshold else 0

        result = " Likely to Purchase" if prediction == 1 else " Not Likely"

        st.subheader(" Prediction Result")
        st.success(result)

        st.info(f"Probability of purchase: {round(probability, 3)}")
        st.caption(f"Threshold used: {threshold}")

    except Exception as e:
        st.error(" Error during prediction")
        st.exception(e)
