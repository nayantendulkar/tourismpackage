import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of customer purchases of Tourism Package.
Enter the customer details and customer interaction details below to get a prediction.
""")

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
Occupation = st.selectbox("Occupation",["Free Lancer","Large Business","Salaried","Small Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.selectbox("Number of Persons Visiting", ["1","2","3","4","5"])
NumberOfFollowups = st.selectbox("Number of Followups", ["1","2","3","4","5","6"])
ProductPitched = st.selectbox("Product Pitched", ["Basic","Deluxe","King","Standard","Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["3","4","5"])
MaritalStatus = st.selectbox("Marital Status", ["Married","Single","Divorced","Unmarried"])
Passport = st.selectbox("Passport", ["0","1"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", ["1","2","3","4","5"])
OwnCar = st.selectbox("Own Car", ["0","1"])
NumberOfChildrenVisiting = st.selectbox("Number of Children Visiting", ["0","1","2","3"])
Designation = st.selectbox("Designation", ["Executive","Manager","AVP","Senior Manager","VP"])

Age  = st.number_input("Age", min_value=18,max_value=100,value=30)
DurationOfPitch = st.number_input("Duration of Pitch", min_value=5,max_value=150,value=30)
NumberOfTrips = st.number_input("Number of Trips", min_value=1,max_value=100,value=15)
MonthlyIncome = st.number_input("Monthly Income", min_value=100,max_value=100000,value=8000)

input_data = pd.DataFrame([{
    "Type of Contact": TypeofContact,
    "City Tier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "Number of Persons Visiting": NumberOfPersonVisiting,
    "Number of Followups": NumberOfFollowups,
    "Product Pitched": ProductPitched,
    "Preferred Property Star": PreferredPropertyStar,
    "Marital Status": MaritalStatus,
    "Passport": Passport,
    "Pitch Satisfaction Score": PitchSatisfactionScore,
    "Own Car": OwnCar,
    "Number of Children Visiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "Age": Age,
    "Duration of Pitch": DurationOfPitch,
    "Number of Trips": NumberOfTrips,
    "Monthly Income": MonthlyIncome
}])

if st.button("Tourism Package Failure"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Purchased" if prediction == 1 else "Tourism Package NOT Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
