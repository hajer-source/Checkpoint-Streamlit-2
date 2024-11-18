import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model (ensure you have saved your model using joblib or pickle)
model = joblib.load('random_forest_model.pkl')

# Streamlit UI
st.title('Financial Inclusion in Africa')

# Input fields for each feature (match these with your dataset columns)

household_size = st.number_input('Household Size', min_value=1, max_value=20)
age_of_respondent = st.number_input('Age of Respondent', min_value=18, max_value=100)

countries = ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']   
Country = st.selectbox('Country', countries)

b_account = ['Yes', 'No']
bank_account= st.selectbox('Tell us if you have a bank account, please :', b_account)

locations=['Urban', 'Rural']
location_type = st.selectbox('Location Type', locations)

cell_access= ['Yes', 'No']
cellphone_access = st.selectbox('Tell us if you have a cellphone access, please :', cell_access)

gender = ['Female', 'Male']
gender_of_respondent = st.selectbox('Your gender:', gender)

Relationship = ['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
 'Other non-relatives']
relationship_with_head = st.selectbox('What is your relationship to the head, please?', Relationship)

Marital_st = ['Married/Living together', 'Widowed', 'Single/Never Married',
 'Divorced/Seperated', 'Dont know']
marital_status = st.selectbox('What is your current marital status, please?', Marital_st)

education_l = ['Secondary education', 'No formal education',
 'Vocational/Specialised training', 'Primary education',
 'Tertiary education', 'Other/Dont know/RTA']
education_level = st.selectbox('What is your education_level?', education_l)

job_status = ['Self employed', 'Government Dependent', 'Formally employed Private',
 'Informally employed', 'Formally employed Government',
 'Farming and Fishing', 'Remittance Dependent', 'Other Income',
 'Dont Know/Refuse to answer', 'No Income']
job_type = st.selectbox('Please, select your job type:', job_status)


loaded_le = joblib.load('le.pkl')
countries = [['Kenya', 'Rwanda', 'Tanzania', 'Uganda']] 
countries = [item for sublist in countries for item in sublist]  
Country_encoded = loaded_le.transform(countries)

loaded_le1 = LabelEncoder()
loaded_le1.fit(['Yes', 'No'])
# Handle unseen labels
loaded_le1.handle_unknown = 'use_encoded_value'  
loaded_le1.unknown_value = -1  
# Transform input data, 'e' will be treated as an unknown label and encoded as -1
bank_account_encoded = loaded_le1.transform(b_account) 

loaded_le2 = joblib.load('le2.pkl')
location_type = loaded_le2.transform(locations)
loaded_le2.fit(['Urban', 'Rural'])


loaded_le3 = joblib.load('le3.pkl')
cellphone_access_encoded = loaded_le3.transform(cell_access)
loaded_le3.fit(['Yes', 'No'])


loaded_le4 = joblib.load('le4.pkl')
gender_of_respondent = loaded_le4.transform(gender)
loaded_le4.fit(['Female', 'Male'])


loaded_le5 = joblib.load('le5.pkl')
relationship_with_head = loaded_le5.transform(Relationship)
loaded_le5.fit(['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
 'Other non-relatives'])


loaded_le6 = joblib.load('le6.pkl')
marital_status = loaded_le6.transform(Marital_st)
loaded_le6.fit(['Married/Living together', 'Widowed', 'Single/Never Married',
 'Divorced/Seperated', 'Dont know'])


loaded_le7 = joblib.load('le7.pkl')
education_level = loaded_le7.transform(education_l)
loaded_le7.fit(['Secondary education', 'No formal education',
 'Vocational/Specialised training', 'Primary education',
 'Tertiary education', 'Other/Dont know/RTA'])


loaded_le8 = joblib.load('le8.pkl')
job_status = loaded_le8.transform(job_status)
loaded_le8.fit(['Self employed', 'Government Dependent', 'Formally employed Private',
 'Informally employed', 'Formally employed Government',
 'Farming and Fishing', 'Remittance Dependent', 'Other Income',
 'Dont Know/Refuse to answer', 'No Income'])

# Load the scaler
scaler = joblib.load('scaler.pkl')

input_data = None
prediction = None 

# Combine the features for prediction
data_to_scale = np.array([household_size, age_of_respondent]).reshape(1, -1)

encoded_data = [
    Country,  
    bank_account,  
    location_type,
    cellphone_access,
    gender_of_respondent,
    relationship_with_head,
    marital_status,
    education_level,
    job_type
]
encoded_data = np.array(encoded_data).reshape(1, -1)
try:
    input_data = np.concatenate([data_to_scale, encoded_data], axis=1)
    # Continue with prediction...
except Exception as e:
    st.error(f"Error concatenating arrays: {e}")

# Combine scaled data with the encoded categorical features
input_data = np.concatenate([data_to_scale, 
                             [[Country_encoded, bank_account_encoded, location_type, 
                               cellphone_access_encoded, gender_of_respondent, 
                               relationship_with_head, marital_status, 
                               education_level, job_type]]], axis=1)

model = joblib.load('random_forest_model.pkl')

# Add a button to trigger the prediction
if st.button('Predict'):
    if input_data is not None:  # Check if input data is ready
        try:
            # Make the prediction using the model
            prediction = model.predict(input_data)
            
            # Show the result
            st.write(f"Prediction: {prediction[0]}")  
            
            if prediction[0] == 0:
                st.write("The model predicts the client will not churn.")
            else:
                st.write("The model predicts the client will churn.")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            prediction = None  # If an error occurs, set prediction to None
    else:
        st.error("Input data is not ready. Please check the input fields.")
    
# Handle case when prediction is not made
if prediction is None:
    st.write("Prediction was not made yet. Please fill in the form and click 'Predict'.")


