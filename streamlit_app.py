import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score
import streamlit.components.v1 as components
import shap



#diabetes website
st.title("Diabetes Prediction App")


load_model = pickle.load(open('trained_model.sav', 'rb'))



#input
input = []

#1 attribute
admission_types = {
    'Emergency': 1,
    'Urgent': 2,
    'Elective': 3,
    'Newborn': 4,
    'Not Available': 5,
    'NULL': 6,
    'Trauma Center': 7,
    'Not Mapped': 8
}
selected_id = st.selectbox(
    'Select an admission type ID', list(admission_types.keys()))
selected_desc = admission_types[selected_id]
st.write('You selected:', selected_id, '(', selected_desc, ')')
input.append(selected_desc)



#second attribute
admission_source_dict = {
    "Physician Referral": 1,
    "Clinic Referral": 2,
    "HMO Referral": 3,
    "Transfer from a hospital": 4,
    "Transfer from a Skilled Nursing Facility (SNF)": 5,
    "Transfer from another health care facility": 6,
    "Emergency Room": 7,
    "Court/Law Enforcement": 8,
    "Not Available": 9,
    "Transfer from critical access hospital": 10,
    "Normal Delivery": 11,
    "Premature Delivery": 12,
    "Sick Baby": 13,
    "Extramural Birth": 14,
    "Transfer From Another Home Health Agency": 18,
    "Readmission to Same Home Health Agency": 19,
    "Not Mapped": 20,
    "Unknown/Invalid": 21,
    "Transfer from hospital inpatient/same facility resulting in a separate claim": 22,
    "Born inside this hospital": 23,
    "Born outside this hospital": 24,
    "Transfer from Ambulatory Surgery Center": 25,
    "Transfer from Hospice": 26,
    "NULL": 17
}

selected_id = st.selectbox('Select an admission source ID', list(admission_source_dict.keys()))
selected_desc = admission_source_dict[selected_id]
st.write('You selected:', selected_id, '(', selected_desc, ')')
input.append(selected_desc)

#3
discharge_dispositions = {
'Discharged to home': 1,
'Discharged/transferred to another short term hospital': 2,
'Discharged/transferred to SNF': 3,
'Discharged/transferred to ICF': 4,
'Discharged/transferred to another type of inpatient care institution': 5,
'Discharged/transferred to home with home health service': 6,
'Left AMA': 7,
'Discharged/transferred to home under care of Home IV provider': 8,
'Admitted as an inpatient to this hospital': 9,
'Neonate discharged to another hospital for neonatal aftercare': 10,
'Still patient or expected to return for outpatient services': 12,
'Discharged/transferred within this institution to Medicare approved swing bed': 15,
'Discharged/transferred/referred another institution for outpatient services': 16,
'Discharged/transferred/referred to this institution for outpatient services': 17,
'NULL': 18,
'Discharged/transferred to another rehab fac including rehab units of a hospital.': 22,
'Discharged/transferred to a long term care hospital.': 23,
'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.': 24,
'Not Mapped': 25,
'Unknown/Invalid': 26,
'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere': 30,
'Discharged/transferred to a federal health care facility.': 27,
'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital': 28,
'Discharged/transferred to a Critical Access Hospital (CAH).': 29
}

selected_id = st.selectbox('Select an discharge disposition ID', list(discharge_dispositions.keys()))
selected_desc = discharge_dispositions[selected_id]
st.write('You selected:', selected_id, '(', selected_desc, ')')
input.append(selected_desc)

#4
time_in_hospital = st.number_input("Enter the time spent in hopsital(in days):")
st.write(f"Entered {time_in_hospital}")
time_in_hospital = round(time_in_hospital)
input.append(time_in_hospital)

#5
number_diagnosis = st.number_input("Enter the number of diagnosis done:")
st.write(f"Entered {number_diagnosis}")
number_diagnosis = round(number_diagnosis)
input.append(number_diagnosis)

#6 and 7
icd={
    'Complications of Pregnancy, Childbirth, and the Puerperium': 0,
    'Congenital Anomalies': 1,
    'Diseases of the Circulatory System': 2,
    'Diseases of the Digestive System': 3,
    'Diseases of the Genitourinary System': 4,
    'Diseases of the Musculoskeletal System and Connective Tissue': 5,
    'Diseases of the Respiratory System': 6,
    'Diseases of the Skin and Subcutaneous Tissue': 7,
    'Injury and Poisoning': 8,
    'Other': 9,
    'Symptoms, Signs, and Ill-Defined Conditions': 10
}

selected_id = st.selectbox('Select the primary health problem:', list(icd.keys()))
selected_desc = icd[selected_id]
st.write('You selected:', selected_id, '(', selected_desc, ')')
input.append(selected_desc)


selected_id = st.selectbox('Select the seco=ondary health problem:', list(icd.keys()))
selected_desc = icd[selected_id]
st.write('You selected:', selected_id, '(', selected_desc, ')')
input.append(selected_desc)

#8
number_medication = st.number_input("Number of distinct generic names administered:")
st.write(f"Entered {number_medication}")
number_medication=round(number_medication)
input.append(number_medication)

#9

age = {'0-10': 1, 
       '11-20': 2, 
       '21-30': 3, 
       '31-40': 4, 
       '41-50': 5, 
       '51-60': 6, 
       '61-70': 7, 
       '71-80': 8, 
       '81-90': 9, 
       '91+': 10
    }

selected_id = st.selectbox('Select the primary health problem:', list(age.keys()))
selected_desc = age[selected_id]
st.write('You selected:', selected_id, '(', selected_desc, ')')
input.append(selected_desc)


#10
gender={
    'Female':0,
    'Male':1
}
selected_id = st.selectbox('Select the primary health problem:', list(gender.keys()))
selected_desc = gender[selected_id]
st.write('You selected:', selected_id, '(', selected_desc, ')')
input.append(selected_desc)

#11
number_inpatient=st.number_input("Number of emergency visits of the inpatient in the year preceding the encounter:")
st.write(f"Entered {number_inpatient}")
number_inpatient=round(number_inpatient)
input.append(number_inpatient)

#12
number_outpatient=st.number_input("Number of emergency visits of the outpatient in the year preceding the encounter:")
st.write(f"Entered {number_outpatient}")
number_outpatient=round(number_outpatient)
input.append(number_outpatient)



# MODEL
input_df = pd.DataFrame([input], columns=['admission_type_id', 'admission_source_id' ,
                                          'discharge_disposition_id', 'time_in_hospital', 
                                          'number_diagnoses', 'diag_1', 'diag_2' ,
                                          'num_medications', 'age', 'gender', 
                                          'number_inpatient', 'number_outpatient'])

pred = load_model.predict(input_df)
print(pred)
st.write(' ')
st.write(' ')
if (pred == 0):
    st.write('### **Readmission chance is high**')
else:
    st.write('### **Chance of readmission is low**')


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.write(' ')
st.write(' ')
st.write(' ')

explainer = shap.TreeExplainer(load_model)

# Calculate Shap values
shap_values = explainer.shap_values(input_df)
shap.initjs()
pred = load_model.predict(input_df)
print(pred)
st.write(' ')
st.write(' ')

st.write(' ')
st.write(' ')
st.write(' ')

explainer = shap.TreeExplainer(load_model)

# Calculate Shap values
shap_values = explainer.shap_values(input_df)
shap.initjs()

st.write('### Contributions of the values are as such (Here scale is 0-1):')
st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_df), 400)
