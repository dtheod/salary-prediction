import json

import requests
import streamlit as st

st.title("Salary Prediction Web App")


data = {}

data["age"] = st.selectbox(
    "Age",
    options=("25-34", "35-45"),
    help="Employee's Age",
)

data["gender"] = st.selectbox(
    "Gender",
    options=("Male", "Female", "Other"),
    help="Employee's Gender",
)

data["job"] = st.selectbox(
    "Job",
    options=("program manager", "senior program manager"),
    help="Employee's Job",
)

data["country"] = st.selectbox(
    "Country",
    options=("united states", "united kingdom"),
    help="Employee's Country",
)

data["years_field_experience"] = st.number_input(
    "experience",
    min_value=6,
    value=6,
    help="Experience",
)

data["education"] = st.number_input(
    "education",
    min_value=10,
    value=14,
    help="Education",
)

data["senior"] = st.number_input(
    "seniority",
    min_value=0,
    value=1,
    help="seniority",
)

data["principal"] = st.number_input(
    "principal",
    min_value=0,
    value=0,
    help="principal",
)

data["staff"] = st.number_input(
    "staff",
    min_value=0,
    value=0,
    help="staff",
)

data["assistant"] = st.number_input(
    "assistant",
    min_value=0,
    value=0,
    help="assistant",
)

data["intern"] = st.number_input(
    "intern",
    min_value=0,
    value=0,
    help="intern",
)

if st.button("Get the cluster of this customer"):

    data_json = json.dumps(data)

    prediction = requests.post(
        "http://127.0.0.1:5000/predict",
        headers={"content-type": "application/json"},
        data=data_json,
    ).text
    st.write(f"Salary for this individual {prediction}")
