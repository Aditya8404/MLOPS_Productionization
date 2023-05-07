import streamlit as st
import numpy as np
from pickle import load

scaler = load(open('models/standard_scaler.pkl', 'rb'))
nb_model = load(open('models/nb_model.pkl', 'rb'))

st.title("Water Potability Prediction Model")

ph = st.slider("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.05)
hd = st.text_input("Hardness", placeholder="Enter value (Range => 70 - 320)")
sd = st.text_input("Solids", placeholder="Enter value (Range => 500 - 55,000)")
sl = st.text_input("Sulphate", placeholder="Enter value (Range => 100 - 500)")
cd = st.text_input("Conductivity", placeholder="Enter value (Range => 200 - 800)")
tm = st.text_input("Trihalomethanes", placeholder="Enter value (Range => 10 - 125)")
cl = st.slider("Chloramines", min_value=1.0, max_value=14.0, value=7.5, step=0.05)
oc = st.slider("Organic Carbon", min_value=2.0, max_value=28.0, value=15.0, step=0.1)
tb = st.slider("Turbidity", min_value=0.0, max_value=7.0, value=3.5, step=0.01)
btn_click = st.button("Predict")

if btn_click == True:
    st.balloons()
    if ph and hd and sd and cl and sl and cd and oc and tm and tb:
        query_point = np.array([float(sl), float(hd), float(sd), float(cl), float(sl), float(cd), float(oc), float(tm), float(tb)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = nb_model.predict(query_point_transformed)

        st.write("---")

        if pred == [1]:
            st.success("Potable")

        else:
            st.warning("Not-Potable")
    else:
        st.error("Enter the values properly.")