# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:08:23 2021

@author: lenovo
"""
import streamlit as st
import pickle
import numpy as np



def predict_type(Radius_Mean,Perimeter_Mean,Area_Mean,Concavity_Mean,Concave_Points_Mean,Radius_Worst, Perimeter_Worst,Area_Worst,Concavity_Worst,Concave_Points_Worst):
    model=pickle.load(open(r"breast_cancer_prediction_model.pkl","rb"))
    input_array=np.array([[Radius_Mean, Perimeter_Mean, Area_Mean, Concavity_Mean,
       Concave_Points_Mean,Radius_Worst, Perimeter_Worst,Area_Worst,Concavity_Worst,Concave_Points_Worst]]).astype(np.float64)
    prediction=model.predict(input_array)
    return "Malignant" if prediction else "Benign"  

def main():
    st.title("Breast Cancer Type Prediction")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Breast Cancer Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Radius_Mean = st.text_input("What is the Mean Radius of the Tumour?")
    Perimeter_Mean = st.text_input("What is the Mean Perimeter of the Tumour?")
    Area_Mean = st.text_input("What is the Mean Area of the Tumour?")
    Concavity_Mean = st.text_input("What is the Mean Concavity of the Tumour?")
    Concave_Points_Mean = st.text_input("What is the Mean Concave Points of the Tumour?")
    Radius_Worst = st.text_input("What is the Worst Radius of the Tumour?")
    Perimeter_Worst = st.text_input("What is the Worst Perimeter of the Tumour?")
    Area_Worst = st.text_input("What is the Worst Area of the Tumour?")
    Concavity_Worst = st.text_input("What is the Worst Concavity of the Tumour?")
    Concave_Points_Worst = st.text_input("What is the Worst Concave Points of the Tumour?")


   

    if st.button("Predict"):
        output=predict_type(Radius_Mean, Perimeter_Mean, Area_Mean, Concavity_Mean,
       Concave_Points_Mean,Radius_Worst, Perimeter_Worst,Area_Worst,Concavity_Worst,Concave_Points_Worst)
        st.success('The Breast cancer Type  will be {}'.format(output))


if __name__=='__main__':
    main()