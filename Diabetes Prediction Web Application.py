# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:24:34 2023

@author: Usman Muawa
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('C:/Users/Usman Muawa/Desktop/DeployMachineLearning/DiabetesClassifierModel.sav','rb'))


"""creating function"""

def prediction(input_data):
    numpy_input_data=np.asarray(input_data)
    input_reshaped=numpy_input_data.reshape(1,-1)
    prediction=loaded_model.predict(input_reshaped)
    if prediction[0] == 0:
      return 'No diabetes found! Thanks'
    else:
      return 'Yes diabetes Found! Please contact to doctor at early stage'
  
    
def main():
    """Input from users"""
    st.title(" Diabetes Prediction Web Application ")
    if st.button("Name"):
           name="Muhammad Usman Aslam Muawa"
    #getting input
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age	
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input("Glucose_Level")
    BloodPressure=st.text_input("Blood Pressure Value")
    SkinThicknessValue=st.text_input("Skin Thickness Value")
    Insulin=st.text_input("Insulin Value")
    BMI=st.text_input("BMI Value")
    DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function Value")
    Age=st.text_input("Age of Person")
    
    diagnosis=''
    #Creating button for prediction
    if st.button("Diabetes Result"):
        diagnosis=prediction([Pregnancies,Glucose,BloodPressure,SkinThicknessValue,Insulin,BMI,DiabetesPedigreeFunction,Age])

        
    st.success(diagnosis)
    

if __name__ =='__main__':
    main()
