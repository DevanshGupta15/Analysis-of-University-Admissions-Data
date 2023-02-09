#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 20:49:58 2023

@author: devansh
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('/Users/devansh/Desktop/Analysis of University Admissions Data/trained_model.sav', 'rb'))





def chance_prediction (input_data):
    
    
    input_data_as_nparray = np.asarray(input_data)
    input_data_reshape =input_data_as_nparray.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshape)
    return prediction



def main():
    st.title(" University Admissions Chance of Admit ")
    GRE_Score = st.text_input('GRE SCORE')
    TOEFL_Score = st.text_input("TOEFL SCORE")
    University_Rating =st.text_input("University Rating")
    SOP=st.text_input("SOP RATING")
    LOR =st.text_input("LOR Rating")
    CGPA = st.text_input("CGPA out of 10")
    Research = st.text_input("Research Paper Published")
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        diagnosis = chance_prediction([GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research])
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()