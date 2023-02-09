#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 20:30:53 2023

@author: devansh
"""

import numpy as np
import pickle
loaded_model = pickle.load(open('/Users/devansh/Desktop/Analysis of University Admissions Data/trained_model.sav', 'rb'))
input_data = (300,113,3,3,3.5,8.65,1)
input_data_as_nparray = np.asarray(input_data)
input_data_reshape =input_data_as_nparray.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshape)
print(prediction)