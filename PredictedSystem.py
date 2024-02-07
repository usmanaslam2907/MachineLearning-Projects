# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:09:54 2023

@author: Usman Muawa
"""

import numpy as np
import pandas as pd
import pickle
"""from sklearn.preprocessing import StandardScaler"""
loaded_model=pickle.load(open('C:/Users/Usman Muawa/Desktop/DeployMachineLearning/DiabetesClassifierModel.sav','rb'))
input_data=(2,197,70,45,543,30.5,0.158,53)
numpy_input_data=np.asarray(input_data)
input_reshaped=numpy_input_data.reshape(1,-1)
prediction=loaded_model.predict(input_reshaped)
if prediction[0] == 0:
  print("No diabetes found! Thanks")
else:
  print("Yes diabetes Found! Please contact to doctor at early stage")