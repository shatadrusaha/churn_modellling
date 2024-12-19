''' This script is used to build a neural network model to predict churn rate of customers '''

''' Importing the libraries '''
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score

''' Load the data '''
filename_data = "Churn_Modelling.csv"
df_churn = pd.read_csv(os.path.join(os.getcwd(), 'data', filename_data))

df_churn.head()

''' Data Preprocessing '''
cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
col_target = "Exited"

X = df_churn.drop(cols_to_drop + [col_target], axis=1)
y = df_churn[col_target]
