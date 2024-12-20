''' This script is used to build a neural network model to predict churn rate of customers '''


''' Importing the libraries '''
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


''' Load the data '''
filename_data = "Churn_Modelling.csv"
df_churn = pd.read_csv(os.path.join(os.getcwd(), 'data', filename_data))

df_churn.head()
df_churn.info()
df_churn.describe()


'''
import tensorflow
print(tensorflow.__version__)
print(tensorflow.config.list_physical_devices())
print(tensorflow.config.list_physical_devices('GPU'))
'''

''' Data Preprocessing '''
cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
col_target = "Exited"

X = df_churn.drop(cols_to_drop + [col_target], axis=1).values
y = df_churn[col_target].values

df_X = df_churn.drop(cols_to_drop + [col_target], axis=1)

label_encoder=LabelEncoder()
X[:,2]=label_encoder.fit_transform(X[:,2])
column_transformer=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(column_transformer.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)