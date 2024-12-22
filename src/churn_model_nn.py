''' This script is used to build a neural network model to predict churn rate of customers '''


''' Importing the libraries '''
import os  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from tensorflow.python.keras.models import Sequential  # noqa: E402
from tensorflow.python.keras.layers import Dense  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, LabelEncoder  # noqa: E402


''' Load the data '''
filename_data = "Churn_Modelling.csv"
df_churn = pd.read_csv(os.path.join(os.getcwd(), 'data', filename_data))

df_churn.head()
df_churn.info()
df_churn.describe()

cols_churn = df_churn.columns
cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
col_target = "Exited"

# Categorical columns.
col_geo = "Geography"
col_gender = "Gender"


'''
import tensorflow
print(tensorflow.__version__)
print(tensorflow.config.list_physical_devices())
print(tensorflow.config.list_physical_devices('GPU'))
'''

''' Data Preprocessing '''
# Encoding categorical data.
for col in [col_geo, col_gender]:
    # df_churn[col] = df_churn[col].astype('category')
    df_churn[col] = pd.Categorical(df_churn[col])
    print(df_churn[col].cat.categories)

X = pd.get_dummies(data=df_churn.drop(cols_to_drop + [col_target], axis=1), dtype=int)
y = df_churn[[col_target]]

# Split the data into train and test sets.
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=14, stratify=y)

# print(y.value_counts(normalize=True))
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

print(X_train.head(), X_test.head(), sep='\n')

# Standardize the data.
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.transform(X=X_test)

print(sc.get_feature_names_out())
print(X_train, X_test, sep='\n')

