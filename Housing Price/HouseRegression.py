

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
import os
print(os.listdir("../input"))
print("Success")

# Any results you write to the current directory are saved as output.


# importing models/layers
from keras.models import Sequential
from keras.layers import Dense
print("Success")

my_data = pd.read_csv('../input/kc_house_data.csv')
my_data.head()

#Splitting Data Up
predictors = my_data.drop(columns=["price","date"])
output = my_data['price']
print("Success")

model = Sequential()
n_cols = predictors.shape[1]
print("Success")


#Dense Layers
model.add(Dense(5,activation ="relu", input_shape=(n_cols,)))
model.add(Dense(5,activation ="relu"))
model.add(Dense(1))
print("Success")


#Optimizer
model.compile(optimizer="adam", loss ="mean_squared_error")
print("Success")


#fitting
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=3)
model.fit(predictors,output,validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

#prediction
prediction = model.predict()
