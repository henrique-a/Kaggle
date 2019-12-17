import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Read data
df = pd.read_csv('train.csv')
IMG_SIZE = 28
X_train = np.array([np.zeros([28,28]) for _ in range(len(df))])
y_train = np.zeros(len(df))
for index, row in df.iterrows():
    y_train[index] = row[0]
    X_train[index] = row[1:].values.reshape((IMG_SIZE,IMG_SIZE))

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_train = X_train/255.0

# Neural Network Model
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(10)) # output layer
model.add(Activation('sigmoid'))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=25, validation_split=0.1)

# Make predictions 
df_test = pd.read_csv('test.csv')
IMG_SIZE = 28
X_test = np.array([np.zeros([IMG_SIZE,IMG_SIZE]) for _ in range(len(df_test))])
for index, row in df_test.iterrows():
    X_test[index] = row.values.reshape((IMG_SIZE,IMG_SIZE))

X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test/255.0

# Write submission file
import csv
with open('submission.csv', mode='w') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    for i in range(len(X_test)):
        pred = model.predict(X_test[i].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
        digit = np.argmax(pred)
        writer.writerow([i+1, digit])