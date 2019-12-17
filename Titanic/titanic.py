import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
import handle_data

def read_data():
    df = pd.read_csv('train.csv')
    df.drop(['PassengerId', 'Name'], 1, inplace=True)
    df.infer_objects() # Attempt to infer better dtypes for object columns.
    df.fillna(0, inplace=True)
    df = handle_data.convert_non_numerical(df)
    return df

df = read_data()
X = np.array(df.drop(['Survived'], 1).astype(float))
X = preprocessing.scale(X)

# Find and remove outliers
clf = LocalOutlierFactor(n_neighbors=4)
pred = clf.fit_predict(X)
outliers = np.where(pred == -1)[0]

df = read_data()
df.drop(outliers, inplace=True)

X = np.array(df.drop(['Survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Survived'])

# Neural Network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(X, y, epochs=50)

# Make predictions on test dataset
test = pd.read_csv('test.csv')
test.drop(['PassengerId', 'Name'], 1, inplace=True)
test.infer_objects() # Attempt to infer better dtypes for object columns.
test.fillna(0, inplace=True)
test = handle_data.convert_non_numerical(test)
X_test = np.array(test.astype(float))
X_test = preprocessing.scale(X_test)

# Write submission file
import csv
with open('submission.csv', mode='w') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    for i in range(len(X_test)):
        pred = model.predict(X_test[i].reshape(1,len(X_test[i])))
        survived = np.argmax(pred)
        writer.writerow([i+892, survived])