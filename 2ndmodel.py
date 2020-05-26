import pandas as pd

def read_train_data():
	df=pd.read_csv('ind.csv')
	# valuesToBeRemoved=['return1','return2','return3','return4','timestamp']
	valuesToBeRemoved=['return1','return2','return3','return4']

	df2=df['return1'].copy()
	df2[(df2>-0.5) & (df2<0.5)]=0
	df2[df2>=0.5]=1
	df2[df2<=-0.5]=2

	df=df.set_index('timestamp')
	df=df.drop(columns=valuesToBeRemoved)
	df['open']=df['open']/df['close']
	df['high']=df['high']/df['close']
	df['low']=df['low']/df['close']
	df['close']=1
	df['volume']=1
	return df,df2

train_X,train_Y=read_train_data()
print(len(train_X))
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
model = keras.Sequential([
	keras.layers.Dense(132, activation='relu'),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(3)
])
model.compile(optimizer='adam',
			loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics=['accuracy',tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)])
x=train_X[:33000].to_numpy()
y=train_Y[:33000].to_numpy()
model.fit(x,y,epochs=3)
x_test=train_X[33000:].to_numpy()
y_true=train_Y[33000:].to_numpy()
y_pred=model.predict(x_test)
print(y_pred)
print(confusion_matrix(y_true, y_pred))

