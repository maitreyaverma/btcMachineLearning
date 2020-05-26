import tensorflow.compat.v1 as tf
import pandas as pd
import tensorflow
tf.disable_v2_behavior() 

def read_train_data():
	df=pd.read_csv('ind.csv')
	# valuesToBeRemoved=['return1','return2','return3','return4','timestamp']
	valuesToBeRemoved=['return1','return2','return3','return4']

	df2=df['return1']
	df=df.set_index('timestamp')
	df=df.drop(columns=valuesToBeRemoved)
	df['open']=df['open']/df['close']
	df['high']=df['high']/df['close']
	df['low']=df['low']/df['close']
	df['close']=1
	df['volume']=1
	return df,df2

def create_placeholders():
	x=tf.placeholder(tf.float32, name = "x")
	y=tf.placeholder(tf.float32, name = "y")
	return x,y

def initialize_parameters():
	W1 = tf.get_variable("W1", [25,132], initializer = tf.initializers.glorot_normal())
	b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [12,25], initializer = tf.initializers.glorot_normal())
	# W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

	b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [3,12], initializer = tf.initializers.glorot_normal())
	b3 = tf.get_variable("b3", [3,1], initializer = tf.zeros_initializer())
	### END CODE HERE ###

	parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2,"W3": W3,"b3": b3}
	return parameters

def forward_propagation(X_given, parameters):
	X=tf.transpose(X_given)
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	
	Z1 = tf.add(tf.matmul(W1,X),b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2,A1),b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3,A2),b3)
	### END CODE HERE ###
	
	return Z3

def calc_loss(logits,y):
	labels=tf.transpose(y)
	cost = tensorflow.keras.losses.MSE(logits,labels)
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels =labels))
	return cost


train_X,train_Y=read_train_data()
x,y=create_placeholders()
parameters=initialize_parameters()
result=forward_propagation(x,parameters)
cost=calc_loss(result,y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost)

xval=train_X[5:7].to_numpy()
yval=train_Y[5:7].to_numpy()
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	result,cost,_=session.run([result,cost,optimizer],feed_dict={x:xval,y:yval})
	print(result)
	print(cost)