import random
import csv
import itertools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras import backend as K

pooledweightspath = 'model/pool.h5'
modelpath = 'model/best'

def cycle(pretrain=True):

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(16, name='dense_1', input_shape=(3, ), activation='relu'))
	model.add(tf.keras.layers.Dense(16, name='dense_2', activation='relu'))
	model.add(tf.keras.layers.Dense(8, name='dense_3', activation='relu'))
	model.add(tf.keras.layers.Dense(4, name='dense_4', activation='relu'))
	model.add(tf.keras.layers.Dense(1, name='dense_5', activation='softplus'))

	def r2_metric(y_true, y_pred):
		SS_res =  K.sum(K.square(y_true - y_pred)) 
		SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
		return (1 - SS_res/(SS_tot + K.epsilon()))

	adam = tf.keras.optimizers.Adam()
	loss = tf.keras.losses.MSE
	metrics = [tf.keras.metrics.MAE, r2_metric]
	model.compile(adam, loss=loss, metrics=metrics)


	earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
		min_delta=0, patience=500)
	checkpoint = tf.keras.callbacks.ModelCheckpoint(pooledweightspath, monitor='val_loss', verbose=1,
		save_best_only=True) 

	'''
	checkpoint = tf.keras.callbacks.ModelCheckpoint(pooledweightspath, monitor='val_loss', verbose=1,
		save_best_only=True, save_weights_only=True) 
	'''
	callbacks = [earlystopping, checkpoint]



	# Round 1 : Pool training
	# Data preparation

	filename = 'insurance.csv'

	with open(filename, mode='rt', newline='') as f:
		reader = csv.reader(f, delimiter=',')
		scheme = next(reader)
		raw = [row for row in reader]

	# in-place shuffle raw data
	random.shuffle(raw)
	raw = np.array(raw)

	# Separate categorical and numerical variables
	categoricals = ['sex', 'smoker', 'region']
	numericals = ['age', 'bmi', 'children']
	target = 'charges'
	reverse_scheme = {f:i for i, f in enumerate(scheme)}

	numerical_indices = [reverse_scheme[item] for item in numericals]
	target_index = reverse_scheme[target]
	all_indices = [target_index] + numerical_indices

	# data : ['charges, 'age', 'bmi', 'children']
	data = raw[:, all_indices].astype(float)


	def lower_upper_bound(array):
		q1 = np.percentile(array, 25)
		q3 = np.percentile(array, 75)
		lower_bound = 2.5 * q1 - 1.5 * q3
		upper_bound = 2.5 * q3 - 1.5 * q1
		return lower_bound, upper_bound

	def min_max(array):
		return min(array), max(array)

	# Exclude mild outliers != [Q1-1.5*IQ, Q3+1.5*IQ] = [2.5*Q1-1.5*Q3, 2.5*Q3-1.5*Q1]
	bounds = [lower_upper_bound(data[:, i]) for i in range(len(all_indices))]

	total_indices = list(range(len(data)))
	invalid_indices = []
	for i, row in enumerate(data):
		for j, item in enumerate(row):
			lower_bound, upper_bound = bounds[j]
			if item < lower_bound or item > upper_bound:
				invalid_indices.append(i)
	valid_indices = sorted(list(set(total_indices) - set(invalid_indices)))

	data = data[valid_indices, :]


	# Uniformly normalize data
	extremes = [min_max(data[:, i]) for i in range(len(all_indices))]
	for i, row in enumerate(data):
		for j, item in enumerate(row):
			a_min, a_max = extremes[j]
			delta = a_max - a_min
			if delta != 0:
				data[i, j] = (item - a_min) / delta

	data_x, data_y = data[:, 1:], data[:, 0]
	val_split = 0.2
	test_indices = random.sample(list(range(len(data_x))), int(val_split * len(data_x)))
	train_indices = list(set(range(len(data_x))) - set(test_indices))

	train_x = data_x[train_indices]
	train_y = data_y[train_indices]
	test_x = data_x[test_indices]
	test_y = data_y[test_indices]


	history = model.fit(train_x, train_y, batch_size=32, epochs=100000, callbacks=callbacks, validation_data=(test_x, test_y))
	# dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])


	# plot training history
	mae = history.history['mean_absolute_error']
	val_mae = history.history['val_mean_absolute_error']
	r2 = history.history['r2_metric']
	val_r2 = history.history['val_r2_metric']

	pooled_val_r2_max = max(val_r2)

	'''
	plt.plot(list(range(len(mae))), mae, label='train')
	plt.plot(list(range(len(val_mae))), val_mae, label='test')
	plt.xlabel("epoch")
	plt.ylabel("mae")
	plt.title("Pooled training")
	plt.legend()
	plt.show()


	plt.plot(list(range(len(r2))), r2, label='train')
	plt.plot(list(range(len(val_r2))), val_r2, label='test')
	plt.xlabel("epoch")
	plt.ylabel("r2")
	plt.ylim((0, 1))
	plt.title("Pooled training")
	plt.legend()
	plt.show()

	model = tf.keras.models.load_model(pooledweightspath, custom_objects={'r2_metric' : r2_metric})
	y_predict = model.predict(test_x)
	x = np.linspace(0, 1, num=100)
	plt.scatter(test_y, y_predict)
	plt.plot(x, x, label='y_true = y_predict')
	plt.xlabel("y_true")
	plt.ylabel("y_predict")
	plt.title("Pooled prediction")
	plt.legend()
	plt.show()
	'''

	# Subgroup training with common low level feature layer
	#['sex', 'smoker', 'region']

	# in-place shuffle raw data
	raw = raw.tolist()
	random.shuffle(raw)
	raw = np.array(raw)

	categorical_indices = [reverse_scheme[item] for item in categoricals]
	data = raw[valid_indices, :]
	tups = set(tuple(row) for row in data[:, categorical_indices])
	dataset = {tup:[] for tup in tups}

	for row in data:
		tup = tuple(row[categorical_indices])
		dataset[tup].append(row[all_indices])

	for tup in tups:
		dataset[tup] = np.array(dataset[tup]).astype(float)

	tup = ('male', 'no', 'northwest')
	data = dataset[tup]

	# Exclude mild outliers != [Q1-1.5*IQ, Q3+1.5*IQ] = [2.5*Q1-1.5*Q3, 2.5*Q3-1.5*Q1]
	total_indices = list(range(len(data)))
	invalid_indices = []
	for i, row in enumerate(data):
		for j, item in enumerate(row):
			lower_bound, upper_bound = bounds[j]
			if item < lower_bound or item > upper_bound:
				invalid_indices.append(i)
	valid_indices = sorted(list(set(total_indices) - set(invalid_indices)))

	data = data[valid_indices, :]

	# Uniformly normalize data
	for i, row in enumerate(data):
		for j, item in enumerate(row):
			a_min, a_max = extremes[j]
			delta = a_max - a_min
			if delta != 0:
				data[i, j] = (item - a_min) / delta

	data_x, data_y = data[:, 1:], data[:, 0]
	val_split = 0.2
	test_indices = random.sample(list(range(len(data_x))), int(val_split * len(data_x)))
	train_indices = list(set(range(len(data_x))) - set(test_indices))

	# Split train and test data
	train_x = data_x[train_indices]
	train_y = data_y[train_indices]
	test_x = data_x[test_indices]
	test_y = data_y[test_indices]


	# Build a new specified model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(16, name='dense_1', input_shape=(3, ), activation='relu'))
	model.add(tf.keras.layers.Dense(16, name='dense_2', activation='relu'))
	model.add(tf.keras.layers.Dense(8, name='dense_3', activation='relu'))
	model.add(tf.keras.layers.Dense(4, name='new_dense_4', activation='relu'))
	model.add(tf.keras.layers.Dense(1, name='new_dense_5', activation='softplus'))

	if pretrain:
		model.load_weights(pooledweightspath, by_name=True)

	model.compile(adam, loss=loss, metrics=metrics)

	earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
		min_delta=0, patience=500)
	checkpoint = tf.keras.callbacks.ModelCheckpoint(modelpath, monitor='val_loss', verbose=1,
		save_best_only=True) 

	callbacks = [earlystopping, checkpoint]

	history = model.fit(train_x, train_y, batch_size=32, epochs=100000, callbacks=callbacks, validation_data=(test_x, test_y))

	# plot training history
	mae = history.history['mean_absolute_error']
	val_mae = history.history['val_mean_absolute_error']
	r2 = history.history['r2_metric']
	val_r2 = history.history['val_r2_metric']
	sub_val_r2_max = max(val_r2)

	'''
	plt.plot(list(range(len(mae))), mae, label='train')
	plt.plot(list(range(len(val_mae))), val_mae, label='test')
	plt.xlabel("epoch")
	plt.ylabel("mae")
	plt.title("Subgroup training")
	plt.legend()
	plt.show()


	plt.plot(list(range(len(r2))), r2, label='train')
	plt.plot(list(range(len(val_r2))), val_r2, label='test')
	plt.xlabel("epoch")
	plt.ylabel("r2")
	plt.ylim((0, 1))
	plt.title("Subgroup training")
	plt.legend()
	plt.show()


	model = tf.keras.models.load_model(modelpath, custom_objects={'r2_metric' : r2_metric})
	y_predict = model.predict(test_x)
	x = np.linspace(0, 1, num=100)
	plt.scatter(test_y, y_predict)
	plt.plot(x, x, label='y_true = y_predict')
	plt.xlabel("y_true")
	plt.ylabel("y_predict")
	plt.title("Subgroup prediction")
	plt.legend()
	plt.show()
	'''

	return pooled_val_r2_max, sub_val_r2_max

if __name__ == '__main__':
	r2_pretrain = np.array([cycle() for _ in range(10)])
	r2_novel = np.array([cycle(pretrain=False) for _ in range(10)])
	mean_r2_pretrain = np.mean(r2_pretrain[:, 1])
	mean_r2_novel = np.mean(r2_novel[:, 1])
	print("mean_r2_pretrain : ", mean_r2_pretrain)
	print("mean_r2_novel : ", mean_r2_novel)
