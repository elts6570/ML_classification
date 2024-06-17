import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


def load_data():
	"""
	This function loads the galaxy color and redshift data.

	Returns
	-------
	x : array_like
		galaxy u-g, g-r, r-i, i-z colors
	y : array_like
		galaxy redshifts
	"""
	f = np.loadtxt("data/SDSS_data.csv", skiprows=1, delimiter=',')
	f = np.array(f)

	u = f[:, 0]; g = f[:, 1]; r = f[:, 2]; i = f[:, 3]; z = f[:, 4]; redshift = f[:, 6]

	# Throw away missing entries

	ind = np.where((u!=-9999)&(g!=-9999)&(r!=-9999)&(i!=-9999)&(z!=-9999))[0]

	x = np.zeros((ind.shape[0], 4))
	x[:, 0] = u[ind] - g[ind]
	x[:, 1] = g[ind] - r[ind]
	x[:, 2] = r[ind] - i[ind]
	x[:, 3] = i[ind] - z[ind]
	y = redshift[ind] 
	return x, y


def split_data(x, y):
	""" 
	This function splits the dataset into training data and test data.

	Returns
	-------
	xtrain : array_like
	    the independent variables used for training
	xtest : array_like
	    the test variables used for testing
	ytrain : array_like
	    the dependent variables used for training
	ytest : array_like
	    the dependent variables used for testing
	"""
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=12345)
	return xtrain, xtest, ytrain, ytest


def construct_regressor(n_neighbors):
	"""
	This function constructs the kNN regressor.

	Parameters
	----------
	n_neighbors : int
		number of nearest neighbors to consider

	Returns
	-------
	knn : sklearn.neighbors._regression.KNeighborsRegressor
		the kNN regressor
	"""
	knn = KNeighborsRegressor(n_neighbors=n_neighbors)
	return knn


def fit_model(knn, xtrain, ytrain):
	""" 
	This function fits the model to the training data.

	Returns
	-------
	model : sklearn.neighbors._regression.KNeighborsRegressor
	    the trained model
	"""
	model = knn.fit(xtrain, ytrain)
	return model 


def predict(model, xtest):
	"""
	This function compares the dependent variables predicted by the model to their 
	true values and prints the accuracy score.

	Returns
	-------
	y_pred_test : array_like
	    the dependent variable predicted by the model
	"""
	y_pred_test = model.predict(xtest)
	return y_pred_test


def rms_error(ytest, y_pred_test):
	"""
	This function computes the root mean squared error.

	Returns
	-------
	rms : double
		root mean squared error
	"""
	rms = np.sqrt(np.mean((ytest - y_pred_test) ** 2))
	return rms


def loss(n_neighbors):
	"""
	This function takes as input the number of nearest neighbors
	and returns the root mean square error.
	"""
	x, y = load_data()
	xtrain, xtest, ytrain, ytest = split_data(x, y)

	knn = construct_regressor(n_neighbors=n_neighbors)
	model = fit_model(knn, xtrain, ytrain)
	y_pred_test = predict(model, xtest)
	rms = rms_error(ytest, y_pred_test)
	return rms


def main():
	"""
	Here, we visualize the root mean square error as a function
	of number of nearest neighbors.
	"""
	NN = np.arange(1, 20)

	rmss = np.zeros(len(NN))

	for i in range(len(NN)):
		rmss[i] = loss(NN[i])

	plt.scatter(NN, rmss)
	plt.xlabel("number of nearest neighbours")
	plt.ylabel("rms")
	plt.show()


if __name__ == "__main__":
	main()