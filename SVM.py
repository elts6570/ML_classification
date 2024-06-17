import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
	"""
    This function loads the galaxy color and morphology data.

    Returns
    -------
    features : array_like
    	galaxy colours in four bands
    morphology : array_like
    	galaxy morphology (merger (0) / elliptical (1) / spiral (2))
    """
	data = np.load("data/galaxy_catalogue.npy")

	features = np.empty(shape=(len(data), 4))
	features[:, 0] = data['u-g']
	features[:, 1] = data['g-r']
	features[:, 2] = data['r-i']
	features[:, 3] = data['i-z']
	
	morphology = data['class']
	
	merger = np.where(morphology=='merger')[0]
	elliptical = np.where(morphology=='elliptical')[0]
	spiral = np.where(morphology=='spiral')[0]
	
	morphology[merger] = 0
	morphology[elliptical] = 1
	morphology[spiral] = 2
	
	morphology = morphology.astype(int)
	return features, morphology


def split_data(features, morphology):
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
	xtrain, xtest, ytrain, ytest = train_test_split(features, morphology, test_size=0.5)
	return xtrain, xtest, ytrain, ytest


def construct_svm(decision_function_shape, kernel):
	"""
	This function constructs the support vector machine.

	Parameters
    ----------
    decision_function_shape : string
    	shape of the decision function, choose among 'ovo', 'ovr'
    kernel : string
    	shape of kernel, choose among ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

    Return
    ------
    SVM : sklearn.svm._classes.SVC
    	the support vector machine
	"""
	SVM = svm.SVC(decision_function_shape = decision_function_shape, kernel = kernel)
	return SVM

def fit_model(SVM, xtrain, ytrain):
	"""
	This function fits the model to the train data.

    Return
    ------
    model : sklearn.svm._classes.SVC
    	the model that is fitted to the data
	"""
	model = SVM.fit(xtrain, ytrain)
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


def accuracy_diagnostic(model, xtest, ytest):
    """
    This function compares the dependent variables predicted by the model to their 
    true values and prints the accuracy score.

    Returns
    -------
    y_pred_test : array_like
        the dependent variable predicted by the model
    """
    y_pred_test = model.predict(xtest)
    print("Accuracy score:", accuracy_score(ytest, y_pred_test))
    return None


def confusion_diagnostic(ytest, y_pred_test):
	"""
    This function builds the confusion matrix and plots it.
    """
	matrix = confusion_matrix(ytest, y_pred_test)
	matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

	plt.figure(figsize=(16,7))
	sns.set(font_scale=1.4)
	sns.heatmap(matrix, annot=True, annot_kws={'size':10},
			cmap=plt.cm.Greens, linewidths=0.2)

	class_names = ['merger', 'elliptical', 'spiral']
	tick_marks = np.arange(len(class_names))
	tick_marks2 = tick_marks + 0.5
	plt.xticks(tick_marks, class_names, rotation=25)
	plt.yticks(tick_marks2, class_names, rotation=0)
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.title('Confusion Matrix for Random Forest Model')
	plt.show()
	return None


def main():
	decision_function_shape = 'ovo'
	kernel = 'linear'

	features, morphology = load_data()
	xtrain, xtest, ytrain, ytest = split_data(features, morphology)
	SVM = construct_svm(decision_function_shape, kernel)
	model = fit_model(SVM, xtrain, ytrain)

	y_pred_test = predict(model, xtest)
	accuracy_diagnostic(model, xtest, ytest)
	confusion_diagnostic(ytest, y_pred_test)


if __name__ == "__main__":
    main()

