import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
    """
    This function loads the density and galaxy formation data. 

    Returns
    -------
    mstars : array_like
        stellar mass in Msun as a function of redshift
    mstardot : array_like
        star-formation rate in Msun / year
    TYPE : array_like
        galaxy type (central (0)/ satellite (1)/ orphan) (2))
    DIST : array_like
        comoving distance in Mpc
    ra : array_like
        right ascension in deg
    dec : array_like
        declination in deg
    density: : array_like
        the 3D dark matter density
    """
    with h5py.File("/Users/eleni/Desktop/snsim/sibelius_dark_1_grid256.hdf5", "r") as f:
        density = f['delta_field_dm'][::]
        density = np.swapaxes(density, 2, 0)

    f = np.load("/Users/eleni/Desktop/snsim/PROPERTIES_low_mass.npz")

    ind = 100000

    mstars=f['mstars'][:ind, :]
    mstardot=f['mstardot'][:ind, :]*1e-9
    TYPE=f['TYPE'][:ind, :]
    DIST=f['DIST'][:ind]
    ra=f['ra'][:ind]
    dec=f['dec'][:ind]  
    return mstars, mstardot, TYPE, DIST, ra, dec, density


def _ang2vec(theta, phi):
    """
    This is the healpy ang2vec function. Required for local compatibility.

    Parameters
    ----------
    theta : array_like
        declination in deg
    phi : array_like
        right ascension in deg

    Internal Parameters:
    --------------------
    u1 : array_like
        x component of the unit vector
    u2 : array_like
        y component of the unit vector
    u3 : array_like
        z component of the unit vector

    Returns:
    u : array_like
        the unit vector in the direction of a given ra, dec
    """
    theta = np.pi / 2.0 - np.radians(phi)
    phi = np.radians(theta)

    u1 = np.sin(theta) * np.cos(phi)
    u2 = np.sin(theta) * np.sin(phi)
    u3 = np.cos(theta)

    u = np.array([u1, u2, u3]).T
    return u


def get_gridded_density(ra, dec, DIST, density):
    """
    This function returns the density at the location of galaxies. Here, we use Nearest Grid
    Point projection to assign galaxies to a 3D box.

    Parameters
    ----------
    density: array_like
        3D NxNxN density field

    Internal Parameters
    -------------------
    L : double
        box side length in Mpc
    N : double
        box side resolution
    box : double
        location of the box center
    resolution : double
        side length of a voxel in Mpc

    Returns
    -------
    local_density : array_like
        density at the galaxies' locations
    """
    L = 1000; N = 128; box = -500; resolution = L/N 
    xyz = _ang2vec(ra, dec).T * DIST - box   
    xyz = np.float32(xyz)
    i,j,k = np.floor(xyz/resolution+0.5).astype(int) 
    local_density = density[i,j,k]
    return local_density


def create_dataframe(mstars, mstardot, local_density, TYPE):
    """
    This function creates a data frame from the loaded data.

    Returns
    -------
    x : array_like
        independent variables (galaxy stellar mass, star-formation rate, local density)
    y: array_like
        dependent variable (galaxy type)

    Notes
    -----
    All galaxy properties are taken at redshift z = 0.
    """
    d = {'mstars_0': mstars[:,0], 'mstardot_0': mstardot[:,0], 'local_density':local_density, 'type_0': TYPE[:,0]}
    data = pd.DataFrame(data = d)
    x = np.array(data[["mstars_0", "mstardot_0", "local_density"]])
    y = np.array(data[["type_0"]])
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
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)
    return xtrain, xtest, ytrain, ytest


def fit_model(xtrain, ytrain):
    """ 
    This function fits the model to the training data.

    Returns
    -------
    model : sklearn.ensemble._forest.RandomForestClassifier
        the trained model
    """
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=4, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    model.fit(xtrain, ytrain)
    return model


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
    return y_pred_test


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

    # Add labels to the plot
    class_names = ['central', 'satellite', 'orphan']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()
    return None


def report_diagnostic(ytest, y_pred_test):
    """
    This function prints the classification report.
    """
    print(classification_report(ytest, y_pred_test))
    return None


def importances(model):
    """
    This function prints the individual importances of each feature.
    """
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0)
    forest_importances = pd.Series(importances, index = ["stellar mass", "SFR", "local density"])

    fig, ax = plt.subplots(dpi = 200)
    forest_importances.plot.bar(yerr = std, ax = ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    return None


def main():
    # Data preparation.
    mstars, mstardot, TYPE, DIST, ra, dec, density = load_data()
    local_density = get_gridded_density(ra, dec, DIST, density)
    x, y = create_dataframe(mstars, mstardot, local_density, TYPE)

    # Splitting in test and training data.
    xtrain, xtest, ytrain, ytest = split_data(x, y)

    # Fit model.
    model = fit_model(xtrain, ytrain)

    # Assess performance.
    y_pred_test = accuracy_diagnostic(model, xtest, ytest)
    confusion_diagnostic(ytest, y_pred_test)
    report_diagnostic(ytest, y_pred_test)
    importances(model)


if __name__ == "__main__":
    main()
