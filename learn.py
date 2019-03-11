# Machine learning for perception neuron data.
import numpy as np
from scipy.optimize import minimize

class LinearModel(object):
    def __init__(self):
        """
        Produce a separate linear moel for each dimension (correpsonding to column) of target Y.
        """
        pass
        
    def fit(self,X,Y):
        """
        Simple linear model.
        2016-02-20

        Params:
        -------

        """
        nSamples,ndimX = X.shape
                
        # A simple linear model.
        linCoeffs = np.zeros((ndimX+1,3))
        for i in range(Y.shape[1]):
            def f(params):
                return ( (Y[:,i] - np.hstack((X,np.ones((nSamples,1)))).dot(params))**2 ).sum()

            linCoeffs[:,i] = minimize(f,np.random.normal(size=ndimX+1))['x']
            
        self.linCoeffs = linCoeffs
            
    def predict(self,X):
        return X.dot(self.linCoeffs[:-1]) + self.linCoeffs[-1][None,:]

def pca(Y,grouped=False):
    """
    PCA on input data with option for doing PCA on groups of columns independently.
    2017-02-09
    
    Params:
    -------
    Y (ndarray)
        n_samples x n_dim
    grouped (int=False)
        If integer, will do PCA on columns grouped in subsets of grouped's value.
    """
    if grouped:
        L,v,c = [],[],[]
        for i in range(Y.shape[1]//grouped):
            result = pca(Y[:,i*grouped:(i+1)*grouped])
            L.append(result[0])
            v.append(result[1])
            c.append(result[2])
    else:
        c = np.cov(Y.T)

        L,v = np.linalg.eig(c)
        sortix = np.argsort(abs(L))
        v = v[:,sortix[::-1]]
        L = L[sortix[::-1]]

    return L,v,c

