#!python
"""
Explore using a prior for the GP mean and covariance functions based on the existing data collected.
"""
from __future__ import division

from numpy import sum, exp, array, eye, argmin, linalg, matrix
import numpy.random
from numpy.linalg import LinAlgError
# import tables

# from utils.latinhypercube import lhcSample
# from model.gaussianprocess import GaussianProcess
from ego.gaussianprocess.kernel import GaussianKernel_iso


class GPMeanPrior(object):

    def __init__(self):
        super(GPMeanPrior, self).__init__()
    
    def mu(self, x):
        raise NotImplementedError('GPMeanPrior-derived class does not have mean function implemented')
        

class RBFNMeanPrior(GPMeanPrior):
    """
    Use a trained RBF network as the prior.  It is necessary to train or load
    this prior in order to use it.
    """
    def __init__(self, means=None, beta=None, theta=10., lowerb=None, width=None):
        super(RBFNMeanPrior, self).__init__()
        self.means = means
        self.beta = beta
        self.theta = theta
        self.lowerb = lowerb  # lower bounds
        self.width = width   # dist from lower to upper
    
    def mu(self, x):
        # we are doing computation in the unit hypercube, but x comes from
        # the source space, so we need to translate x
        x = (x - self.lowerb) / self.width
        norms = [linalg.norm(m-x) for m in self.means]
        rbf = array([self.RBF(n) for n in norms])
        return sum(self.beta*rbf)
        
    def negmu(self, x):
        # for optimizers
        return -self.mu(x)

    def RBF(self, r):
        return exp(-self.theta*r**2)
    
    
    def train(self, X, Y, bounds=None, k=10, delta=100, kernel=None, seed=None):
        """
        using the X, Y data, train a Radial Basis Function network as follows:
    
            1.  cluster the data into k clusters using k-means
            2.  using the centroids of each cluster as the origins of the RBFs,
                learn the RBF weights beta
    
        'RBF' is a radial basis function of the form y = rbf(r, args=rbfargs), where
        r = ||c-x|| and 'args' is any necessary arguments for the RBF.
    
        Returns the means and weights of the RNF network.
        """
        def RBFN(means, x):
            norms = [linalg.norm(m-x) for m in means]
            rbf = array([self.RBF(n) for n in norms])
            return rbf
    
        rs = numpy.random.RandomState(seed)
        
        # let's cluster the data using k means.  first, project X to unit hypercube
        if bounds is not None:
            self.lowerb = array([b[0] for b in bounds])
            self.width = array([b[1]-b[0] for b in bounds])
        X = array([(x-self.lowerb)/self.width for x in X])
        
        Y = array(Y)
    
        NX, _ = X.shape
    
        r = range(NX)
        rs.shuffle(r)
        means = X[r[:k]]
    
        for _ in xrange(10):
            # assign each cluster to closest mean
            C = [argmin([linalg.norm(m-x) for m in means]) for x in X]
        
            # means become centroids of their clusters
            means = []
            for j in xrange(k):
                clust = [x for x, c in zip(X, C) if c==j]
                if len(clust)==0:
                    # for empty clusters, restart centered on a random datum
                    means.append(X[rs.randint(NX)])
                else:
                    means.append(array(clust).mean(0))
    
        # okay, now we can analytically compute RBF weights
        if kernel is None:
            kernel = GaussianKernel_iso(array([.2]))
        
        reg = .1
        # L = cholesky(kernel.covMatrix(X))
        while True:
            try:
                invK = linalg.inv(matrix(kernel.covMatrix(X)) + eye(NX)*reg) # add regularizer
            except LinAlgError:
                reg *= 2
                print 'LinAlgError: increase regularizer to %f' % reg
            else:
                break
                
    
        # compute unweighted basis values for the data
        H = matrix([RBFN(means, x) for x in X])
        Y = matrix(Y)
    
        try:
            beta = linalg.inv(H.T*invK*H + delta**-2) * (H.T*invK*Y.T)
        except linalg.linalg.LinAlgError:
            # add a powerful regularizer...
            beta = linalg.inv(H.T*invK*H + eye(H.shape[1]) + delta**-2) * (H.T*invK*Y.T)

        ## this is from Ruben's paper.  I can't get it to make sense...
        # a = b = 10.
        # sighat = diag((b + Y*K.I*Y.T - (H.T*K.I*H + delta**-2) * (muhat*muhat.T)) / (NX+a+2))

        # beta should be a 1D array
        self.beta = array(beta).reshape(-1)
        self.means = means
        