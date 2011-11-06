#!/usr/bin/python

# Copyright (C) 2010, 2011 by Eric Brochu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# from numpy import array, sqrt, nan, exp, arange
# from matplotlib.pylab import figure, subplot, plot, clf, poly_between, axvline
# import numpy.random as numpyrandom
# from random import sample, random
import ctypes
import ctypes.util
from ctypes import cdll, POINTER, c_int, c_double, c_char_p
from time import time
# from copy import copy
import pdb

from numpy import *
from numpy.linalg import inv, LinAlgError

from scipy.optimize import fmin_bfgs

from .kernel import GaussianKernel_ard, GaussianKernel_iso, MaternKernel3, MaternKernel5



#############################################################################
# this implementation of erf, cdf and pdf is substantially faster than
# the scipy implementation (a C implementation would probably be faster yet)
#############################################################################
#
# from: http://www.cs.princeton.edu/introcs/21function/ErrorFunction.java.html
# Implements the Gauss error function.
#   erf(z) = 2 / sqrt(pi) * integral(exp(-t*t), t = 0..z)
#
# fractional error in math formula less than 1.2 * 10 ^ -7.
# although subject to catastrophic cancellation when z in very close to 0
# from Chebyshev fitting formula for erf(z) from Numerical Recipes, 6.2
def erf(z):
    t = 1.0 / (1.0 + 0.5 * abs(z))
    # use Horner's method
    ans = 1 - t * exp( -z*z -  1.26551223 +
                                t * ( 1.00002368 +
                                t * ( 0.37409196 + 
                                t * ( 0.09678418 + 
                                t * (-0.18628806 + 
                                t * ( 0.27886807 + 
                                t * (-1.13520398 + 
                                t * ( 1.48851587 + 
                                t * (-0.82215223 + 
                                t * ( 0.17087277))))))))))
    if z >= 0.0:
        return ans
    else:
        return -ans

def CDF(x):
    return 0.5 * (1 + erf((x) * 0.707106))
    
def PDF(x):
    return  exp(-(x**2/2)) * 0.398942



class GaussianProcess(object):

    def __init__(self, kernel, X=None, Y=None, prior=None, noise=.1, gnoise=1e-4, G=None):
        """
        Initialize a Gaussian Process.
        
        @param kernel:       kernel object to use
        @param prior:        object defining the GP prior on the mean.  must 
                             be a descendant of GPMeanPrior
        @param noise:        noise hyperparameter sigma^2_n
        @param X:            initial training data
        @param Y:            initial observations
        """
        self.kernel = kernel
        self.prior = prior
        self.noise = noise
        self.gnoise = array(gnoise, ndmin=1)
        
        self.R = None
        
        if (X is None and Y is not None) or (X is not None and Y is None):
            raise ValueError
            
        self.X = zeros((0,0))
        self.Y = zeros((0))
        
        self.G = None
        
        self.name = 'GP'            # for legend
        self.starttime = time()     # for later analysis

        if X is not None:
            self.addData(X, Y)
            
        self.augR = None
        self.augL = None
        self.augX = None
        
        # mostly for testing/logging
        self.selected = None
        self.endtime = None
        
        # if prior is None:
        #     print 'prior is None'
        # else:
        #     print 'prior is NOT None'
        #     
        # if self.prior is None:
        #     print 'self.prior is None'
        # else:
        #     print 'self.prior is NOT None'
        
        
    def _computeCorrelations(self, X):
        """ compute correlations between data """

        M, (N,D) = len(self.X), X.shape
        r = eye(N, dtype=float) + self.noise
        m = empty((M,N))
        
        for i in xrange(N):
            for j in xrange(i): 
                r[i,j] = r[j,i] = self.kernel.cov(X[i], X[j])
                
        for i in xrange(M):
            for j in xrange(N): 
                m[i,j] = self.kernel.cov(self.X[i], X[j])
                
        return r, m
        
    def _computeAugCorrelations(self, X):
        """ compute correlations between data """

        M, (N,D) = len(self.augX), X.shape
        r = eye(N, dtype=float) + self.noise
        m = empty((M,N))
        
        for i in xrange(N):
            for j in xrange(i): 
                r[i,j] = r[j,i] = self.kernel.cov(X[i], X[j])
                
        for i in xrange(M):
            for j in xrange(N): 
                m[i,j] = self.kernel.cov(self.augX[i], X[j])
                
        return r, m
        

    def posterior(self, X, getvar=True):
        """ Get posterior mean and variance for a point X. """
        if len(self.X)==0:
            if self.prior is None:
                if getvar:
                    return 0.0, 1.0
                else:
                    return 0.0
            else:
                if getvar:
                    return self.prior.mu(X), 1.0
                else:
                    return self.prior.mu(X)
        
        X = array(X, copy=False, dtype=float, ndmin=2)
        M, (N,D) = len(self.X), X.shape
        
        m = 0.0
        if self.prior is not None:
            m = self.prior.mu(X)
            assert isscalar(m)
        
        if self.G is None:
            # NO GRADIENT DATA.
            d = self.Y-m
            r = empty((M, N))
            for i in xrange(M):
                for j in xrange(N): 
                    r[i,j] = self.kernel.cov(self.X[i], X[j])
        else:
            # WITH GRADIENT DATA.
            d = hstack(map(hstack, zip(self.Y-m, self.G)))
            r = empty((M*(D+1), N))
            for i in xrange(M):
                for j in xrange(N):
                    A = i*(D+1)
                    cov = self.kernel.covWithGradients(self.X[i], X[j])
                    r[A:A+D+1,j] = cov[:,0]
        
        # calculate the mean.
        Lr = linalg.solve(self.L, r)
        mu = m + dot(Lr.T, linalg.solve(self.L,d))
        
        if getvar:
            # calculate the variance.
            if self.augL is None:
                sigma2 = (1 + self.noise) - sum(Lr**2, axis=0)
            else:
                M, (N,D) = len(self.augX), X.shape
                r = empty((M, N))
                for i in xrange(M):
                    for j in xrange(N): 
                        r[i,j] = self.kernel.cov(self.augX[i], X[j])
                Lr = linalg.solve(self.augL, r)
                sigma2 = (1 + self.noise) - sum(Lr**2, axis=0)
            sigma2 = clip(sigma2, 10e-8, 10)
        
            return mu[0], sigma2[0]
        else:
            return mu[0]
    
    
    def posteriors(self, X):
        """
        get arrays of posterior values for the array in X
        """
        M = []
        V = []
        for x in X:
            if isscalar(x):
                m, v = self.posterior(array([x]))
            else:
                m, v = self.posterior(x)
            M.append(m)
            V.append(v)
        return array(M), array(V)
        
        
    def mu(self, x):
        """
        get posterior mean for a point x
        
        NOTE: if you are getting the variance as well, this is less efficient
        than using self.posterior()
        """
        return self.posterior(x, getvar=False)
            
            
    def negmu(self, x):
        """
        needed occasionally for optimization
        """
        nm = -self.mu(x)
        # if self.prior is not None and len(self.X)==0:
        #     print 'no data, using prior = %.4f'%nm
        return nm
        
        
    def addData(self, X, Y, G=None):
        """
        Add new data to model and update. 

        We assume that X is an (N,D)-array, Y is an N-vector, and G is either
        an (N,D)-array or None. Further, if X or G are a single D-dimensional
        vector these will be interpreted as (1,D)-arrays, i.e. one observation.
        """
        X = array(X, copy=False, dtype=float, ndmin=2)
        Y = array(Y, copy=False, dtype=float, ndmin=1).flatten()
        G = array(G, copy=False, dtype=float, ndmin=2) if (G is not None) else None

        assert len(Y) == len(X), 'wrong number of Y-observations given'
        assert G is None or G.shape == X.shape, 'wrong number (or dimensionality) of gradient-observations given'
        # print '(', len(self.X), self.G, G, ')'
        # assert not (len(self.X) > 0 and self.G is not None and G is None), 'gradients must either be always or never given'

        # this just makes sure that if we used the default gradient noise for
        # each dimension it gets lengthened to the proper size.
        if len(self.X) == 0 and len(self.gnoise) == 1: 
            self.gnoise = tile(self.gnoise, X.shape[1])

        # compute the correlations between our data points.
        r, m = \
            self._computeCorrelations(X) if (G is None) else \
            self._computeCorrelationsWithGradients(X)

        if len(self.X) == 0:
            self.X = copy(X)
            self.Y = copy(Y)
            self.G = copy(G) if (G is not None) else None
            self.R = r
            self.L = linalg.cholesky(self.R)
        else:
            self.X = r_[self.X, X]
            self.Y = r_[self.Y, Y]
            self.G = r_[self.G, G] if (G is not None) else None
            self.R = r_[c_[self.R, m], c_[m.T, r]]

            z = linalg.solve(self.L, m)
            d = linalg.cholesky(r - dot(z.T, z))
            self.L = r_[c_[self.L, zeros(z.shape)], c_[z.T, d]]
        # print '\nself.G =', G, ', for which selfG is None is', (self.G is None)
            
            
    def getYfromX(self, qx):
        """
        get the (first) Y value for a given training datum X.  return None if x 
        is not found.
        """
        for x, y in zip(self.X, self.Y):
            if all(qx==x):
                return y
        return None
        
    def done(self, x):
        """
        indication that the GP has been terminated and that a final point has
        been selected (mostly relevant for logging)
        """
        self.selected = x
        self.endtime = time()

        
class PrefGaussianProcess(GaussianProcess):
    """
    Like a regular Gaussian Process, but trained on preference data.  Note
    that you cannot (currently) add non-preference data.  This is because I
    haven't gotten around to it, not because it's impossible.
    """
    def __init__(self, kernel, prefs=None, **kwargs):
        super(PrefGaussianProcess, self).__init__(kernel, **kwargs)
        
        self.preferences = []
        self.C = None
        
        if prefs is not None:
            self.addPreferences(prefs)
        

    def addPreferences(self, prefs, useC=True, showPrefLikelihood=False):
        """
        Add a set of preferences to the GP and update.

        @param  prefs:  sequence of preference triples (xv, xu, d) where xv
                        is a datum preferred to xu and d is the degree of
                        preference (0 = 'standard', 1 = 'greatly preferred')
        """
        def S(x, prefinds, L, useC=False):
            """
            the MAP functional to be minimized
            """
            # print '***** x =',
            # for xx in x:
            #     print '%.3f'%xx,
            if useC:
                lpath = ctypes.util.find_library('ego')
                lib = cdll[lpath]
                lib.logCDFs.restype = c_double
                lib.logCDFs.argtypes = [c_int, POINTER(c_int), POINTER(c_double)]

                pf = array(prefinds[:], dtype=c_int).reshape(-1)
                cx = array(x.copy(), dtype=c_double)
                result = lib.logCDFs(c_int(len(pf)), pf.ctypes.data_as(POINTER(c_int)), cx.ctypes.data_as(POINTER(c_double)))
                logCDFs = result
            else:
                logCDFs = 0.
                sigma = 1
                epsilon = 1e-10
                Z = sqrt(2) * sigma
                for v, u, d in prefinds:
                    logCDFs += (d+1) * log(CDF((x[v]-x[u])/Z)+epsilon)

            Lx = linalg.solve(L, x)
            val = -logCDFs + dot(Lx, Lx)/2
            if not isfinite(val):
                print 'non-finite val!'
                pdb.set_trace()
            # print '\n***** val =', val
            return val

        # add new preferences
        self.preferences.extend(prefs)

        x2ind = {}
        ind = 0

        prefinds = []
        vs = set()
        for v, u, d in self.preferences:
            v = tuple(v)
            vs.add(v)
            u = tuple(u)
            if v not in x2ind:
                x2ind[v] = ind
                ind += 1
            if u not in x2ind:
                x2ind[u] = ind
                ind += 1
            prefinds.append((x2ind[v], x2ind[u], d))

        newX = array([x for x, _ in sorted(x2ind.items(), key=lambda x:x[1])])

        # use existing Ys as starting point for optimizer
        lastY = {}
        for x, y in zip(self.X, self.Y):
            lastY[tuple(x)] = y

        if len(self.Y) > 0:
            ymax = max(self.Y)
            ymin = min(self.Y)
        else:
            ymax = .5
            ymin = -.5
            
        start = []
        for x in newX:
            if tuple(x) in lastY:
                start.append(lastY[tuple(x)])
            else:
                if tuple(x) in vs:
                    start.append(ymax)
                else:
                    start.append(ymin)

        # update X, R
        self.X = zeros((0,0))
        r, m = \
            self._computeCorrelations(newX)
        self.X = newX
        self.R = r
        self.L = linalg.cholesky(self.R)

        # optimize S to find a good Y
        # self.Y = fmin_bfgs(S, start, args=(prefinds, self.L), epsilon=0.1, maxiter=30, disp=0)
        self.Y = fmin_bfgs(S, start, args=(prefinds, self.L), disp=0)
        
        # print '[addPreferences] checking pref pairs'
        for r, c, _ in self.preferences:
            r = tuple(r)
            c = tuple(c)
            # print '%d  %d' % (x2ind[r], x2ind[c]),
            if self.Y[x2ind[r]] <= self.Y[x2ind[c]]:
                # print '   FAILED!  %.2f ! > %.2f'  % (self.Y[x2ind[r]], self.Y[x2ind[c]])
                # print '    can we fix it?'
                # if there is nothing preferred to this item, bump it up
                for r1, c1, _ in self.preferences:
                    if all(c1==r):
                        break
                else:
                    self.Y[x2ind[r]] = self.Y[x2ind[c]] + .1
                    # print '    changed Y to %.2f' % self.Y[x2ind[r]]

        # now we can learn the C matrix
        self.C = eye(len(self.X), dtype=float) * 5
        # self.C = zeros((len(self.X), len(self.X)))
        for i in xrange(len(self.X)):
            for j in xrange(len(self.X)):
                for r, c, _ in self.preferences:
                    # print '******', r,c
                    alpha = 0
                    if all(r==self.X[i]) and all(c==self.X[j]):
                        alpha = -1
                    elif all(r==self.X[j]) and all(c==self.X[i]):
                        alpha = -1
                    elif all(r==self.X[i]) and i==j:
                        alpha = 1
                    elif all(c==self.X[i]) and i==j:
                        alpha = 1
                    if alpha != 0:
                        # print 'have an entry for %d, %d!' % (i,j)
                        d = (self.mu(r)-self.mu(c)) / (sqrt(2)*sqrt(self.noise))
                        # print '\td=',d
                        cdf = CDF(d)
                        pdf = PDF(d)
                        if cdf < 1e-10:
                            cdf = 1e-10
                        if pdf < 1e-10:
                            pdf = 1e-10
                        self.C[i,j] += alpha / (2*self.noise) * (pdf**2/cdf**2 + d * pdf/cdf)
        try:
            self.L = linalg.cholesky(self.R+linalg.inv(self.C))
        except LinAlgError:
            print '[addPreferences] GP.C matrix is ill-conditioned, adding regularizer delta = 1'
            for i in xrange(10):
                self.C += eye(len(self.X))
                try:
                    self.L = linalg.cholesky(self.R+linalg.inv(self.C))
                except LinAlgError:
                    print '[addPreferences] GP.C matrix is ill-conditioned, adding regularizer delta = %d' % (i+2)
                else:
                    break
                    
        
        
    def addObservationPoint(self, X):
        """
        Add a point at which we will observe, but for which we don't have the
        observation yet.  (Used by the gallery selection algorithms.)
        """
        X = array(X, copy=False, dtype=float, ndmin=2)
        
        if self.augR is None:
            self.augR = self.R.copy()
            self.augX = self.X.copy()
        
        r, m = self._computeAugCorrelations(X)
        self.augR = r_[c_[self.augR, m], c_[m.T, r]]
        
        invC = zeros_like(self.augR)
        invC[:self.C.shape[0], :self.C.shape[0]] = linalg.inv(self.C)
        self.augL = linalg.cholesky(self.augR+invC)
        self.augX = r_[self.augX, X]
        
            
    
    def addData(self, X, Y, G=None):
        """
        I have an idea about how to do this... (see notebook).
        """
        raise NotImplementedError("can't (yet) add explicit ratings to preference GP")
        
        
