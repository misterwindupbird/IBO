#!python
from __future__ import division

from time import asctime

from matplotlib.pylab import *
from numpy.linalg import cholesky, solve, inv
from numpy.linalg.linalg import LinAlgError

import pdb

def logBadParams(kernel):
    """
    Log bad params.
    """
    fname = 'badhyper.log'
    line = asctime()
    line += '\t%s'%kernel.__class__
    for h in kernel.hyperparams:
        line += '\t%f'%h
    f = open(fname, 'a')
    f.write(line+'\n')
    f.close()


def marginalLikelihood(kernel, X, Y, nhyper, computeGradient=True, useCholesky=True, noise=1e-3):
    """
    get the negative log marginal likelihood and its partial derivatives wrt
    each hyperparameter
    """
    NX = len(X)
    assert NX == len(Y)
    # compute covariance matrix
    K = kernel.covMatrix(X) + eye(NX)*noise
    
    if useCholesky:
        # avoid inversion by using Cholesky decomp.
        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L, Y))
        except LinAlgError, e:
            print '\n ================ error in matrix'
            print '\thyper =', kernel.hyperparams
            print '===================================='
            logBadParams(kernel)
            pdb.set_trace()
        nlml = 0.5 * dot(Y, alpha) + sum(log(diag(L))) + 0.5 * NX * log(2.0*pi)
        if computeGradient:
            W = solve(L.T, solve(L, eye(NX))) - outer(alpha, alpha)
            dnlml = array([sum(W*kernel.derivative(X, i)) / 2.0 for i in xrange(nhyper)])
            # print '  loglik =', nlml, '   d loglik =', dnlml
            return nlml, dnlml
        else:
            return nlml
    
    else:
        try:
            invK = inv(K)
            alpha = dot(invK, Y)
        except LinAlgError, e:
            print '\n ================ error in matrix'
            print '\thyper =', kernel.hyperparams
            print '===================================='
            logBadParams(kernel)
            raise

        # negative log marginal likelihood.  eqn 5.8 of Rasmussen & Williams
        lml = -0.5 * dot(Y, alpha) - .5 * log(det(K)) - 0.5 * NX * log(2.0*pi)
        if computeGradient:
            W = invK - outer(alpha, alpha)
            dnlml = array([sum(W*kernel.derivative(X, i)) / 2.0 for i in xrange(nhyper)])
            return -lml, dnlml
        else:
            return -lml
        
    
    
def nlml(loghyper, kernel, X, Y, *args):
    """
    Negative log marginal likelihood computation.  Note that when we optimize,
    we want to use the log hyperparameters, which is what we expect to get 
    passed in, but the kernels themselves do not take log hyperparameters.
    """
    # print 'hyper =', exp(loghyper),
    # print 'loghyper =', loghyper
    
    k = kernel(exp(loghyper))
    try:
        ml = marginalLikelihood(k, X, Y, len(loghyper), computeGradient=False)
    except LinAlgError, e:
        print e
        ml = 100
        print 'returning nlml = 100'
    return ml
    

def nlmlMulti(loghyper, kernel, X, Y, *args):
    """
    Negative log marginal likelihood computation over multiple instances.
    """
    k = kernel(exp(loghyper))
    
    ml = 0.0
    for x, y in zip(X, Y):
        ml += marginalLikelihood(k, x, y, len(loghyper))[0]
    return ml
    

def dnlml(loghyper, kernel, X, Y):
    """
    Derivatives of the negative log marginal likelihood.  Again, we are
    expecting log hyperparameters, which we will convert.
    """
    k = kernel(exp(loghyper))
    return marginalLikelihood(k, X, Y, len(loghyper), computeGradient=True)[1]
    
    