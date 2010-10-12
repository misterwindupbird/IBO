#!/usr/bin/env python
# encoding: utf-8
"""
gallery.py

Created by Eric on 2010-03-21.
Copyright (c) 2010 Eric Brochu. All rights reserved.
"""

from copy import deepcopy

from numpy import inf, array, clip
from numpy.linalg import norm
from scipy.optimize import fmin_bfgs

from ego.gaussianprocess import GaussianProcess
from ego.utils.latinhypercube import lhcSample
from ego.acquisition import UCB, EI, maximizeEI

DEBUG = False

def fastUCBGallery(GP, bounds, N, useBest=True, samples=300, useCDIRECT=True):
    """
    Use UCB to generate a gallery of N instances using Monte Carlo to 
    approximate the optimization of the utility function.
    """
    gallery = []

    if len(GP.X) > 0:
        if useBest:
            # find best sample already seen, that lies within the bounds
            bestY = -inf
            bestX = None
            for x, y in zip(GP.X, GP.Y):
                if y > bestY:
                    for v, b in zip(x, bounds):
                        if v < b[0] or v > b[1]:
                            break
                    else:
                        bestY = y
                        bestX = x
            if bestX is not None:
                gallery.append(bestX)
    
        # create a "fake" GP from the GP that was passed in (can't just copy 
        # b/c original could have been PrefGP)
        hallucGP = GaussianProcess(deepcopy(GP.kernel), deepcopy(GP.X), deepcopy(GP.Y), prior=GP.prior)
    elif GP.prior is None:            
        # if we have no data and no prior, start in the center
        x = array([(b[0]+b[1])/2. for b in bounds])
        gallery.append(x)
        hallucGP = GaussianProcess(deepcopy(GP.kernel), [x], [0.0], prior=GP.prior)
    else:
        # optimize from prior
        if DEBUG: print 'GET DATA FROM PRIOR'
        bestmu = -inf
        bestX = None
        for m in GP.prior.means:
            argmin = fmin_bfgs(GP.negmu, m, disp=False)
            if DEBUG: print argmin,
            for i in xrange(len(argmin)):
                argmin[i] = clip(argmin[i], bounds[i][0], bounds[i][1])
            # if DEBUG: print 'converted to', argmin
            if GP.mu(argmin) > bestmu:
                bestX = argmin
                bestmu = GP.mu(argmin)
                if DEBUG: print '***** bestmu =', bestmu
                if DEBUG: print '***** bestX =', bestX
        gallery.append(bestX)
        hallucGP = GaussianProcess(deepcopy(GP.kernel), bestX, bestmu, prior=GP.prior)
        
        
    while len(gallery) < N:
        if DEBUG: print '\n\n\thave %d data for gallery' % len(gallery)
        bestUCB = -inf
        bestX = None
        # ut = UCB(hallucGP, len(bounds), N)
        ut = EI(hallucGP, xi=.4)
        
        if DEBUG: print '\tget with max EI'
        opt, optx = maximizeEI(hallucGP, bounds, xi=.3, useCDIRECT=useCDIRECT)
        if len(gallery)==0 or min(norm(optx-gx) for gx in gallery) > .5:
            if DEBUG: print '\tgot one'
            bestUCB = opt
            bestX = optx
        else:
            if DEBUG: print '\ttoo close to existing'
        
        # try some random samples
        if DEBUG: print '\ttry random samples'
        for x in lhcSample(bounds, samples):
            u = -ut.negf(x)
            if u > bestUCB and min(norm(x-gx) for gx in gallery) > .5:
                '\they, this one is even better!'
                bestUCB = u
                bestX = x
        
        # now try the prior means
        if hallucGP.prior is not None:
            if DEBUG: print '\ttry prior means (bestUCB = %f)'%bestUCB
            for x in hallucGP.prior.means:
                x = array([clip(x[i], bounds[i][0], bounds[i][1]) for i in xrange(len(x))])
                x = x * hallucGP.prior.width + hallucGP.prior.lowerb
                u = -ut.negf(x)
                # if DEBUG: print 'u = %f', u
                if u > bestUCB:
                    if len(gallery)==0 or min(norm(x-gx) for gx in gallery) > .5:
                        if DEBUG: print '\tthis one is even better!  prior mean %s has u = %f' % (x, u)
                        bestUCB = u
                        bestX = x
                    
        gallery.append(bestX)
        
        hallucGP.addData(bestX, hallucGP.mu(bestX))
        
    return gallery