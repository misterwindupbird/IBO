#!/usr/bin/env python
# encoding: utf-8
"""
demo.py

Just a little script to demonstrate how to call the interactive Bayesian optimization (IBO/EGO) code.

Created by Eric on 2010-03-20.
Copyright (c) 2010 Eric Brochu. All rights reserved.
"""

from copy import deepcopy

from numpy import array, arange
from matplotlib.pylab import *

from ego.gaussianprocess import GaussianProcess, PrefGaussianProcess
from ego.gaussianprocess.kernel import GaussianKernel_ard
from ego.acquisition import EI, UCB, maximizeEI, maximizeUCB
from ego.acquisition.prefutil import query2prefs
from ego.acquisition.gallery import fastUCBGallery
from ego.utils.testfunctions import Hartman6


def demoObservations():
    """
    Simple demo for a scenario where we have direct observations (ie ratings
    or responses) with noise.  The model has three parameters, but after
    initial training, we fix one to be 1.0 and optimize the other two.  At
    each step, we visualize the posterior mean, variance and expected
    improvement.  We then find the point of maximum expected improvement and
    ask the user for the scalar response value.  
    
    To see how the model adapts to inputs, try rating the first few values 
    higher or lower than predicted and see what happens to the visualizations.
    """

    # the kernel parameters control the impact of different values on the 
    # parameters.  we are defining a model with three parameters
    kernel = GaussianKernel_ard(array([.5, .5, .3]))
    
    # we want to allow some noise in the observations -- the noise parameter
    # is the variance of the additive Gaussian noise   Y + N(0, noise)
    noise = 0.1
    
    # create the Gaussian Process using the kernel we've just defined
    GP = GaussianProcess(kernel, noise=noise)
    
    # add some data to the model.  the data must have the same dimensionality 
    # as the kernel
    X = [array([1, 1.5, 0.9]),
         array([.8, -.2, -0.1]),
         array([2, .8, -.2]),
         array([0, 0, .5])]
    Y = [1, .7, .6, -.1]
    
    print 'adding data to model'
    for x, y in zip(X, Y):
        print '\tx = %s, y = %.1f' % (x, y)
        
    GP.addData(X, Y)
    
    # the GP.posterior(x) function returns, for x, the posterior distribution
    # at x, characterized as a normal distribution with mean mu, variance 
    # sigma^2
    testX = [array([1, 1.45, 1.0]),
             array([-10, .5, -10])]
    
    for tx in testX:
        mu, sig2 = GP.posterior(tx)
        print 'the posterior of %s is a normal distribution N(%.3f, %.3f)' % (tx, mu, sig2)
        
    # now, let's find the best points to evaluate next.  we fix the first 
    # dimension to be 1 and for the others, we search the range [-2, 2]
    bound = [[1, 1], [-1.99, 1.98], [-1.99, 1.98]]
    
    figure(1, figsize=(5, 10))
    while True:
        _, optx = maximizeEI(GP, bound, xi=.1)

        # visualize the mean, variance and expected improvement functions on 
        # the free parameters
        x1 = arange(bound[1][0], bound[1][1], 0.1)
        x2 = arange(bound[2][0], bound[2][1], 0.1)
        X1, X2 = meshgrid(x1, x2)
        ei = zeros_like(X1)
        m = zeros_like(X1)
        v = zeros_like(X1)
        for i in xrange(X1.shape[0]):
            for j in xrange(X1.shape[1]):
                z = array([1.0, X1[i,j], X2[i,j]])
                ei[i,j] = -EI(GP).negf(z)
                m[i,j], v[i,j] = GP.posterior(z)
        
        clf()
        for i, (func, title) in enumerate(([m, 'prediction (posterior mean)'], [v, 'uncertainty (posterior variance)'], [ei, 'utility (expected improvement)'])):
            ax = subplot(3, 1, i+1)
            cs = ax.contourf(X1, X2, func, 20)
            ax.plot(optx[1], optx[2], 'wo')
            colorbar(cs)
            ax.set_title(title)
            ax.set_xlabel('x[1]')
            ax.set_ylabel('x[2]')
            ax.set_xticks([-2,0,2])
            ax.set_yticks([-2,0,2])


        m, v = GP.posterior(optx)
        try:
            response = input('\nmaximum expected improvement is at parameters x = [%.3f, %.3f, %.3f], where mean is %.3f, variance is %.3f.  \nwhat is the value there (non-numeric to quit)? ' % (optx[0], optx[1], optx[2], m, v))
        except:
            break
        GP.addData(optx, response)
        print 'updating model.'


def demoPrefGallery():
    """
    A simpler demo, showing how to use a preference gallery.  This demo
    is not interactive -- it uses the 6D Hartman test function to generate
    the preferences.
    """
    
    N = 3   # gallery size
    
    # use the Hartman6 test function, which has a kernel and bounds predefined
    tf = Hartman6()
    bounds = tf.bounds
    kernel = tf.createKernel(GaussianKernel_ard)
    
    # set up a Preference Gaussian Process, in which the observations are
    # preferences, rather than scalars
    GP = PrefGaussianProcess(kernel)
    
    # initial preferences -- since we have no informative prior on the space, 
    # the gallery will be a set of points that maximize variance
    gallery = fastUCBGallery(GP, bounds, N)
    
    # this is a utility function for automtically testing the preferences -- 
    # given a test functions and some sample points, it will return a list of 
    # preferences
    prefs = query2prefs(gallery, tf.f)
    
    # preferences have the form [r, c, degree], where r is preferred to c.  
    # degree is degree of preference.  Just leave degree at 0 for now.
    for r, c, _ in prefs:
        print '%s preferred to %s' % (r, c)
        
    # add preferences to the model
    GP.addPreferences(prefs)
    
    # get another gallery, but with the first three dimensions fixed to .5
    nbounds = deepcopy(bounds)
    nbounds[:3] = [[.5,.5]]*3
    
    gallery = fastUCBGallery(GP, nbounds, N)
    prefs = query2prefs(gallery, tf.f)
    for r, c, _ in prefs:
        print '%s preferred to %s' % (r, c)
    
    # get another gallery, but with the *last* three dimensions fixed to .5
    nbounds = deepcopy(bounds)
    nbounds[3:] = [[.5,.5]]*3

    gallery = fastUCBGallery(GP, nbounds, N)
    prefs = query2prefs(gallery, tf.f)
    for r, c, _ in prefs:
        print '%s preferred to %s' % (r, c)
    
    # preferences don't have to come from the gallery
    r = array([0, 0, .5, 0, 1, .25])
    c = array([1, 1, .75, 1, 0, .5])
    pref = (r, c, 0)
    GP.addPreferences([pref])
    

if __name__ == '__main__':
    demoObservations()
    # demoPrefGallery()

