#!/usr/bin/env python
# encoding: utf-8
"""
unittest_RF.py

Created by Eric on 2010-04-30.
Copyright (c) 2010 Eric Brochu. All rights reserved.
"""
from __future__ import division

import unittest
from copy import deepcopy

from matplotlib.pylab import *
from scipy import optimize
from numpy import *

from ego.randomforest import *
from ego.gaussianprocess import GaussianProcess, PrefGaussianProcess
from gaussianprocess.kernel import GaussianKernel_ard, GaussianKernel_iso, MaternKernel3, SVGaussianKernel_iso, SVGaussianKernel_ard
from gaussianprocess.prior import RBFNMeanPrior
from ego.acquisition import *
from acquisition.prefutil import query2prefs
from utils.latinhypercube import lhcSample
from gaussianprocess.trainhyper import marginalLikelihood, nlml, dnlml
from utils.testfunctions import Poly4, Poly6, Schubert1, GoldsteinPrice, Shekel5, Camelback, Branin, Hartman3, Hartman6


class TestRandomForest(unittest.TestCase):
        
    def testOneTree(self):
        
        forest = RandomForest(ntrees=1, m=4, ndata=2, pRetrain=.2)
        self.failUnlessEqual(forest.ntrees, 1)
        self.failUnlessEqual(forest.m, 4)
        self.failUnlessEqual(forest.ndata, 2)
        self.failUnlessEqual(forest.pRetrain, .2)
        
        tf = Branin()
        X = lhcSample(tf.bounds, 20, seed=0)
        Y = [tf.f(x) for x in X]
        
        forest.addData(X, Y)
        self.failUnlessEqual(len(forest.forest), 1)
        # checkTree(forest.forest[0])

        # maximizeEI(forest, tf.bounds)
        # print forest.forest[0]
        
        if False:
            figure(1, figsize=(5,10))
            c0 = [(i/100.)*(tf.bounds[0][1]-tf.bounds[0][0])+tf.bounds[0][0] for i in xrange(101)]
            c1 = [(i/100.)*(tf.bounds[1][1]-tf.bounds[1][0])+tf.bounds[1][0] for i in xrange(101)]

            ax = subplot(121)
            mu = array([[forest.mu(array([i, j])) for i in c0] for j in c1])
            cs = ax.contourf(c0, c1, mu, 50)
            colorbar(cs)
            ax.plot([x[0] for x in X], [x[1] for x in X], 'ro', alpha=.2)
            ax.set_xbound(tf.bounds[0][0], tf.bounds[0][1])
            ax.set_ybound(tf.bounds[1][0], tf.bounds[1][1])
            ax.set_title(r'$\mu$')
            
            ax = subplot(122)
            mu = array([[forest.sigma2(array([i, j])) for i in c0] for j in c1])
            cs = ax.contourf(c0, c1, mu, 50)
            colorbar(cs)
            ax.plot([x[0] for x in X], [x[1] for x in X], 'ro', alpha=.2)
            ax.set_xbound(tf.bounds[0][0], tf.bounds[0][1])
            ax.set_ybound(tf.bounds[1][0], tf.bounds[1][1])
            ax.set_title(r'$\sigma^2$')

            show()
            
    
    def testForestPI(self):
        
        RF = RandomForest(ntrees=2)
        tf = Branin()
        X = lhcSample(tf.bounds, 20, seed=0)
        Y = [tf.f(x) for x in X]
        RF.addData(X, Y)
        mu, sigma = RF.posterior(ones(len(tf.bounds))*.4)
        print '[python] = 0.4 x 2, mu =', mu, '  sigma =', sigma
        
        pif1 = PI(RF)
        dopt1, doptx1 = direct(pif1.negf, tf.bounds, maxiter=10)
        copt1, coptx1 = cdirect(pif1.negf, tf.bounds, maxiter=10)
        mopt1, moptx1 = maximizePI(RF, tf.bounds, maxiter=10)

        self.failUnlessAlmostEqual(dopt1, copt1, 4)
        self.failUnlessAlmostEqual(-dopt1, mopt1, 4)
        self.failUnlessAlmostEqual(-copt1, mopt1, 4)

        self.failUnless(sum(abs(doptx1-coptx1)) < .01)
        self.failUnless(sum(abs(moptx1-coptx1)) < .01)
        self.failUnless(sum(abs(moptx1-doptx1)) < .01)

        pif2 = PI(RF, xi=0.5)
        dopt2, doptx2 = direct(pif2.negf, tf.bounds, maxiter=10)
        copt2, coptx2 = cdirect(pif2.negf, tf.bounds, maxiter=10)
        mopt2, moptx2 = maximizePI(RF, tf.bounds, xi=0.5, maxiter=10)

        self.failUnlessAlmostEqual(dopt2, copt2, 4)
        self.failUnlessAlmostEqual(-dopt2, mopt2, 4)
        self.failUnlessAlmostEqual(-copt2, mopt2, 4)

        self.failUnless(sum(abs(doptx2-coptx2)) < .01)
        self.failUnless(sum(abs(moptx2-coptx2)) < .01)
        self.failUnless(sum(abs(moptx2-doptx2)) < .01)
        
        self.failIfAlmostEqual(dopt1, dopt2, 4)
        self.failIfAlmostEqual(copt1, copt2, 4)
        self.failIfAlmostEqual(mopt1, mopt2, 4)


    def testForestUCB(self):
        
        RF = RandomForest(ntrees=2)
        tf = Branin()
        X = lhcSample(tf.bounds, 20, seed=0)
        Y = [tf.f(x) for x in X]
        RF.addData(X, Y)
        
        ucbf = UCB(RF, len(tf.bounds))
        dopt, doptx = direct(ucbf.negf, tf.bounds, maxiter=10)
        copt, coptx = cdirect(ucbf.negf, tf.bounds, maxiter=10)
        mopt, moptx = maximizeUCB(RF, tf.bounds, maxiter=10)

        self.failUnlessAlmostEqual(dopt, copt, 4)
        self.failUnlessAlmostEqual(-dopt, mopt, 4)
        self.failUnlessAlmostEqual(-copt, mopt, 4)

        self.failUnless(sum(abs(doptx-coptx)) < .01)
        self.failUnless(sum(abs(moptx-coptx)) < .01)
        self.failUnless(sum(abs(moptx-doptx)) < .01)
        
        
    def _testTreeAccuracy(self):
        
        RF1 = RandomForest(ntrees=1, m=4, ndata=2, pRetrain=.2)
        RF10 = RandomForest(ntrees=10, m=4, ndata=2, pRetrain=.2)
        RF100 = RandomForest(ntrees=10, m=4, ndata=2, pRetrain=.2)
        
        tf = Branin()
        X = lhcSample(tf.bounds, 20, seed=0)
        Y = [tf.f(x) for x in X]
        
        RF1.addData(X, Y)
        RF10.addData(X, Y)
        RF100.addData(X, Y)
        
        rmse1 = 0.0
        rmse10 = 0.0
        rmse100 = 0.0
        
        nsamp = 1000
        
        for testx in lhcSample(tf.bounds, nsamp, seed=1):
            testy = tf.f(testx)
            rmse1 += (RF1.mu(testx)-testy)**2
            rmse10 += (RF10.mu(testx)-testy)**2
            rmse100 += (RF100.mu(testx)-testy)**2
        
        # this isn't consistent, since random forests are, you know, random
        print 'RMSE 1 = %.4f'%(rmse1/nsamp)
        print 'RMSE 10 = %.4f'%(rmse10/nsamp)
        print 'RMSE 100 = %.4f'%(rmse100/nsamp)
        
        self.failUnless(rmse1 > rmse100)
    
if __name__ == '__main__':

    tests = [TestRandomForest]

    for test in tests:
        suite = unittest.TestLoader().loadTestsFromTestCase(test)
        unittest.TextTestRunner(verbosity=2).run(suite)
