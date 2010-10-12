#!python
"""
Unittests for the EGO module go here.
"""
from __future__ import division

import unittest
from math import sin
from copy import deepcopy

from matplotlib.pylab import figure, plot, subplot, show, plt, cm, annotate, clf
from scipy import optimize
from numpy import arange, array, any, all, sin, mean, matrix, sum, log, copy

from ego.gaussianprocess import GaussianProcess, PrefGaussianProcess
from ego.gaussianprocess.kernel import GaussianKernel_ard, GaussianKernel_iso, MaternKernel3
from ego.gaussianprocess.prior import RBFNMeanPrior
from ego.acquisition.prefutil import query2prefs
from ego.acquisition.gallery import fastUCBGallery
from ego.utils.latinhypercube import lhcSample
#from gaussianprocess.trainhyper import marginalLikelihood, nlml, dnlml
from ego.utils.testfunctions import Poly4, Poly6, Schubert1, GoldsteinPrice, Shekel5, Camelback, Branin, Hartman3, Hartman6
from ego.utils.performance import Gdelta
from ego.utils.optimize import direct, cdirect
from ego.acquisition import maximizeEI, maximizePI, maximizeUCB, EI, PI, UCB


class TestLatinhypercube(unittest.TestCase):
    
    def testLatinHypercubeSampling(self):
        
        # test 1D case
        samples = lhcSample([[0., 1.]], 100, seed=20)
        self.assertEqual(len(samples), 100)
        self.assert_(min(samples) >= 0.0)        
        self.assert_(max(samples) <= 1.0)
        
        isamp = [int(s*100) for s in samples]
        self.assertNotEqual(isamp, range(100))
        isamp.sort()
        self.assertEqual(isamp, range(100))
        
        # test 100D case
        samples = lhcSample([(i*10., (i+1.)*10.) for i in xrange(100)], 10, seed=21)
        for s in samples:
            self.assertEqual(len(s), 100)
            for i, x in enumerate(s):
                self.assert_(x > i*10.)
                self.assert_(x < (i+1)*10.)
    
    
    
class TestDIRECT(unittest.TestCase):
    
    def testShekel(self):

        S = Shekel5(maximize=False)
        opt, optx = direct(S.f, S.bounds, maxiter=20, debug=False)
        self.assertAlmostEqual(opt, S.minimum, 3)
        for x, a in zip(optx, S.argmin):
            self.assertAlmostEqual(x, a, 3)
            
    
    def testArgs(self):
        
        # test passing args into DIRECT
        def foo(x, a1, a2):
            return -sum(sin(array(x)*a1)+array(x)*a2)
        
        bounds = [[0., 5.]] * 3
        args1 = [3.0, 0.0]
        opt, optx = direct(foo, bounds, args=args1, maxiter=20)        
        # print 'args = %s, opt=%s, optx=%s' % (args1, opt, optx)
        self.assertAlmostEqual(opt, -3.0, 1)
        self.assertAlmostEqual(optx[0], 2.6234, 1)

        args2 = [-2.0, 2.0]
        opt, optx = direct(foo, bounds, args=args2, maxiter=10)
        # print 'args = %s, opt=%s, optx=%s' % (args2, opt, optx)
        # self.failUnlessAlmostEqual(opt, -31.0540, 4)
        self.assert_(abs(optx[0]-5) < .5)
        
        
class TestCDIRECT(unittest.TestCase):
    
    def testShekel(self):
        
        S = Shekel5(maximize=False)        
        opt, optx = cdirect(S.f, S.bounds, maxiter=20)

        self.assertAlmostEqual(opt, S.minimum, 3)
        for x, a in zip(optx, S.argmin):
            self.assertAlmostEqual(x, a, 3)
            
    
    def testArgs(self):
        
        # test passing args into DIRECT
        def foo(x, a1, a2):
            return -sum(sin(array(x)*a1)+array(x)*a2)
        
        bounds = [[0., 5.]] * 3
        args1 = [3.0, 0.0]
        opt, optx = cdirect(foo, bounds, args=args1, maxiter=10)        
        # print 'args = %s, opt=%s, optx=%s' % (args1, opt, optx)
        self.assertAlmostEqual(opt, -3.00, 2)
        self.assertAlmostEqual(optx[0], 2.618, 2)

        args2 = [-2.0, 2.0]
        opt, optx = cdirect(foo, bounds, args=args2, maxiter=10)
        # print 'args = %s, opt=%s, optx=%s' % (args2, opt, optx)
        self.assert_(abs(-opt-31) < 1.)
        self.assert_(abs(optx[0]-5) < .5)
        
        
        
class TestMaximizePI(unittest.TestCase):
    
    def test1DcPI(self):
        
        f = lambda x: float(sin(x*5.))
        X = lhcSample([[0., 1.]], 5, seed=22)
        Y = [f(x) for x in X]

        kernel = GaussianKernel_ard(array([1.0]))
        GP = GaussianProcess(kernel)
        GP.addData(X, Y)
        
        # should use optimizeGP.cpp
        pif = PI(GP)
        dopt, doptx = direct(pif.negf, [[0., 1.]], maxiter=10)
        copt, coptx = cdirect(pif.negf, [[0., 1.]], maxiter=10)
        mopt, moptx = maximizePI(GP, [[0., 1.]], maxiter=10)
        
        self.failUnlessAlmostEqual(dopt, copt, 4)
        self.failUnlessAlmostEqual(-dopt, mopt, 4)
        self.failUnlessAlmostEqual(-copt, mopt, 4)
    
        self.failUnless(sum(abs(doptx-coptx)) < .01)
        self.failUnless(sum(abs(moptx-coptx)) < .01)
        self.failUnless(sum(abs(moptx-doptx)) < .01)
    
    
    def testXi(self):
        
        S5 = Shekel5()
        
        GP1 = GaussianProcess(GaussianKernel_iso([.2]))
        # self.failUnlessEqual(GP1.xi, 0.0)
        X = lhcSample(S5.bounds, 10, seed=0)
        Y = [S5.f(x) for x in X]
        GP1.addData(X, Y)

        pif1 = PI(GP1, xi=0.0)
        dopt1, _ = direct(pif1.negf, S5.bounds, maxiter=10)
        copt1, _ = cdirect(pif1.negf, S5.bounds, maxiter=10)
        mopt1, _ = maximizePI(GP1, S5.bounds, xi=0.0, maxiter=10)

        self.failUnlessAlmostEqual(dopt1, copt1, 4)
        self.failUnlessAlmostEqual(-dopt1, mopt1, 4)
        self.failUnlessAlmostEqual(-copt1, mopt1, 4)

        GP2 = GaussianProcess(GaussianKernel_iso([.3]), X, Y)
        pif2 = PI(GP2, xi=0.01)    
        dopt2, _ = direct(pif2.negf, S5.bounds, maxiter=10)
        copt2, _ = cdirect(pif2.negf, S5.bounds, maxiter=10)
        mopt2, _ = maximizePI(GP2, S5.bounds, xi=0.01, maxiter=10)
        self.failUnlessAlmostEqual(dopt2, copt2, 4)
        self.failUnlessAlmostEqual(-dopt2, mopt2, 4)
        self.failUnlessAlmostEqual(-copt2, mopt2, 4)

        self.failIfAlmostEqual(dopt1, dopt2, 4)
        self.failIfAlmostEqual(copt1, copt2, 4)
        self.failIfAlmostEqual(mopt1, mopt2, 4)

        GP3 = GaussianProcess(GaussianKernel_iso([.3]), X, Y)
        pif3 = PI(GP3, xi=0.1)    
        dopt3, _ = direct(pif3.negf, S5.bounds, maxiter=10)
        copt3, _ = cdirect(pif3.negf, S5.bounds, maxiter=10)
        mopt3, _ = maximizePI(GP3, S5.bounds, xi=0.1, maxiter=10)
        self.failUnlessAlmostEqual(dopt3, copt3, 4)
        self.failUnlessAlmostEqual(-dopt3, mopt3, 4)
        self.failUnlessAlmostEqual(-copt3, mopt3, 4)

        self.failIfAlmostEqual(dopt1, dopt3, 4)
        self.failIfAlmostEqual(copt1, copt3, 4)
        self.failIfAlmostEqual(mopt1, mopt3, 4)
        self.failIfAlmostEqual(dopt2, dopt3, 4)
        self.failIfAlmostEqual(copt2, copt3, 4)
        self.failIfAlmostEqual(mopt2, mopt3, 4)
    
    
class TestMaximizeUCB(unittest.TestCase):
    
    def test1DcUCB(self):
        
        f = lambda x: float(sin(x*5.))
        X = lhcSample([[0., 1.]], 5, seed=22)
        Y = [f(x) for x in X]

        kernel = GaussianKernel_ard(array([1.0]))
        GP = GaussianProcess(kernel)
        GP.addData(X, Y)
        
        # should use optimizeGP.cpp
        ucbf = UCB(GP, 1)
        dopt, doptx = direct(ucbf.negf, [[0., 1.]], maxiter=10)
        copt, coptx = cdirect(ucbf.negf, [[0., 1.]], maxiter=10)
        mopt, moptx = maximizeUCB(GP, [[0., 1.]], maxiter=10)
        
        self.failUnlessAlmostEqual(dopt, copt, 4)
        self.failUnlessAlmostEqual(-dopt, mopt, 4)
        self.failUnlessAlmostEqual(-copt, mopt, 4)
    
        self.failUnless(sum(abs(doptx-coptx)) < .01)
        self.failUnless(sum(abs(moptx-coptx)) < .01)
        self.failUnless(sum(abs(moptx-doptx)) < .01)
    
    
    def testXi(self):
        
        S5 = Shekel5()
        
        GP1 = GaussianProcess(GaussianKernel_iso([.2]))
        # self.failUnlessEqual(GP1.xi, 0.0)
        X = lhcSample(S5.bounds, 10, seed=0)
        Y = [S5.f(x) for x in X]
        GP1.addData(X, Y)

        ucbf1 = UCB(GP1, len(S5.bounds), scale=0.5)
        dopt1, _ = direct(ucbf1.negf, S5.bounds, maxiter=10)
        copt1, _ = cdirect(ucbf1.negf, S5.bounds, maxiter=10)
        mopt1, _ = maximizeUCB(GP1, S5.bounds, scale=0.5, maxiter=10)

        self.failUnlessAlmostEqual(dopt1, copt1, 4)
        self.failUnlessAlmostEqual(-dopt1, mopt1, 4)
        self.failUnlessAlmostEqual(-copt1, mopt1, 4)

        GP2 = GaussianProcess(GaussianKernel_iso([.3]), X, Y)
        ucbf2 = UCB(GP2, len(S5.bounds), scale=0.01)    
        dopt2, _ = direct(ucbf2.negf, S5.bounds, maxiter=10)
        copt2, _ = cdirect(ucbf2.negf, S5.bounds, maxiter=10)
        mopt2, _ = maximizeUCB(GP2, S5.bounds, scale=.01, maxiter=10)
        self.failUnlessAlmostEqual(dopt2, copt2, 4)
        self.failUnlessAlmostEqual(-dopt2, mopt2, 4)
        self.failUnlessAlmostEqual(-copt2, mopt2, 4)

        self.failIfAlmostEqual(dopt1, dopt2, 4)
        self.failIfAlmostEqual(copt1, copt2, 4)
        self.failIfAlmostEqual(mopt1, mopt2, 4)

        GP3 = GaussianProcess(GaussianKernel_iso([.3]), X, Y)
        ucbf3 = UCB(GP3, len(S5.bounds), scale=.9)    
        dopt3, _ = direct(ucbf3.negf, S5.bounds, maxiter=10)
        copt3, _ = cdirect(ucbf3.negf, S5.bounds, maxiter=10)
        mopt3, _ = maximizeUCB(GP3, S5.bounds, scale=0.9, maxiter=10)
        self.failUnlessAlmostEqual(dopt3, copt3, 4)
        self.failUnlessAlmostEqual(-dopt3, mopt3, 4)
        self.failUnlessAlmostEqual(-copt3, mopt3, 4)

        self.failIfAlmostEqual(dopt1, dopt3, 4)
        self.failIfAlmostEqual(copt1, copt3, 4)
        self.failIfAlmostEqual(mopt1, mopt3, 4)
        self.failIfAlmostEqual(dopt2, dopt3, 4)
        self.failIfAlmostEqual(copt2, copt3, 4)
        self.failIfAlmostEqual(mopt2, mopt3, 4)
        
        
class TestMaximizeEI(unittest.TestCase):
    
    def test1DcEI(self):
        
        f = lambda x: float(sin(x*5.))
        X = lhcSample([[0., 1.]], 5, seed=22)
        Y = [f(x) for x in X]

        kernel = GaussianKernel_ard(array([1.0]))
        GP = GaussianProcess(kernel)
        GP.addData(X, Y)
        
        # should use optimizeGP.cpp
        maxei = maximizeEI(GP, [[0., 1.]])
        
        if False:
            figure(1)
            plot(X, Y, 'ro')
            plot([x/100 for x in xrange(100)], [GP.ei(x/100) for x in xrange(100)])
            plot(maxei[1][0], maxei[0], 'ko')
            show()
        
    def test2DcEI(self):
        
        f = lambda x: sum(sin(x))
        bounds = [[0., 5.], [0., 5.]]
        X = lhcSample(bounds, 5, seed=23)
        Y = [f(x) for x in X]

        kernel = GaussianKernel_iso(array([1.0]))
        GP = GaussianProcess(kernel, X, Y)

        maxei = maximizeEI(GP, bounds)
        
        if False:
            figure(1)
            c0 = [(i/100.)*(bounds[0][1]-bounds[0][0])+bounds[0][0] for i in xrange(101)]
            c1 = [(i/100.)*(bounds[1][1]-bounds[1][0])+bounds[1][0] for i in xrange(101)]
            z = array([[GP.ei(array([i, j])) for i in c0] for j in c1])

            ax = plt.subplot(111)
            ax.contour(c0, c1, z, 10, alpha=0.5, cmap=cm.Blues_r)
            plot([x[0] for x in X], [x[1] for x in X], 'ro')
            for i in xrange(len(X)):
                annotate('%2f'%Y[i], X[i])
            plot(maxei[1][0], maxei[1][1], 'ko')
            show()

    def test2DpyEI(self):
        
        f = lambda x: sum(sin(x))
        bounds = [[0., 5.], [0., 5.]]
        X = lhcSample(bounds, 5, seed=24)
        Y = [f(x) for x in X]

        kernel = GaussianKernel_ard(array([1.0, 1.0]))
        GP = GaussianProcess(kernel, X, Y)

        maxei = maximizeEI(GP, bounds)
        
        if False:
            figure(1)
            c0 = [(i/50.)*(bounds[0][1]-bounds[0][0])+bounds[0][0] for i in xrange(51)]
            c1 = [(i/50.)*(bounds[1][1]-bounds[1][0])+bounds[1][0] for i in xrange(51)]
            z = array([[GP.ei(array([i, j])) for i in c0] for j in c1])

            ax = plt.subplot(111)
            cs = ax.contour(c0, c1, z, 10, alpha=0.5, cmap=cm.Blues_r)
            plot([x[0] for x in X], [x[1] for x in X], 'ro')
            for i in xrange(len(X)):
                annotate('%2f'%Y[i], X[i])
            plot(maxei[1][0], maxei[1][1], 'ko')
            show()
            
            
    # deactivated for being slow to run
    def _testKernelMaxEI(self):
        
        # test different methods of optimizing kernel
        S5 = Shekel5()
        
        hv = 0.1
        testkernels = [GaussianKernel_iso([hv]), 
                   GaussianKernel_ard([hv, hv, hv, hv]),
                   MaternKernel3([hv, 1.0])]
                   # MaternKernel5([hv, 1.0])]

        for kernel in testkernels:
            # print
            # print kernel.__class__
            
        
            # train GPs
            X = lhcSample(S5.bounds, 10, seed=0)
            Y = [S5.f(x) for x in X]
        
            GP = GaussianProcess(kernel, X, Y)
        
            eif = EI(GP)
            dopt, doptx = direct(eif.negf, S5.bounds, maxiter=10)
            copt, coptx = cdirect(eif.negf, S5.bounds, maxiter=10)
            mopt, moptx = maximizeEI(GP, S5.bounds, maxiter=10)
            # print dopt, doptx
            # print copt, coptx
            # print mopt, moptx
        
            self.failUnlessAlmostEqual(dopt, copt, 4)
            self.failUnlessAlmostEqual(-dopt, mopt, 4)
            self.failUnlessAlmostEqual(-copt, mopt, 4)
        
            self.failUnless(sum(abs(doptx-coptx)) < .01)
            self.failUnless(sum(abs(moptx-coptx)) < .01)
            self.failUnless(sum(abs(moptx-doptx)) < .01)
        
            # train GP w/prior
            pX = lhcSample(S5.bounds, 100, seed=101)
            pY = [S5.f(x) for x in pX]
            prior = RBFNMeanPrior()
            prior.train(pX, pY, bounds=S5.bounds, k=10, seed=102)
        
            GP = GaussianProcess(kernel, X, Y, prior=prior)        
        
            eif = EI(GP)
            pdopt, pdoptx = direct(eif.negf, S5.bounds, maxiter=10)
            pcopt, pcoptx = cdirect(eif.negf, S5.bounds, maxiter=10)
            pmopt, pmoptx = maximizeEI(GP, S5.bounds, maxiter=10)
        
            self.failIfAlmostEqual(pdopt, dopt, 3)
            self.failUnlessAlmostEqual(pdopt, pcopt, 4)
            self.failUnlessAlmostEqual(-pdopt, pmopt, 4)
            self.failUnlessAlmostEqual(-pcopt, pmopt, 4)
        
            self.failUnless(sum(abs(pdoptx-pcoptx)) < .01)
            self.failUnless(sum(abs(pmoptx-pcoptx)) < .01)
            self.failUnless(sum(abs(pmoptx-pdoptx)) < .01)
        
        
    def testXi(self):
        
        S5 = Shekel5()
        
        GP1 = GaussianProcess(GaussianKernel_iso([.2]))
        # self.failUnlessEqual(GP1.xi, 0.0)
        X = lhcSample(S5.bounds, 10, seed=0)
        Y = [S5.f(x) for x in X]
        GP1.addData(X, Y)

        eif1 = EI(GP1, xi=0.0)
        dopt1, _ = direct(eif1.negf, S5.bounds, maxiter=10)
        copt1, _ = cdirect(eif1.negf, S5.bounds, maxiter=10)
        mopt1, _ = maximizeEI(GP1, S5.bounds, xi=0.0, maxiter=10)

        self.failUnlessAlmostEqual(dopt1, copt1, 4)
        self.failUnlessAlmostEqual(-dopt1, mopt1, 4)
        self.failUnlessAlmostEqual(-copt1, mopt1, 4)

        GP2 = GaussianProcess(GaussianKernel_iso([.3]), X, Y)
        eif2 = EI(GP2, xi=0.01)    
        self.failUnlessEqual(eif2.xi, 0.01)    
        dopt2, _ = direct(eif2.negf, S5.bounds, maxiter=10)
        copt2, _ = cdirect(eif2.negf, S5.bounds, maxiter=10)
        mopt2, _ = maximizeEI(GP2, S5.bounds, xi=0.01, maxiter=10)
        self.failUnlessAlmostEqual(dopt2, copt2, 4)
        self.failUnlessAlmostEqual(-dopt2, mopt2, 4)
        self.failUnlessAlmostEqual(-copt2, mopt2, 4)

        self.failIfAlmostEqual(dopt1, dopt2, 4)
        self.failIfAlmostEqual(copt1, copt2, 4)
        self.failIfAlmostEqual(mopt1, mopt2, 4)

        GP3 = GaussianProcess(GaussianKernel_iso([.3]), X, Y)
        eif3 = EI(GP3, xi=0.1)    
        dopt3, _ = direct(eif3.negf, S5.bounds, maxiter=10)
        copt3, _ = cdirect(eif3.negf, S5.bounds, maxiter=10)
        mopt3, _ = maximizeEI(GP3, S5.bounds, xi=0.1, maxiter=10)
        self.failUnlessAlmostEqual(dopt3, copt3, 4)
        self.failUnlessAlmostEqual(-dopt3, mopt3, 4)
        self.failUnlessAlmostEqual(-copt3, mopt3, 4)

        self.failIfAlmostEqual(dopt1, dopt3, 4)
        self.failIfAlmostEqual(copt1, copt3, 4)
        self.failIfAlmostEqual(mopt1, mopt3, 4)
        self.failIfAlmostEqual(dopt2, dopt3, 4)
        self.failIfAlmostEqual(copt2, copt3, 4)
        self.failIfAlmostEqual(mopt2, mopt3, 4)
    
    
    def testNoise(self):
        
        tf = Branin()
        
        X = lhcSample(tf.bounds, 10, seed=0)
        Y = [tf.f(x) for x in X]
        GP1 = GaussianProcess(MaternKernel3([1.0, 1.0]), X, Y, noise=1e-4)
        self.failUnlessEqual(GP1.noise, 1e-4)

        eif1 = EI(GP1)
        dopt1, _ = direct(eif1.negf, tf.bounds, maxiter=10)
        copt1, _ = cdirect(eif1.negf, tf.bounds, maxiter=10)
        mopt1, _ = maximizeEI(GP1, tf.bounds, maxiter=10)

        self.failUnlessAlmostEqual(dopt1, copt1, 4)
        self.failUnlessAlmostEqual(-dopt1, mopt1, 4)
        self.failUnlessAlmostEqual(-copt1, mopt1, 4)

        GP2 = GaussianProcess(MaternKernel3([1.0, 1.0]), X, Y, noise=0.01)
        self.failUnlessEqual(GP2.noise, 0.01)
        
        eif2 = EI(GP2)
        dopt2, _ = direct(eif2.negf, tf.bounds, maxiter=10)
        copt2, _ = cdirect(eif2.negf, tf.bounds, maxiter=10)
        mopt2, _ = maximizeEI(GP2, tf.bounds, maxiter=10)
        self.failUnlessAlmostEqual(dopt2, copt2, 4)
        self.failUnlessAlmostEqual(-dopt2, mopt2, 4)
        self.failUnlessAlmostEqual(-copt2, mopt2, 4)

        self.failIfAlmostEqual(dopt1, dopt2, 4)
        self.failIfAlmostEqual(copt1, copt2, 4)
        self.failIfAlmostEqual(mopt1, mopt2, 4)

        GP3 = GaussianProcess(MaternKernel3([1.0, 1.0]), X, Y, noise=0.1)
        self.failUnlessEqual(GP3.noise, 0.1)
        eif3 = EI(GP3)
        dopt3, _ = direct(eif3.negf, tf.bounds, maxiter=10)
        copt3, _ = cdirect(eif3.negf, tf.bounds, maxiter=10)
        mopt3, _ = maximizeEI(GP3, tf.bounds, maxiter=10)
        self.failUnlessAlmostEqual(dopt3, copt3, 4)
        self.failUnlessAlmostEqual(-dopt3, mopt3, 4)
        self.failUnlessAlmostEqual(-copt3, mopt3, 4)

        self.failIfAlmostEqual(dopt1, dopt3, 4)
        self.failIfAlmostEqual(copt1, copt3, 4)
        self.failIfAlmostEqual(mopt1, mopt3, 4)
        self.failIfAlmostEqual(dopt2, dopt3, 4)
        self.failIfAlmostEqual(copt2, copt3, 4)
        self.failIfAlmostEqual(mopt2, mopt3, 4)
        
        # for GP in [GP1, GP2, GP3]:
        #     mu, sigma2 = GP.posterior(X[0])
        #     print '\n%f, %f' % (mu, sigma2)
            # self.failUnlessEqual(mu, Y[0])
            # self.failUnlessEqual(sigma2, GP.noise)
        
    def testMaxEIPrior(self):

        # make sure that the prior works with the different methods of EI
        # maximization
        
        S5 = Shekel5()
        pX = lhcSample(S5.bounds, 100, seed=511)
        pY = [S5.f(x) for x in pX]
        prior = RBFNMeanPrior()
        prior.train(pX, pY, bounds=S5.bounds, k=10, seed=504)
        
        hv = .1
        hyper = [hv, hv, hv, hv]
        kernel = GaussianKernel_ard(hyper)
        
        # train GPs
        X = lhcSample(S5.bounds, 10, seed=512)
        Y = [S5.f(x) for x in X]
        
        # validation
        valX = list(x.copy() for x in X)
        valY = copy(Y)
        
        GP = GaussianProcess(kernel, X, Y, prior=prior)
        
        eif = EI(GP)
        copt, _ = cdirect(eif.negf, S5.bounds, maxiter=20)
        mopt, _ = maximizeEI(GP, S5.bounds, maxiter=20)

        self.failUnlessAlmostEqual(-copt, mopt, 2)
        
        for i in xrange(len(GP.X)):
            self.failUnless(all(valX[i]==GP.X[i]))
            self.failUnless(valY[i]==GP.Y[i])
        
        GP.prior.mu(GP.X[0])
        self.failUnless(all(valX[0]==GP.X[0]))
        # print GP.X
        
        for i in xrange(len(GP.X)):
            self.failUnless(all(valX[i]==GP.X[i]))
            self.failUnless(valY[i]==GP.Y[i])
        
        GP.prior.mu(GP.X[0])
        self.failUnless(all(valX[0]==GP.X[0]))
        # print GP.X
        

    def testPriorAndPrefs(self):
        
        S5 = Shekel5()
        
        pX = lhcSample(S5.bounds, 100, seed=13)
        pY = [S5.f(x) for x in pX]
        prior = RBFNMeanPrior()
        prior.train(pX, pY, S5.bounds, k=10)
        
        hv = .1
        hyper = [hv, hv, hv, hv]
        gkernel = GaussianKernel_ard(hyper)
        GP = PrefGaussianProcess(gkernel, prior=prior)
        
        X = [array([i+.5]*4) for i in xrange(5)]
        valX = [x.copy() for x in X]
        
        prefs = []
        for i in xrange(len(X)):
            for j in xrange(i):
                if S5.f(X[i]) > S5.f(X[j]):
                    prefs.append((X[i], X[j], 0))
                else:
                    prefs.append((X[j], X[i], 0))
        
        GP.addPreferences(prefs)
        opt, optx = maximizeEI(GP, S5.bounds)
        



class TestDataprior(unittest.TestCase):

    def testRBFN_1D(self):
        
        # sample from a synthetic function and see how much we improve the
        # error by using the prior function
        def foo(x):
            return sum(sin(x*20))
            
        X = lhcSample([[0., 1.]], 50, seed=3)
        Y = [foo(x) for x in X]
        
        prior = RBFNMeanPrior()
        prior.train(X, Y, [[0., 1.]], k=10, seed=100)
        
        # See how well we fit the function by getting the average squared error
        # over 100 samples of the function.  Baseline foo(x)=0 MSE is 0.48.
        # We will aim for MSE < 0.05.
        S = arange(0, 1, .01)
        error = mean([foo(x)-prior.mu(x) for x in S])
        self.failUnless(error < 0.05)

        # for debugging
        if False:
            figure(1)
            plot(S, [foo(x) for x in S], 'b-')
            plot(S, [prior.mu(x) for x in S], 'k-')
            show()
    
    
    def testRNFN_10D(self):
        
        # as above, but with a 10D test function and more data
        def foo(x):
            return sum(sin(x*2))
            
        bounds = [[0., 1.]]*10
        X = lhcSample(bounds, 100, seed=4)
        Y = [foo(x) for x in X]
        
        prior = RBFNMeanPrior()
        prior.train(X, Y, bounds, k=20, seed=5)
        
        S = lhcSample(bounds, 100, seed=6)
        RBNError = mean([(foo(x)-prior.mu(x))**2 for x in S])
        baselineError = mean([foo(x)**2 for x in S])
        
        # print '\nRBN err  =', RBNError
        # print 'baseline =', baselineError
        self.failUnless(RBNError < baselineError)
        
        
    def testGPPrior(self):
        
        # see how GP works with the dataprior...
        def foo(x):
            return sum(sin(x*20))
        
        bounds = [[0., 1.]]
        # train prior
        pX = lhcSample([[0., 1.]], 100, seed=6)
        pY = [foo(x) for x in pX]
        prior = RBFNMeanPrior()
        prior.train(pX, pY, bounds, k=10, seed=102)
        
        X = lhcSample([[0., 1.]], 2, seed=7)
        Y = [foo(x) for x in X]
        
        kernel = GaussianKernel_ard(array([.1]))
        GP = GaussianProcess(kernel, X, Y, prior=prior)
        GPnoprior = GaussianProcess(kernel, X, Y)

        S = arange(0, 1, .01)

        nopriorErr = mean([(foo(x)-GPnoprior.mu(x))**2 for x in S])
        priorErr = mean([(foo(x)-GP.mu(x))**2 for x in S])
        
        # print '\nno prior Err =', nopriorErr
        # print 'prior Err =', priorErr
        
        self.failUnless(priorErr < nopriorErr*.5)
        
        if False:
            figure(1)
            clf()
            plot(S, [prior.mu(x) for x in S], 'g-', alpha=0.3)
            plot(S, [GPnoprior.mu(x) for x in S], 'b-', alpha=0.3)
            plot(S, [GP.mu(x) for x in S], 'k-', lw=2)
            plot(X, Y, 'ko')
            show()
            
    
    def testShekelGPPrior(self):
        
        # see how the GP works on the Shekel function
        S5 = Shekel5()

        pX = lhcSample(S5.bounds, 100, seed=8)
        pY = [S5.f(x) for x in pX]
        prior = RBFNMeanPrior()
        prior.train(pX, pY, S5.bounds, k=10, seed=103)
        
        X = lhcSample(S5.bounds, 10, seed=9)
        Y = [S5.f(x) for x in X]

        hv = .1
        hyper = [hv, hv, hv, hv]
        gkernel = GaussianKernel_ard(hyper)
        priorGP = GaussianProcess(gkernel, X, Y, prior=prior)
        nopriorGP = GaussianProcess(gkernel, X, Y)
        
        S = lhcSample(S5.bounds, 1000, seed=10)
        nopriorErr = mean([(S5.f(x)-nopriorGP.mu(x))**2 for x in S])
        priorErr = mean([(S5.f(x)-priorGP.mu(x))**2 for x in S])
        
        # print '\nno prior Err =', nopriorErr
        # print 'prior Err =', priorErr
        self.failUnless(priorErr < nopriorErr*.8)
        
        
    def testMaxEIPrior(self):

        # make sure that the prior works with the different methods of EI
        # maximization
        
        S5 = Shekel5()
        pX = lhcSample(S5.bounds, 100, seed=511)
        pY = [S5.f(x) for x in pX]
        prior = RBFNMeanPrior()
        prior.train(pX, pY, bounds=S5.bounds, k=10, seed=504)
        
        hv = .1
        hyper = [hv, hv, hv, hv]
        kernel = GaussianKernel_ard(hyper)
        X = lhcSample(S5.bounds, 10, seed=512)
        Y = [S5.f(x) for x in X]
        GP = GaussianProcess(kernel, X, Y, prior=prior)
        
        # validation
        valX = list(x.copy() for x in X)
        valY = copy(Y)

        eif = EI(GP)
        dopt, _ = direct(eif.negf, S5.bounds, maxiter=20)
        copt, _ = cdirect(eif.negf, S5.bounds, maxiter=20)
        mopt, _ = maximizeEI(GP, S5.bounds, maxiter=20)

        self.failUnlessAlmostEqual(dopt, copt, 2)
        self.failUnlessAlmostEqual(-dopt, mopt, 2)
        
        for i in xrange(len(GP.X)):
            self.failUnless(all(valX[i]==GP.X[i]))
            self.failUnless(valY[i]==GP.Y[i])
        
        GP.prior.mu(GP.X[0])
        self.failUnless(all(valX[0]==GP.X[0]))
        # print GP.X
        
        for i in xrange(len(GP.X)):
            self.failUnless(all(valX[i]==GP.X[i]))
            self.failUnless(valY[i]==GP.Y[i])
        
        GP.prior.mu(GP.X[0])
        self.failUnless(all(valX[0]==GP.X[0]))
        # print GP.X
        
        

class TestPerformanceMeasures(unittest.TestCase):
    
    def testGDelta(self):
        
        # usually, Gdelta==G
        GP = GaussianProcess(GaussianKernel_iso([0.05]))
        X = lhcSample([[0., 1.]], 5, seed=10)
        Y = [x**2 for x in X]
        GP.train(X, Y)
        
        G = (Y[0]-max(Y)) / (Y[0]-1)
        self.failUnlessEqual(G, Gdelta(GP, [[0.,1.]], Y[0], 1.0, 0.01))
        
        # sometimes, though, Gdelta > G -- this GP has a very high confidence
        # prediction of a very good point at x ~ .65
        GP = GaussianProcess(GaussianKernel_iso([0.1]))
        X = array([[.5], [.51], [.59], [.6]])
        Y = array([1., 2., 2., 1.])
        GP.train(X, Y)
        # figure(1)
        # A = arange(0, 1, 0.01)
        # post = [GP.posterior(x) for x in A]
        # plot(A, [p[0] for p in post], 'k-')
        # plot(A, [p[0]+p[1] for p in post], 'k:')
        # show()
        G = (Y[0]-max(Y)) / (Y[0]-4.0)
        Gd = Gdelta(GP, [[0., 1.]], Y[0], 4.0, 0.01)
        self.failUnless(G < Gd)
        
        # however, if there is more variance, we will collapse back to G
        GP = GaussianProcess(GaussianKernel_iso([.001]))
        GP.train(X, Y)
        G = (Y[0]-max(Y)) / (Y[0]-4.0)
        self.failUnlessEqual(G, Gdelta(GP, [[0., 1.]], Y[0], 4.0, 0.01))


TestFunctions = [Poly4, Poly6, Schubert1, GoldsteinPrice, Shekel5, Camelback, Branin, Hartman3, Hartman6]
class TestTestFunctions(unittest.TestCase):
    
    def _testMinimization(self):
        
        for TestFunction in TestFunctions:
            tf = TestFunction(maximize=False)
            opt, _ = cdirect(tf.f, tf.bounds, maxiter=50)
            print '[%s] found minimum %f.  declared minimum was %f.' % (tf.name, opt, tf.minimum)
            self.failUnlessAlmostEqual(opt, tf.minimum, 2)
            
    def testFunctionValues(self):
        
        for TestFunction in TestFunctions:
            tf = TestFunction(maximize=False)
            for i in xrange(100):
                x = lhcSample(tf.bounds, 1, seed=i)[0]
                self.failIf(tf.f(x) < tf.minimum)
                

        for TestFunction in TestFunctions:
            tf = TestFunction(maximize=True)
            for i in xrange(100):
                x = lhcSample(tf.bounds, 1, seed=i)[0]
                self.failIf(tf.f(x) > -tf.minimum)

class TestGallery(unittest.TestCase):
    
    def testFastGallery(self):
        
        tf = Hartman3()
        kernel = tf.createKernel(GaussianKernel_ard)
        X = lhcSample(tf.bounds, 10, seed=23)
        Y = [tf.f(x) for x in X]
        prefs = query2prefs(X, tf.f)
        
        GP = PrefGaussianProcess(kernel)
        GP.addPreferences(prefs)
    
        gallery = fastUCBGallery(GP, tf.bounds, 4)
        print 'gallery returned:'
        for x in gallery:
            print '\t', x
            
        GP.addPreferences(query2prefs(gallery, tf.f))
        
        # make sure we don't return anything out of bounds
        bounds = copy(tf.bounds)
        bounds[0] = [0., 0.]
        gallery = fastUCBGallery(GP, bounds, 4)        
        print 'gallery returned:'
        for x in gallery:
            print '\t', x
            for v, b in zip(x, bounds):
                self.failUnless(v>=b[0] and v<=b[1])
                
            
            

if __name__ == '__main__':

    tests = [TestLatinhypercube, TestDIRECT, TestCDIRECT, TestMaximizePI, TestMaximizeEI, TestDataprior, TestTestFunctions, TestGallery]

    for test in tests:
        suite = unittest.TestLoader().loadTestsFromTestCase(test)
        unittest.TextTestRunner(verbosity=2).run(suite)
    