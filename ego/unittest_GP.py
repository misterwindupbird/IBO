#!python
"""
Unittests for the EGO module go here.
"""
from __future__ import division

import unittest
from math import sin
from copy import deepcopy

# from matplotlib.pylab import figure, plot, subplot, show, plt, cm, annotate, clf
from scipy import optimize
from numpy import *

from gaussianprocess import GaussianProcess, PrefGaussianProcess
from gaussianprocess.kernel import GaussianKernel_ard, GaussianKernel_iso, MaternKernel3, SVGaussianKernel_iso, SVGaussianKernel_ard
from gaussianprocess.prior import RBFNMeanPrior
from acquisition.prefutil import query2prefs
from utils.latinhypercube import lhcSample
from gaussianprocess.trainhyper import marginalLikelihood, nlml, dnlml
from utils.testfunctions import Poly4, Poly6, Schubert1, GoldsteinPrice, Shekel5, Camelback, Branin, Hartman3, Hartman6
# from utils.performance import Gdelta
# from optimize.direct import direct
# from optimize.cdirect import cdirect



    
class TestGaussianProcess(unittest.TestCase):
    
    def test1DGP(self):
        
        f = lambda x: float(sin(x*5.))
        X = lhcSample([[0., 1.]], 5, seed=25)
        Y = [f(x) for x in X]
        
        kernel = GaussianKernel_ard(array([1.0, 1.0]))
        GP = GaussianProcess(kernel, X=X, Y=Y)
        
    
    def testFromEmptyGP(self):
        # test a GP that has no data to start
        f = lambda x: float(sin(x*10)+x)
        kernel = GaussianKernel_iso(array([1.0]))
        GP = GaussianProcess(kernel)
        
        for x in arange(0., 1., .1):
            GP.addData(array([x]), f(x))
            
        for x in arange(1., 2., .1):
            GP.addData(array([x]), f(x))
        
        self.failUnlessEqual(len(GP.X), 20)


    def testShekelClass(self):
        
        S = Shekel5()
        
        # get 50 latin hypercube samples
        X = lhcSample(S.bounds, 50, seed=2)
        Y = [S.f(x) for x in X]
        
        hyper = [.2, .2, .2, .2]
        noise = 0.1
        
        gkernel = GaussianKernel_ard(hyper)
        # print gkernel.sf2
        GP = GaussianProcess(gkernel, X, Y, noise=noise)
        
        # let's take a look at the trained GP.  first, make sure variance at
        # the samples is determined by noise
        mu, sig2 = GP.posteriors(X)
        for m, s, y in zip(mu, sig2, Y):
            # print m, s
            self.failUnless(s < 1/(1+noise))
            self.failUnless(abs(m-y) < 2*noise)
        
        # now get some test samples and see how well we are fitting the function
        testX = lhcSample(S.bounds, 50, seed=3)
        testY = [S.f(x) for x in X]
        for tx, ty in zip(testX, testY):
            m, s = GP.posterior(tx)
            # prediction should be within one stdev of mean
            self.failUnless(abs(ty-m)/sqrt(s) < 1)
        
        
    def testTraining(self):
        
        # test that sequential training gives the same result as batch
        
        tf = Shekel5()
        X = lhcSample(tf.bounds, 25, seed=1)
        Y = [tf.f(x) for x in X]
        
        # GP1 adds all data during initialization
        GP1 = GaussianProcess(GaussianKernel_iso([.1]), X, Y, noise=.2)
        
        # GP2 adds data one at a time
        GP2 = GaussianProcess(GaussianKernel_iso([.1]), noise=.2)
        
        # GP3 uses addData()
        GP3 = GaussianProcess(GaussianKernel_iso([.1]), noise=.2)
        
        # GP4 adds using various methods
        GP4 = GaussianProcess(GaussianKernel_iso([.1]), X[:10], Y[:10], noise=.2)
        
        for x, y in zip(X, Y):
            GP2.addData(x, y)
            
        for i in xrange(0, 25, 5):
            GP3.addData(X[i:i+5], Y[i:i+5])
        
        GP4.addData(X[10], Y[10])
        GP4.addData(X[11:18], Y[11:18])
        for i in xrange(18, 25):
            GP4.addData(X[i], Y[i])
        
        
        self.failUnless(all(GP1.R==GP2.R))
        self.failUnless(all(GP1.R==GP3.R))
        self.failUnless(all(GP1.R==GP4.R))
        
        testX = lhcSample(tf.bounds, 25, seed=2)
        for x in testX:
            mu1, s1 = GP1.posterior(x)
            mu2, s2 = GP2.posterior(x)
            mu3, s3 = GP3.posterior(x)
            mu4, s4 = GP4.posterior(x)
            self.failUnlessEqual(mu1, mu2)
            self.failUnlessEqual(mu1, mu3)
            self.failUnlessEqual(mu1, mu4)
            self.failUnlessEqual(s1, s2)
            self.failUnlessEqual(s1, s3)
            self.failUnlessEqual(s1, s4)



class TestHyper(unittest.TestCase):
    
    # these tests mostly compare the computer result to results collected from
    # Carl Rasmussen's MATLAB code
    
    def setUp(self):
        
        self.X = [array([.5, .1, .3]),
             array([.9, 1.2, .1]),
             array([.55, .234, .1]),
             array([.234, .547, .675])]
        self.Y = array([.5, 1., .5, 2.])
        

    def testARDGaussianKernelHyperparameterLearning(self):

        hyper = array([2., 2., .1])
        
        # test derivatives
        target0 = matrix('[0 .0046 .0001 0; .0046 0 .0268 0; .0001 .0268 0 0; 0 0 0 0]')
        target1 = matrix('[0 .0345 .0006 0; .0345 0 .2044 0; .0006 .2044 0 0; 0 0 0 0]')
        target2 = matrix('[0 .4561 .54 .012; .4561 0 0 0; .54 .0 0 0; .012 0 0 0]')
        target3 = matrix('[2 .2281 .27 .0017; .2281 2 1.7528 0; .27 1.7528 2 0; .0017 0 0 2]')
        pder0 = GaussianKernel_ard(hyper).derivative(self.X, 0)
        pder1 = GaussianKernel_ard(hyper).derivative(self.X, 1)
        pder2 = GaussianKernel_ard(hyper).derivative(self.X, 2)
        # pder3 = GaussianKernel_ard(hyper).derivative(self.X, 3)
        
        epsilon = .0001
        
        for i, (target, pder) in enumerate([(target0, pder0), (target1, pder1), (target2, pder2)]):
            for j in xrange(4):
                for k in xrange(4):
                    if abs(target[j, k]-pder[j, k]) > epsilon:
                        print '\nelement [%d, %d] of pder%d differs from expected by > %f' % (j, k, i, epsilon)
                        print '\ntarget:'
                        print target
                        print '\npder:'
                        print pder
                        assert False
        
        # marginal likelihood and likelihood gradient
        gkernel = GaussianKernel_ard(hyper)
        margl, marglderiv = marginalLikelihood(gkernel, self.X, self.Y, len(hyper), useCholesky=True)        
        self.assertAlmostEqual(margl, 5.8404, 2)
        for d, t in zip(marglderiv, [0.0039, 0.0302, -0.1733, -1.8089]):
            self.assertAlmostEqual(d, t, 2)
            
        # make sure we're getting the same results for inversion and cholesky
        imargl, imarglderiv = marginalLikelihood(gkernel, self.X, self.Y, len(hyper), useCholesky=False) 
        self.assertAlmostEqual(margl, imargl, 2)
        for c, i in zip(marglderiv, imarglderiv):
            self.assertAlmostEqual(c, i, 2)
        
        
        # optimize the marginal likelihood over log hyperparameters using BFGS
        hyper = array([2., 2., .1, 1.])
        argmin = optimize.fmin_bfgs(nlml, log(hyper), dnlml, args=[SVGaussianKernel_ard, self.X, self.Y], disp=False)
        for v, t in zip(argmin, [6.9714, 0.95405, -0.9769, 0.36469]):
            self.assertAlmostEqual(v, t, 2)
        
        
    def testIsotropicGaussianKernelHyperparameterLearning(self):
        
        
        hyper = array([1.5, 1.1])
        gkernel = SVGaussianKernel_iso(hyper)
        
        # marginal likelihood and likelihood gradient
        #
        # in MATLAB:
        #    [nlml dnlml] = gpr(log([1.5, 1.1])', 'covSEiso', 
        #           [.5, .1, .3; .9, 1.2, .1; .55, .234, .1; .234, .547, .675] , 
        #           [.5, 1, .5, 2]')
        margl, marglderiv = marginalLikelihood(gkernel, self.X, self.Y, len(hyper), True) 
        self.assertAlmostEqual(margl, 7.514, 2)
        for v, t in zip(marglderiv, [11.4659, -10.0714]):
            self.assertAlmostEqual(v, t, 2)
        
        # compare partial derivatives with result from Rasmussen's code
        target0 = matrix('[0 .5543 .0321 .2018; .5543 0 .449 .4945; .0321 .449 0 .2527; .2018 .4945 .2527 0]')
        target1 = matrix('[2.42 1.769 2.3877 2.2087; 1.769 2.42 1.914 1.8533; 2.3877 1.914 2.42 2.1519; 2.2087 1.8533 2.1519 2.42]')
        pder0 = gkernel.derivative(self.X, 0)
        pder1 = gkernel.derivative(self.X, 1)
        for i, (target, pder) in enumerate([(target0, pder0), (target1, pder1)]):
            for j in xrange(4):
                self.assertAlmostEqual(target[i,j], pder[i,j], 2)
            
        # optimize the marginal likelihood over the log hyperparameters 
        # using BFGS
        argmin = optimize.fmin_bfgs(nlml, log(hyper), dnlml, args=[SVGaussianKernel_iso, self.X, self.Y], disp=False)
        for d, t in zip(argmin, [-0.0893, 0.29]):
            self.assertAlmostEqual(d, t, 2)
            
            
    def testMaternKernelHyperparameterLearning(self):
        
        # we'll just confirm the likelihood & likelihood gradient, rather than 
        # all the intermediate steps.  you can use the Gaussian kernel examples
        # if you want to figure out the partial derivatives.
        hyper = array([1.5, 1.1])

        mkernel3 = MaternKernel3(hyper)
        margl3, marglderiv3 = marginalLikelihood(mkernel3, self.X, self.Y, len(hyper)) 
        self.assertAlmostEqual(margl3, 5.1827, 2)
        self.assertAlmostEqual(marglderiv3[0], 1.6947897766, 2)
        self.assertAlmostEqual(marglderiv3[1], -2.9350, 2)

        # mkernel5 = MaternKernel5(hyper)
        # margl5, marglderiv5 = marginalLikelihood(mkernel5, self.X, self.Y, len(hyper)) 
        # self.assertAlmostEqual(margl5, 5.6652, 4)
        # self.assertAlmostEqual(marglderiv5[0], 4.4782, 4)
        # self.assertAlmostEqual(marglderiv5[1], -4.8737, 4)
            

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
        
        hv = .1
        hyper = [hv, hv, hv, hv]
        gkernel = GaussianKernel_ard(hyper)
        X = lhcSample(S5.bounds, 10, seed=9)
        Y = [S5.f(x) for x in X]
        priorGP = GaussianProcess(gkernel, X, Y, prior=prior)
        nopriorGP = GaussianProcess(gkernel, X, Y, prior=None)
        
        
        S = lhcSample(S5.bounds, 1000, seed=10)
        nopriorErr = mean([(S5.f(x)-nopriorGP.mu(x))**2 for x in S])
        priorErr = mean([(S5.f(x)-priorGP.mu(x))**2 for x in S])
        
        # print '\nno prior Err =', nopriorErr
        # print 'prior Err =', priorErr
        self.failUnless(priorErr < nopriorErr*.8)
        
        

class TestPreferences(unittest.TestCase):
    
    def test1DPreferences(self):
        
        showit = False  # show figures for debugging
        
        x1 = array([.2])
        x2 = array([.7])
        x3 = array([.4])
        x4 = array([.35])
        x5 = array([.9])
        x6 = array([.1])
        
        GP = PrefGaussianProcess(GaussianKernel_ard(array([.1])))
        GP.addPreferences([(x1, x2, 0)])
        self.failUnless(GP.mu(x1) > GP.mu(x2))
        # print GP.X
        
        if showit:
            figure(1)
            clf()
            S = arange(0, 1, .01)
            ax = subplot(1, 3, 1)
            ax.plot(S, [GP.mu(x) for x in S], 'k-')
            ax.plot(GP.X, GP.Y, 'ro')
        
        GP.addPreferences([(x3, x4, 0)])
        self.failUnless(GP.mu(x1) > GP.mu(x2))
        self.failUnless(GP.mu(x3) > GP.mu(x4))

        if showit:
            ax = subplot(1, 3, 2)
            ax.plot(S, [GP.mu(x) for x in S], 'k-')
            ax.plot(GP.X, GP.Y, 'ro')

        # x5 is greatly preferred to x6 - we should expect f(x5)-f(x6) to have 
        # the most pronounced difference
        GP.addPreferences([(x5, x6, 1)])
        self.failUnless(GP.mu(x1) > GP.mu(x2))
        self.failUnless(GP.mu(x3) > GP.mu(x4))
        self.failUnless(GP.mu(x5) > GP.mu(x6))
        self.failUnless(GP.mu(x5)-GP.mu(x6) > GP.mu(x1)-GP.mu(x2))
        self.failUnless(GP.mu(x5)-GP.mu(x6) > GP.mu(x3)-GP.mu(x4))
        
        if showit:
            ax = subplot(1, 3, 3)
            ax.plot(S, [GP.mu(x) for x in S], 'k-')
            ax.plot(GP.X, GP.Y, 'ro')

            show()
            
        
    def test5DPreferences(self):
        
        def foo(x):
            return sum(x)
            
        X = [array([.1, .2, .3, .4]),
             array([.7, .2, .3, .4]),
             array([.5, .5, .5, .5]),
             array([.5, .5, .5, .5]),
             array([.5, .8, .5, .5]),
             array([.5, .9, .9, .1]),
             array([.2, .6, .1, .5])]
        
        GP = PrefGaussianProcess(GaussianKernel_iso(array([.1])))
        for i in xrange(len(X)):
            for j in xrange(i):
                if foo(X[i]) == foo(X[j]):
                    continue
                elif foo(X[i]) > foo(X[j]):
                    p = [X[i], X[j], 0]
                else:
                    p = [X[j], X[i], 0]
                GP.addPreferences([p])
        
        for i in xrange(len(GP.X)):
            for j in xrange(i):
                # print 'foo = %.1f, GP = %.3f vs foo = %.1f, GP = %.3f' % (foo(GP.X[i]), GP.Y[i], foo(GP.X[j]), GP.Y[j])
                if foo(GP.X[i]) > foo(GP.X[j]):
                    self.failUnless(GP.mu(GP.X[i]) > GP.mu(GP.X[j]))
                else:
                    self.failUnless(GP.mu(GP.X[j]) > GP.mu(GP.X[i]))
        
        
    def testQuery2Prefs(self):
        
        def foo(x):
            return sum(x)
            
        X = [array([.1, .2, .3, .4]),
             array([.7, .2, .3, .4]),
             array([.5, .5, .5, .5]),
             array([.5, .5, .5, .5]),
             array([.5, .8, .5, .5]),
             array([.5, .9, .9, .1]),
             array([.2, .6, .1, .5])]
        
        prefs = query2prefs(X, foo)
        
        for u, v, d in prefs:
            self.failUnless(d==0)
            self.failUnless(foo(u) > foo(v))
            
            


if __name__ == '__main__':
    tests = [TestGaussianProcess, TestDataprior, TestPreferences, TestHyper]
    # tests = [TestHyper]
    for test in tests:
        suite = unittest.TestLoader().loadTestsFromTestCase(test)
        unittest.TextTestRunner(verbosity=2).run(suite)
    