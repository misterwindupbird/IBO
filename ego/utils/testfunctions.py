#!/usr/bin/env python
# encoding: utf-8
"""
testfuncs.py

Created by Eric on 2009-12-28.
Copyright (c) 2009 Eric Brochu. All rights reserved.

Functions used for testing optimizations.
"""
from __future__ import division

import unittest

from numpy import *
from numpy.random import normal
from matplotlib.pyplot import *
from scipy.optimize import fmin_bfgs

from ego.gaussianprocess import GaussianProcess
from ego.gaussianprocess.kernel import GaussianKernel_iso, GaussianKernel_ard, MaternKernel3
from ego.utils.latinhypercube import lhcSample
from ego.gaussianprocess.trainhyper import *



class TestFunction(object):

    def __init__(self, name, minimum, argmin, bounds, maximize=True, kernelHP=None, **kwargs):
        super(TestFunction, self).__init__()
        self.name = name
        self.maximize = maximize
        self.minimum = minimum
        self.argmin = argmin
        self.bounds = bounds
        
        self.defaultHP = kernelHP
        
        
    def f(self, x):
        """ return f(x).  if maximize=True (the default), return -f(x) """
        pass
        
    def createKernel(self, Kernel):
        """
        given a kernel class, create an object of that class with the right
        hyperparameters
        """
        if self.defaultHP is None or Kernel not in self.defaultHP:
            raise ValueError('test function %s has no default values for kernel %s'%(self.name, Kernel.__name__))
        return Kernel(array(self.defaultHP[Kernel]))
        

class Poly4(TestFunction):
    """ 4th-order poly """
    def __init__(self, **kwargs):
        super(Poly4, self).__init__("4th-order poly", 0, [0], array([[-10, 10]], dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [1.620, 1],
                          GaussianKernel_ard: [1.628, 1],
                          MaternKernel3: [4.635, 1]}
    
    def f(self, x):
        y = abs(x[0]**3 + x[0]**2 + x[0])
        
        y /= 100.0  # rescale
        
        if self.maximize:
            return -y
        return y
        
        
class Poly6(TestFunction):
    """ 6th-order poly """
    def __init__(self, **kwargs):
        super(Poly6, self).__init__("Goldstein 6th-order", 0.07, [0], array([[-4, 4]], dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.285, 1],
                          GaussianKernel_ard: [0.287, 1],
                          MaternKernel3: [0.678, 1]}
    
    def f(self, x):
        y = ((x[0]**2  -15) * x[0]**2 + 27) * x[0]**2 + 250
        
        y /= 100
        
        if self.maximize:
            return -y
        return y


class Schubert1(TestFunction):
    """ 1D Schubert function """
    def __init__(self, **kwargs):
        super(Schubert1, self).__init__("Schubert", -8.5178, [-0.195], array([[-1, 1]], dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.192, 1],
                          GaussianKernel_ard: [0.192, 1],
                          MaternKernel3: [0.279, 1]}
    
    def f(self, x):
        y = sum(i * cos((i+1)*x[0] + i) for i in range(1,6))
        if self.maximize:
            return -y
        return y

        
        
class GoldsteinPrice(TestFunction):

    """
    Goldstein-Price function
    
    d = 2
    
    fmin = 3
    """
    def __init__(self, **kwargs):
        super(GoldsteinPrice, self).__init__("Goldstein-Price", 1.0986, ones(2), array([[-2, 2]]*2, dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.376],
                          GaussianKernel_ard: [0.428, 0.383],
                          MaternKernel3: [0.888, 1]}
    
    def f(self, x):
        y = (1 + (sum(x) + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) * (30 + (2*x[0]-3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
        if self.maximize:
            return -log(y)
        return log(y)
    


        
class Shekel(TestFunction):

    def __init__(self, name, minimum, argmin, **kwargs):
        super(Shekel, self).__init__(name, minimum, argmin, array([[0.,10.]]*4), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [4.750, 1],
                          GaussianKernel_ard: [5.146, 4.189, 4.622, 5.843, 1],
                          MaternKernel3: [15.0, 1]}
        
        self.A = array([[4]*4,
                       [1]*4,
                       [8]*4,
                       [6]*4,
                       [3, 7, 3, 7],
                       [2, 9, 2, 9],
                       [5, 5, 3, 3],
                       [8, 1, 8, 1],
                       [6, 2, 6, 2],
                       [7, 3.6, 7, 3.6]], dtype=float)
        self.C = array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
        
        
class Shekel5(Shekel):

    def __init__(self, **kwargs):
        super(Shekel5, self).__init__("Shekel 5", -10.1532, array([4.0]*4), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.245],
                          GaussianKernel_ard: [0.245, 0.245, 0.245, 0.245]}
        
    def f(self, x):
        y = -sum(1./(dot(x-a, x-a)+c) for a, c in zip(self.A[:5], self.C[:5]))
        if self.maximize:
            return -y
        return y


class Shekel7(Shekel):

    def __init__(self, **kwargs):
        super(Shekel7, self).__init__("Shekel 7", -10.4029, array([4.0]*4), **kwargs)
        
    def f(self, x):
        y = -sum(1./(dot(x-a, x-a)+c) for a, c in zip(self.A[:7], self.C[:7]))
        if self.maximize:
            return -y
        return y


class Shekel10(Shekel):

    def __init__(self, **kwargs):
        super(Shekel10, self).__init__("Shekel 10", -10.5364, array([4.0]*4), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.9]}
        
    def f(self, x):
        y = -sum(1./(dot(x-a, x-a)+c) for a, c in zip(self.A, self.C))
        if self.maximize:
            return -y
        return y
        
        
class Camelback(TestFunction):
    """
    six-hump camelback function
    
    d = 2
    
    6 local minima, 2 global minima at [-0.0898, 0.7126], [0.0898, -0.7126]
    """
    def __init__(self, **kwargs):
        super(Camelback, self).__init__("6-Hump Camelback", -1.032, None, array([[-2, 2], [-1, 1]], dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.384],
                          GaussianKernel_ard: [0.393, 0.387, 1],
                          MaternKernel3: [0.842, 1]}
    
    def f(self, x):
        y = (4 - 2.1 * x[0]**2 + x[0]**4/3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2
        if self.maximize:
            return -y
        return y
            
            
class Branin(TestFunction):
    """
    Branin function
    
    d = 2
    
    3 global minima
    """
    def __init__(self, **kwargs):
        super(Branin, self).__init__("Branin", 0.004, array([3.142,  4.275]), array([[-5, 10], [0, 15]], dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [3.8],
                          GaussianKernel_ard: [3.4, 10.0]}
    
    def f(self, x):
        y = (x[1] - 2 - 5.1/(4*pi**2)*x[0]**2 + 5/pi * x[0] - 6)**2 + 10 * (1 - 1/(8*pi)) * cos(x[0]) + 10

        y /= 100
        if self.maximize:
            return -y
        return y
        

        
class Hartman3(TestFunction):
    """
    Hartman, d = 3.
    """
    def __init__(self, **kwargs):
        super(Hartman3, self).__init__("Hartman 3", -3.86278 , array([.114614, .555649, 0.852547]), array([[0, 1]]*3, dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.225],
                          GaussianKernel_ard: [1.1, 0.31, 0.17]}
        
        self.A = array([[3, 10, 30],
                        [.1, 10, 35],
                        [3, 10, 30],
                        [.1, 10, 35]], dtype=float)
        self.B = array([[0.3689, 0.1170, 0.2673],
                        [0.4699, 0.4387, 0.7470],
                        [0.1091, 0.8732, 0.5547],
                        [0.03825, 0.5743, 0.8828]], dtype=float)
        self.C = array([1, 1.2, 3, 3.2], dtype=float)
        
    def f(self, x):
        y = -sum(self.C[i] * exp(-sum(self.A[i] * (x-self.B[i])**2)) for i in xrange(4))
        if self.maximize:
            return -y
        return y


class Hartman6(TestFunction):
    """
    Hartman, d = 6
    """
    def __init__(self, **kwargs):
        super(Hartman6, self).__init__("Hartman 6", -3.3224 , array([0.2017, 0.15, 0.4769, 0.2753, 0.3117, 0.6573]), array([[0, 1]]*6, dtype=float), **kwargs)
        self.defaultHP = {GaussianKernel_iso: [0.39],
                          GaussianKernel_ard: [0.53, 0.57, 2.5, 0.34, 0.27, 0.35]}
        
        self.A = array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]], dtype=float)
        self.B = array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]], 
                        dtype=float)
        self.C = array([1, 1.2, 3, 3.2], dtype=float)
    
    def f(self, x):
        y = -sum(self.C[i] * exp(-sum(self.A[i] * (x-self.B[i])**2)) for i in xrange(4))
        if self.maximize:
            return -y
        return y



class Levy(TestFunction):
    """
    d-dimensional Levy function.
    """
    def __init__(self, d=2, **kwargs):
        super(Levy, self).__init__("Levy %d"%d, 0, ones(d), [[-10.0, 10.0]]*d, **kwargs)
        self.defaultHP = {}
        self.d = defaultHP
    
    def createKernel(self, Kernel):
        if Kernel==GaussianKernel_iso:
            defaultHP = {2: 0.9, 4: 2.8}
            return Kernel(array([defaultHP[self.d]]))
        else:
            raise ValueError('test function %s has no default values for kernel %s'%(self.name, Kernel.__name__))

    
    def f(self, x):
        
        z = 1+(x-1)/4.0
        s = sin(pi*z[0])**2
        for zz in z[:-1]:
            s += (zz-1)**2 * (1+10*(sin(pi*zz+1))**2)
        y = s + (z[-1]-1)**2*(1+sin(2*pi*z[-1])**2)
        if self.maximize:
            return -y
        return y


class Michalewics(TestFunction):
    """
    d-dimensional Michalewics function.  Valid values of d are {2, 5, 10}.
    """
    def __init__(self, d=2, **kwargs):
        assert d in [2, 5, 10]
        self.d = d
        mins = {2: -1.8014, 5: -4.687658, 10: -9.66015}
        argmins = {2: array([2.2029, 1.5708]), 5: None, 10: None}
        super(Michalewics, self).__init__("Michalewics %d"%d, mins[d], argmins[d], [[0.0, pi]]*d, **kwargs)
        self.defaultHP = {}
        
    def createKernel(self, Kernel):
        if Kernel==GaussianKernel_iso:
            defaultHP = {2: 0.28, 5: 0.68, 10: 1.36}
            return Kernel(array([defaultHP[self.d]]))
        else:
            raise ValueError('test function %s has no default values for kernel %s'%(self.name, Kernel.__name__))


    def f(self, x):
        
        y = -sum(sin(x)*sin(arange(1, self.d+1)*x**2/pi)**20)
        if self.maximize:
            return -y
        return y
    

class Perm(TestFunction):
    """
    d-dimensional Perm function, with Beta=0.5
    """
    def __init__(self, d=4, beta=0.5, **kwargs):
        super(Perm, self).__init__("Perm %d"%d, 0, arange(1, d+1), [[-d-1, d+1]]*d, **kwargs)
        self.beta = beta
        self.d = d
        
    def f(self, x):
        j = arange(1, self.d+1)
        y = 0
        for k in j:
            y += sum((j**k+self.beta)*((x/j)**k-1))**2
        if self.maximize:
            return -y
        return y


class Sphere(TestFunction):
    """
    d-dimensional Sphere function
    """
    def __init__(self, d=4, **kwargs):
        super(Sphere, self).__init__("Sphere %d"%d, 0, zeros(d), [[-5.12, 5.12]]*d, **kwargs)
        self.d = d
        
    def f(self, x):
        y = sum(x**2)
        if self.maximize:
            return -y
        return y


class SumSquares(TestFunction):
    """
    d-dimensional scaled sphere function
    """
    def __init__(self, d=4, **kwargs):
        super(SumSquares, self).__init__("SumSquares %d"%d, 0, zeros(d), [[-10.0, 10.0]]*d, **kwargs)
        self.d = d
    

    def createKernel(self, Kernel):
        if Kernel==GaussianKernel_iso:
            defaultHP = {2: 2.42, 4: 5.3, 8: 0.5}
            return Kernel(array([defaultHP[self.d]]))
        else:
            raise ValueError('test function %s has no default values for kernel %s'%(self.name, Kernel.__name__))

    
    
    def f(self, x):
        y = sum(arange(1, self.d+1)*x**2)
        if self.maximize:
            return -y
        else:
            return y
            

class Zakharov(TestFunction):
    """
    d-dimensional Zakharov function
    """
    def __init__(self, d=2, **kwargs):
        super(Zakharov, self).__init__("Zakharov %d"%d, 0, zeros(d), [[-5.0, 10.0]]*d, **kwargs)
        self.d = d
        self.defaultHP = {GaussianKernel_iso: [0.5]}
        
        
    def f(self, x):
        a = sum(0.5*arange(1, self.d+1)*x)
        y = sum(x**2) + a**2 + a**4
        if self.maximize:
            return -y
        return y
        
        

######## code from interweb doesn't work
# class Powell(TestFunction):
#     """
#     d-dimensional test function.  d must be multiple of 4.
#     """
#     def __init__(self, d=24, **kwargs):
#         assert d%4==0
#         super(Powell, self).__init__("Powell %d"%d, 0, array([3.0, -1.0, 0.0, 1.0]*(d//4)).reshape(-1), [[-4.0, 5.0]]*d, **kwargs)
#         self.d = d
#     
#     def f(self, x):
#         fv = zeros(self.d)
#         for i in xrange(1, (self.d//4)+1):
#             fv[4*i-4] = x[4*i-4] + 10*x[4*i-3]
#             fv[4*i-3] = sqrt(5) * (x[4*i-2]-x[4*i-1])
#             fv[4*i-2] = (x[4*i-3]-2*(x[4*i-2]))**2
#             fv[4*i-1] = sqrt(10) * (x[4*i-4]-x[4*i-1])**2
# 
#         y = linalg.norm(fv)**2
#         if self.maximize:
#             return -y
#         return y

############################################################################
#############             synthetic test function
############################################################################
class Synthetic(TestFunction):
    """
    randomly-generated synthetic function
    """
    def __init__(self, kernel, bounds, NX, noise=0.05, xstar=None, **kwargs):
        super(Synthetic, self).__init__("Synthetic", 0, None, bounds, **kwargs)
        
        self.name += ' %d'%len(bounds)
        
        self.GP = GaussianProcess(kernel)
        X = lhcSample(bounds, NX)
        self.GP.addData([X[0]], [normal(0, 1)])
        if xstar is not None:
            ystar = min(self.GP.Y[0]-1.0, -2.0)
            self.GP.addData(xstar, ystar)
        for x in X[1:]:
            mu, sig2 = self.GP.posterior(x)
            y = normal(mu, sqrt(sig2)) + normal(0, noise)
            # preserve min if necessary
            if xstar is not None and y < ystar+.5:
                y = ystar+.5
            self.GP.addData(x, y)
            
        # now, try minimizing with BFGS
        start = self.GP.X[argmin(self.GP.Y)]
        xopt = fmin_bfgs(self.GP.mu, start, disp=False)
        
        print "\t[synthetic] optimization started at %s, ended at %s" % (start, xopt)
        
        if xstar is not None:
            print '\t[synthetic] realigning minimum'
            # now, align minimum with what we specified
            for i, (target, origin) in enumerate(zip(xstar, xopt)):
                self.GP.X[:,i] += target-origin
            xopt = xstar
            
        
        self.minimum = self.GP.mu(xopt)
        self.xstar = xopt
        
        # print self.GP.X
        # print self.GP.Y
        print '\t[synthetic] x+ = %s, f(x+) = %.3f' % (self.xstar, self.f(self.xstar))
            
            
    def f(self, x):
        
        y = self.GP.mu(x)
        if y < self.minimum:
            self.minimum = y
            
        if self.maximize:
            return -y
        else:
            return y


        
############################################################################
#############              utilities
############################################################################
def learnHyper(tf, Kernel):
    """
    for a given kernel and test functions, learn some hyperparameters
    """
    D = len(tf.bounds)
    X = lhcSample(tf.bounds, D*40)
    Y = array([tf.f(x) for x in X])
    loghyper = fmin_bfgs(nlml, log(ones(1)*.5), dnlml, args=[Kernel, X, Y])
    return exp(loghyper)
    
    
    
def checkMinimum(testfuncs):
    """
    try minimizing the function and see if we find a better minimum than we 
    currently have
    """
    for tf in testfuncs:
        argmin = fmin_bfgs(tf.f, tf.argmin)
        print '[%s] was told argmin = %s, min = %.2f' % (tf.name, tf.argmin, tf.minimum)
        print '[%s] check argmin = %s, min = %.2f' % (tf.name, tf.argmin, tf.f(tf.argmin))
        print '[%s] found argmin = %s, min = %.2f' % (tf.name, argmin, tf.f(argmin))
        for x in lhcSample(tf.bounds, 100):
            if tf.f(x) < tf.minimum:
                print 'sample x = %s, y = %.4f is lower than minimum %.4f' % (x, tf.f(x), tf.minimum)
        
        
def plot2D(tf):
    """
    make and display a 2D synthetic function
    """
    # S2 = Synthetic(GaussianKernel_iso([.4]), [[-1,1]]*2, 15, xstar=[0,0])
    N = 50
    c0 = [(i/N)*(tf.bounds[0][1]-tf.bounds[0][0])+tf.bounds[0][0] for i in xrange(N+1)]
    c1 = [(i/N)*(tf.bounds[1][1]-tf.bounds[1][0])+tf.bounds[1][0] for i in xrange(N+1)]
    z = array([[tf.f(array([i,j])) for i in c0] for j in c1])
    figure(1)
    clf()
    ax = subplot(111)
    # cs = ax.contour(c0, c1, z, 50, alpha=0.5, cmap=cm.jet)
    cs2 = ax.contourf(c0, c1, z, 50, alpha=0.9, cmap=cm.jet)
    colorbar(cs2)
    if tf.argmin is not None:
        ax.plot(tf.argmin[0], tf.argmin[1], 'wo')
    ax.set_xbound(tf.bounds[0][0], tf.bounds[0][1])
    ax.set_ybound(tf.bounds[1][0], tf.bounds[1][1])
    draw()
    



if __name__=="__main__":
    
    # test2D()
    # show()
    # checkMinimum([Zakharov(4, maximize=False), Zakharov(10, maximize=False)])
    # plot2D(Zakharov(2, maximize=False))
    # show()
    vals = []
    for tf in [Michalewics(2), SumSquares(2), SumSquares(4), SumSquares(8)]:
        hp = learnHyper(tf, GaussianKernel_iso)
        assert len(hp)==1
        vals.append((tf.name, hp[0]))
    
    print '\nRESULTS:'
    for name, hp in vals:
        print '%s\t%.4f' % (name, hp)
    