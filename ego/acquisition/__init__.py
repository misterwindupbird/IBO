#!/usr/bin/env python
# encoding: utf-8

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

"""
acquisition.py

Created by Eric on 2010-02-16.
"""

from numpy import array, sqrt, max, arange, log, nan, linalg, isscalar, pi
from matplotlib.pylab import figure, subplot, poly_between, draw, show
from ctypes import *
import ctypes.util
import os.path

from ego.gaussianprocess import GaussianProcess, PrefGaussianProcess, CDF, PDF
from ego.randomforest import RandomForest
from ego.gaussianprocess.kernel import GaussianKernel_iso, GaussianKernel_ard, MaternKernel3, MaternKernel5
from ego.utils.optimize import direct, cdirect
from ego.utils.latinhypercube import lhcSample


class oldUCB(object):
    def __init__(self, GP, NA, T, delta=0.1, **kwargs):
        super(UCB, self).__init__()
        self.GP = GP
        tau = NA / 2
        self.sBeta = sqrt(4 * log(2*T**(tau+1) / delta)**2)
    
    def negf(self, x):
        
        mu, sig2 = self.GP.posterior(x)
        return -(mu + self.sBeta * sqrt(sig2))
        
    def f(self, x):
        
        return -self.negf(x)


class UCB(object):
    def __init__(self, GP, NA, delta=0.1, scale=0.2, **kwargs):
        super(UCB, self).__init__()
        self.GP = GP
        self.scale = scale
        t = len(self.GP.Y)+1
        self.sBeta = sqrt(2.0*log(t**(NA/2+2) * pi**2 / (3.0*delta)))
    
    def negf(self, x):
        
        mu, sig2 = self.GP.posterior(x)
        return -(mu + sqrt(self.scale * self.sBeta) * sqrt(sig2))
        
    def f(self, x):
        
        return -self.negf(x)


def maximizeUCB(model, bounds, delta=0.1, scale=0.2, useCDIRECT=True, maxiter=50, maxtime=30, maxsample=10000, **kwargs):
    """
    Maximize the upper confidence bound (UCB), as described in 
    [Srinivas 2009a].
    """
    if not useCDIRECT:
        print 'using DIRECT'
        ucb = UCB(model, len(bounds), delta=delta, scale=scale, **kwargs)
        opt, optx = direct(ucb.negf, bounds, **kwargs)
        return -opt, optx
        
    if isinstance(model, GaussianProcess):
        return cdirectGP(model, bounds, maxiter, maxtime, maxsample, acqfunc='ucb', delta=delta, scale=scale, **kwargs)

    elif isinstance(model, RandomForest):
        return cdirectRF(model, bounds, maxiter, maxtime, maxsample, acqfunc='ucb', delta=delta, scale=scale, **kwargs)

    else:
        raise ValueError
    
    

class PI(object):

    def __init__(self, GP, xi=.01, **kwargs):
        super(PI, self).__init__()
        self.GP = GP
        self.Z = max(self.GP.Y) + xi
        
    def negf(self, x):
        
        mu, sig2 = self.GP.posterior(x)
        return -CDF((mu - self.Z) / sqrt(sig2))
        
    def f(self, x):
        
        return -self.negf(x)


def maximizePI(model, bounds, xi=0.01, maxiter=50, maxtime=30, maxsample=10000, useCDIRECT=True, **kwargs):
    """
    Maximize the probability of improvement, as described in [Lizotte 2008].
    """
    if not useCDIRECT:
        print 'using DIRECT'
        pi = PI(model, xi, **kwargs)
        opt, optx = direct(pi.negf, bounds, maxiter=maxiter, maxtime=maxtime, maxsample=maxsample, **kwargs)
        return -opt, optx
    
    if isinstance(model, GaussianProcess):
        return cdirectGP(model, bounds, maxiter, maxtime, maxsample, acqfunc='pi', xi=xi, **kwargs)
    
    elif isinstance(model, RandomForest):
        return cdirectRF(model, bounds, maxiter, maxtime, maxsample, acqfunc='pi', xi=xi, **kwargs)
    
    else:
        raise ValueError



class EI(object):
    
    def __init__(self, GP, xi=.01, **kwargs):
        super(EI, self).__init__()
        self.GP = GP
        self.ymax = max(self.GP.Y)
        self.xi = xi

        assert isscalar(self.ymax)
        assert isscalar(self.xi)
        

    def negf(self, x):
        
        mu, sig2 = self.GP.posterior(x)
        assert isscalar(mu)
        assert isscalar(sig2)
        
        ydiff = mu - self.ymax - self.xi
        s = sqrt(sig2)
        Z = float(ydiff / s)

        EI = (ydiff * CDF(Z)) + (s * PDF(Z))
        if EI is nan:
            return 0.
        # print '[python] EI =', EI
        return -EI
    

    def f(self, x):
        
        return -self.negf(x)

        
        
    
def maximizeEI(model, bounds, useCDIRECT=True, xi=0.01, maxiter=50, maxtime=30, maxsample=10000, **kwargs):
    """
    Try to maximize EI in the most efficient way possible.  The methods
    will be attempted in this order:
    
        1.  If there is a C implementation of the EI objective for the
            kernel, it will be used (currently only works for Gaussian
            kernel with ARD).
            
        2.  If the C implementation of DIRECT is available, it will be used.
        
        3.  Otherwise, the Python implementation of DIRECT will be used.
    """
    if not useCDIRECT:
        print 'using DIRECT'
        ei = EI(model, xi, **kwargs)
        opt, optx = direct(ei.negf, bounds, maxiter=maxiter, maxtime=maxtime, maxsample=maxsample, **kwargs)
        return -opt, optx
    
    if isinstance(model, GaussianProcess):
        return cdirectGP(model, bounds, maxiter, maxtime, maxsample, acqfunc='ei', xi=xi, **kwargs)
    
    elif isinstance(model, RandomForest):
        return cdirectRF(model, bounds, maxiter, maxtime, maxsample, acqfunc='ei', xi=xi, **kwargs)



def cdirectRF(model, bounds, maxiter, maxtime, maxsample, acqfunc=None, xi=-1, beta=-1, scale=-1, delta=-1, **kwargs):
    
    class NODE(Structure):
        _fields_ = [("feature", c_int),
                    ("value", c_double),
                    ("label", c_double),
                    ("leftChild", c_int),
                    ("rightChild", c_int),
                    ("ndata", c_int),
                    ("dataind", POINTER(c_int))]
    
    nodes = []
    x2ind = dict((tuple(x), i) for i, x in enumerate(model.X))
    
    def addNodeToList(node):
        nid = len(nodes)    # index of this node
        if node.label is None:
            lid = addNodeToList(node.leftChild)
            rid = addNodeToList(node.rightChild)
            nodes.append(NODE(node.feature, node.value, 0.0, lid, rid, 0, pointer(c_int(0))))
        else:
            data = array([x2ind[tuple(x)] for x in node.X])
            DATAARRAY = c_int * len(data)
            dataarray = DATAARRAY()
            for i in xrange(len(data)):
                dataarray[i] = data[i]
            nodes.append(NODE(0, 0, node.label, -1, -1, len(data), dataarray))
            
        return len(nodes) + len(model.forest) -1
    
    roots = []
    for root in model.forest:
        addNodeToList(root)
        roots.append(nodes.pop(-1))
    
    for r in roots:
        nodes.insert(0, r)
        
    NODEARRAY = NODE * len(nodes)
    narray = NODEARRAY()
    for i in xrange(len(nodes)):
        narray[i] = nodes[i]
    
    
    if acqfunc=='ei':
        acquisition=0
        parm = xi
    elif acqfunc=='pi':
        acquisition=1
        parm = xi
    elif acqfunc=='ucb':
        acquisition=2
        t = len(model.Y)+1
        NA = len(bounds)
        parm = scale * sqrt(2.0*log(t**(NA/2+2) * pi**2 / (3.0*delta)))
    else:
        raise NotImplementedError('unknown acquisition function %s'%acqfunc)
    
    # for i, n in enumerate(nodes): 
    #     print i, ':', n.feature, n.value, n.label, n.leftChild, n.rightChild
    
    c_lower = array([b[0] for b in bounds], dtype=c_double)
    c_upper = array([b[1] for b in bounds], dtype=c_double)
    c_X = array(array(model.X).reshape(-1), dtype=c_double)
    c_Y = array(model.Y, dtype=c_double)
    
    lpath = ctypes.util.find_library('ego')
    lib = cdll[lpath]
    lib.maxRF.restype = POINTER(c_double)
    lib.maxRF.argtypes = [c_int, 
                    POINTER(c_double), 
                    POINTER(c_double),
                    c_int,
                    c_int,
                    POINTER(NODE),
                    POINTER(c_double), 
                    POINTER(c_double),
                    c_int,
                    c_int,
                    c_double,
                    c_int,
                    c_int,
                    c_int]
                    
    result = lib.maxRF(c_int(len(bounds)),
                    c_lower.ctypes.data_as(POINTER(c_double)),
                    c_upper.ctypes.data_as(POINTER(c_double)),
                    c_int(len(model.forest)),
                    c_int(len(nodes)),
                    narray,
                    c_X.ctypes.data_as(POINTER(c_double)),
                    c_Y.ctypes.data_as(POINTER(c_double)),
                    c_int(acquisition),
                    c_int(len(model.X)),
                    c_double(parm),
                    c_int(maxiter),
                    c_int(maxtime),
                    c_int(maxsample))
                    
    opt = -result[0]
    optx = array([x for x in result[1:len(bounds)+1]])

    # print 'Random Forest optimization returned ', opt, optx
    return opt, optx
    

def cdirectGP(model, bounds, maxiter, maxtime, maxsample, acqfunc=None, xi=-1, beta=-1, scale=-1, delta=-1, **kwargs):
    try:
        if acqfunc=='ei':
            acquisition=0
            parm = xi
        elif acqfunc=='pi':
            acquisition=1
            parm = xi
        elif acqfunc=='ucb':
            acquisition=2
            t = len(model.Y)+1
            NA = len(bounds)
            parm = sqrt(scale * 2.0*log(t**(NA/2+2) * pi**2 / (3.0*delta)))
        else:
            raise NotImplementedError('unknown acquisition function %s'%acqfunc)
            
        if isinstance(model.kernel, GaussianKernel_ard):
            kerneltype = 0
        elif isinstance(model.kernel, GaussianKernel_iso):
            kerneltype = 1
        elif isinstance(model.kernel, MaternKernel3):
            kerneltype = 2
        elif isinstance(model.kernel, MaternKernel5):
            print 'Matern 5'
            kerneltype = 3
        else:
            raise NotImplementedError('kernel not implemented in C++: %s'%model.kernel.__class__)

        lpath = ctypes.util.find_library('ego')
        while lpath is None:
            for lp in ['/global/home/eric/EGOcode/cpp/libs/libego.so', '/Users/eric/Dropbox/EGOcode/ego/libs/libego.so']:
                if os.path.exists(lp):
                    lpath = lp
        if lpath is None:
            print '\n[python] could not find ego library!  Did you forget to export DYLD_LIBRARY_PATH?'
        lib = cdll[lpath]
        lib.acqmaxGP.restype = POINTER(c_double)
        lib.acqmaxGP.argtypes = [c_int, 
                             POINTER(c_double), 
                             POINTER(c_double), 
                             POINTER(c_double), 
                             POINTER(c_double), 
                             POINTER(c_double),
                             c_int,
                             c_int,
                             c_int,
                             POINTER(c_double),
                             c_int,
                             POINTER(c_double),
                             POINTER(c_double),
                             c_double,
                             POINTER(c_double),
                             POINTER(c_double),
                             c_double,
                             c_double,
                             c_int, 
                             c_int, 
                             c_int]
        NX = len(model.Y)
        NA = len(bounds)
        npbases = 0 if model.prior is None else len(model.prior.means)
        pbtheta = 0 if model.prior is None else model.prior.theta
        if model.prior is None:
            c_pmeans = array([0], dtype=c_double)
            c_pbeta = array([0], dtype=c_double)
            c_pblowerb = array([0], dtype=c_double)
            c_pbwidth = array([0], dtype=c_double)
        else:
            c_pmeans = array(array(model.prior.means).reshape(-1), dtype=c_double)
            c_pbeta = array(model.prior.beta, dtype=c_double)
            c_pblowerb = array(model.prior.lowerb, dtype=c_double)
            c_pbwidth = array(model.prior.width, dtype=c_double)
            
        c_lower = array([b[0] for b in bounds], dtype=c_double)
        c_upper = array([b[1] for b in bounds], dtype=c_double)
        c_hyper = array(model.kernel.hyperparams, dtype=c_double)
        
        # TODO: use cholesky on the C++ side
        if isinstance(model, PrefGaussianProcess) and model.C is not None:
            c_invR = array(linalg.inv(model.R+linalg.inv(model.C)).reshape(-1), dtype=c_double)
        else:
            c_invR = array(linalg.inv(model.R).reshape(-1), dtype=c_double)

        c_X = array(array(model.X).reshape(-1), dtype=c_double)
        c_Y = array(model.Y, dtype=c_double)

        # print c_int(NA)
        # print c_lower.ctypes.data_as(POINTER(c_double))
        # print c_upper.ctypes.data_as(POINTER(c_double))
        # print c_invR.ctypes.data_as(POINTER(c_double))
        # print c_X.ctypes.data_as(POINTER(c_double))
        # print c_Y.ctypes.data_as(POINTER(c_double))
        # print c_int(NX)
        # print c_int(acqfunc)
        # print c_int(kerneltype)
        # print c_hyper.ctypes.data_as(POINTER(c_double))
        # print c_int(npbases)
        # print c_pmeans.ctypes.data_as(POINTER(c_double))
        # print c_pbeta.ctypes.data_as(POINTER(c_double))
        # print c_double(pbtheta)
        # print c_pblowerb.ctypes.data_as(POINTER(c_double))
        # print c_pbwidth.ctypes.data_as(POINTER(c_double))                             
        # print c_double(xi)
        # print c_double(model.noise)
        # print c_int(maxiter)
        # print c_int(maxtime)
        # print c_int(maxsample)

        # print '[python] calling C++ %s (%d) with X.shape = %s' % (acqfunc, acquisition, model.X.shape)
        result = lib.acqmaxGP(c_int(NA),
                            c_lower.ctypes.data_as(POINTER(c_double)),
                            c_upper.ctypes.data_as(POINTER(c_double)),
                            c_invR.ctypes.data_as(POINTER(c_double)),
                            c_X.ctypes.data_as(POINTER(c_double)),
                            c_Y.ctypes.data_as(POINTER(c_double)),
                            c_int(NX),
                            c_int(acquisition),
                            c_int(kerneltype),
                            c_hyper.ctypes.data_as(POINTER(c_double)),
                            c_int(npbases),
                            c_pmeans.ctypes.data_as(POINTER(c_double)),
                            c_pbeta.ctypes.data_as(POINTER(c_double)),
                            c_double(pbtheta),
                            c_pblowerb.ctypes.data_as(POINTER(c_double)),
                            c_pbwidth.ctypes.data_as(POINTER(c_double)),                                
                            c_double(parm),
                            c_double(model.noise),
                            c_int(maxiter),
                            c_int(maxtime),
                            c_int(maxsample))
        # print '[python] result =', result.__class__
        # print '[python] result =', result[0]
        
        opt = -result[0]
        optx = array([x for x in result[1:NA+1]])
        
        # free the pointer
        libc = CDLL(ctypes.util.find_library('libc'))
        libc.free.argtypes = [c_void_p] 
        libc.free.restype = None
        libc.free(result)
        
    except:
        try:
            print '[python] C++ MaxEI implementation unavailable, attempting C++ DIRECT on Python objective function.'
            opt, optx = cdirect(ei.negf, bounds, maxiter=maxiter, maxtime=maxtime, maxsample=maxsample, **kwargs)
        except:
            # couldn't access cDIRECT, use Python DIRECT
            print '[python] C++ DIRECT unavailable, attempting Python DIRECT'
            opt, optx = direct(ei.negf, bounds, maxiter=maxiter, maxtime=maxtime, maxsample=maxsample, **kwargs)
        opt = -opt
    
    if False:
        # do a few random searches to see if we can get a better result.  
        # mostly necessary for 1D or 2D optimizations which terminate too 
        # soon.
        ei = EI(model)
        for s in lhcSample(bounds, 500):
            if -ei.negf(s) > opt:
                opt = -ei.negf(s)
                optx = s
    return opt, optx
        
        
def test():
    
    GP = GaussianProcess(GaussianKernel_iso([.2, 1.0]))
    X = array([[.2], [.3], [.5], [1.5]])
    Y = [1, 0, 1, .75]
    GP.addData(X, Y)
    
    figure(1)
    A = arange(0, 2, 0.01)
    mu = array([GP.mu(x) for x in A])
    sig2 = array([GP.posterior(x)[1] for x in A])
    
    Ei = EI(GP)
    ei = [-Ei.negf(x) for x in A]
    
    Pi = PI(GP)
    pi = [-Pi.negf(x) for x in A]
    
    Ucb = UCB(GP, 1, T=2)
    ucb = [-Ucb.negf(x) for x in A]
    
    ax = subplot(1, 1, 1)
    ax.plot(A, mu, 'k-', lw=2)
    xv, yv = poly_between(A, mu-sig2, mu+sig2)
    ax.fill(xv, yv, color="#CCCCCC")
    
    ax.plot(A, ei, 'g-', lw=2, label='EI')
    ax.plot(A, ucb, 'g--', lw=2, label='UCB')
    ax.plot(A, pi, 'g:', lw=2, label='PI')
    ax.plot(X, Y, 'ro')
    ax.legend()
    draw()
    show()
    
    
    
if __name__=="__main__":
    test()
    