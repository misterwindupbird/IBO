#/usr/bin/python
"""
DIRECT optimization for Python.  direct.demo() is an example of how to use it,
but in short, it works like this:

    import direct
    def foo(x):
        # function to be optimized: y = f(x)
        return y
    
    # set bounds of 3D space to optimize over
    bounds = [(0.0, 10.0), (0.5, 1.0), (-1000.0, 1000.0)]
    
    # run 10 iterations of DIRECT on foo
    optimum, report = direct(foo, bounds, maxiter=10)
    
    # optimal location and value
    print 'Optimum at %s. Value = %f.' % (optimum[1], optimum[0])
    optValue, optLocation = optimum
"""
from __future__ import division

from copy import copy
from time import time
import ctypes
import ctypes.util
from ctypes import c_int, c_double, POINTER, cdll, c_char_p, CFUNCTYPE

import matplotlib.pyplot as plt
from numpy import sum, array, sin, double
from numpy.random import random_sample

FMIN = []
SAMPLES = 0


class Rectangle(object):
    """docstring for Rectangle"""
    def __init__(self, lb, ub, y):
        self.lb = copy(lb)
        self.ub = copy(ub)
        self.y = y
        self.center = [l+(u-l)/2. for u, l in zip(self.lb, self.ub)]
        self.d = sum([(l-c)**2. for l, c in zip(self.lb, self.center)])**0.5
        
        
def direct(f, bounds, args=None, debug=False, maxiter=None, maxsample=None, maxtime=None):
    """ 
    Do DIRECT optimization of function f.
    
    bounds is the range over which to optimize.  It must be a sequence of (min, max) tuples.  The length of range must be the dimensionality of the function f.
    
    f must be formed such that y = f(x, args) where x is the sample point and y is the scalar response to be optimized.
    
    The optimization does not (currently) terminate based on performance, so
    external criteria must be set.  Termination will occur when the first of
    the follow apply, if set:
    
        args:       Parameters passed into f s.t. y = f(x, args).
        
        maxiter     The number of DIRECT iterations (identifying and dividing
                    possibly optimal rectangles).
                    
        maxsample   The number of calls to the target function.
        
        maxtime     (Real) elapsed clock time, in seconds.
        
    Note that the later two *will* interrupt the POR division and re-evaluate
    fmin based on all the rectangles it was able to get to.  This can result in 
    non-deterministic behaviour.
    
    Returns:
    
        optimum     (value, location) tuple of the optimum
    """
    if not (maxiter or maxsample or maxtime):
        raise ValueError("No termination criterion set!")
    
    global SAMPLES, FMIN
    FMIN = []
    SAMPLES = 0
    
    if args is None:
        args = []
    
    tic = time()
    fminevol = []

    def samplef(x):
        """
        Translate sample point from unit cube to actual domain and sample
        function and get value from function.
        """
        global FMIN, SAMPLES

        xprime = [z*(b[1]-b[0])+b[0] for z, b in zip(x, bounds)]
        y = f(xprime, *args)
        
        # print 'sampled x  =', x
        # print "        x' =", xprime
            
        SAMPLES += 1
        
        if len(FMIN) == 0 or y < FMIN[0]:
            FMIN = [y, x]
        return y
    
    def divrec(rect):
        """
        Divide a rectangle according to DIRECT algorithm.
        """
        global FMIN
        rectangles.remove(rect)
        # split rectangle along longest side(s)
        maxlength = max([u-l for u, l in zip(rect.ub, rect.lb)])
        
        if (debug): print '***** chop rectangle centered at', rect.center
        I = []
        for i in xrange(N):
            if rect.ub[i] - rect.lb[i] == maxlength:
                s1 = copy(rect.center)
                s2 = copy(rect.center)
                iwidth = rect.ub[i]-rect.lb[i]
                s1[i] = rect.lb[i] + iwidth / 3.
                s2[i] = rect.lb[i] + 2. * iwidth / 3.

                I.append((i, min(samplef(s1), samplef(s2))))
        I.sort(key=lambda x:x[1])
        if (debug): print '      (along %d axes): %s' % (len(I), I)
        
        oldrect = rect
        for i, _ in I:
            # for each long dimension...        
            dwidth = oldrect.ub[i] - oldrect.lb[i]
            split1 = oldrect.lb[i] + dwidth * (1/3)
            split2 = oldrect.lb[i] + dwidth * (2/3)
            # if jitter:
            #     split1 += random.uniform(-dwidth*.02, dwidth*.02)
            #     split2 += random.uniform(-dwidth*.02, dwidth*.02)

            lb1 = copy(oldrect.lb)
            ub1 = copy(oldrect.ub)
            ub1[i] = split1
            rectangles.add(Rectangle(lb1, ub1, samplef([l+(u-l)/2. for u, l in zip(lb1, ub1)])))

            lb2 = copy(oldrect.lb)
            ub2 = copy(oldrect.ub)
            lb2[i] = split1
            ub2[i] = split2
            target = Rectangle(lb2, ub2, oldrect.y)
            
            lb3 = copy(oldrect.lb)
            ub3 = copy(oldrect.ub)
            lb3[i] = split2
            rectangles.add(Rectangle(lb3, ub3, samplef([l+(u-l)/2. for u, l in zip(lb3, ub3)])))
            
            oldrect = target
        rectangles.add(target)
        
    def results():
        """
        Translate results to the original space and return results to caller.
        """
        if debug: report = {}
        lbounds = array([x[0] for x in bounds], dtype=float)
        ubounds = array([x[1] for x in bounds], dtype=float)
        def trans(x):
            """
            Translate from unit space (which DIRECT works in) to the space of
            the calling function.
            """
            return x * (ubounds-lbounds) + lbounds
            
        if debug: report['fmin evolution'] = [(y, trans(x)) for y, x in fminevol]
        if debug: report['rectangles'] = [Rectangle(trans(r.lb), trans(r.ub), r.y) for r in rectangles]
        
        optimum = (FMIN[0], trans(FMIN[1]))
        
        if debug: 
            return optimum, report
        return optimum
        
    
    # INITIALIZATION    
    N = len(bounds) # dimensionality
    rectangles = set()
    first = Rectangle([0.]*N, [1.]*N, samplef([.5]*N))
    rectangles.add(first)
    divrec(first)
    
    # okay, let's get choppy!
    iteration = 0
    epsilon = 10e-10
    while True:
        # print '[direct] iteration %d, samples = %d' % (iteration, SAMPLES)
        iteration += 1
        if maxiter and iteration > maxiter:
            if (debug): print 'Reached maximum iterations.'
            return results()
            
        # find potentially optimal rectangles
        potopts = []
        for Rj in rectangles:
            maxI1 = None
            minI2 = None
            for Ri in rectangles:
                if Ri == Rj: continue
                if Ri.d < Rj.d:
                    # I1
                    val = (Rj.y - Ri.y) / (Rj.d - Ri.d)
                    if maxI1 is None or val > maxI1:
                        maxI1 = val
                elif Ri.d > Rj.d:
                    # I2
                    val = (Ri.y - Rj.y) / (Ri.d - Rj.d)
                    if minI2 is None or val < minI2:
                        minI2 = val
                        if minI2 <= 0.:
                            break
                else:
                    # I3
                    if Rj.y > Ri.y:
                        break
                        
                if maxI1 is not None and minI2 is not None and minI2 < maxI1:
                    break
                    
            else:
                # last check...
                if not minI2:
                    potopts.append(Rj)
                    
                elif FMIN[0] == 0:
                    if Rj.y <= Rj.d * minI2:
                        potopts.append(Rj)
                elif epsilon <= (FMIN[0]-Rj.y)/abs(FMIN[0]) + (Rj.d/abs(FMIN[0])) * minI2:
                    
                    
                    potopts.append(Rj)
            
            if maxtime and time()-tic >= maxtime:
                # return without further dividing
                if (debug): print 'Reached maximum time.'
                return results()
            
        
        for Rj in potopts:
            divrec(Rj)
            if maxsample and SAMPLES >= maxsample:
                if (debug): print 'Reached maximum samples'
                fminevol.append(FMIN)
                return results()
            if maxtime and time()-tic>= maxtime:
                if (debug): print 'Reached maximum time'
                fminevol.append(FMIN)
                return results()
        
        if (debug): print '[%.3fs] iteration %d: %d potentially optimal rectangles divided. Total samples = %s.  FMIN = %s' % (time()-tic, iteration, len(potopts), SAMPLES, FMIN)
        fminevol.append(FMIN)


def debugRectangles(rects):
    """
    Sanity test on rectangle sets go here.
    """
    # first, are any out of bounds?
    for r in rects:
        if r.lb[0] < 0. or r.lb[1] < 0.:
            print "ERROR: rectangle has lb", r.lb
        if r.ub[0] > 1. or r.ub[1] > 1.:
            print "ERROR: rectangle has ub", r.ub
        if r.lb[0] > r.ub[0] or r.lb[1] > r.ub[1]:
            print 'ERROR: lower bound greater than upper'
        assert r.center == [l+(u-l)/2. for u, l in zip(r.lb, r.ub)]
        assert r.d == sum([(l-c)**2. for l, c in zip(r.lb, r.center)])**0.5
        
    
    # okay, now sample
    for _ in xrange(10000):
        hits = 0
        samp = random_sample(2)
        for r in rects:
            if all(samp > r.lb) and all(samp < r.ub):
                hits += 1
        if hits != 1:
            print "ERROR: location has %d hits: %s" % (hits, samp)


def cdirect(f, bounds, args=None, maxiter=10, maxtime=10, maxsample=200000, **kwargs):
    
    if args is None:
        args = []

    lpath = ctypes.util.find_library('ego')
    if lpath is None:
        lpath = '/global/home/eric/EGOcode/cpp/libs/libego.so'            
    lib = cdll[lpath]

    lib.direct.restype = POINTER(c_double)
    OBJECTIVE = CFUNCTYPE(c_double, c_int, POINTER(c_double))
    lib.direct.argtypes = [OBJECTIVE, c_int, POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
    N = len(bounds)
    
    def objective(n, x):
        X = array([x[i] for i in xrange(n)])
        y = f(X, *args)
        # print '[python] sampling', X, '=', y
        return y
        
    lower = array([b[0] for b in bounds], dtype=c_double)
    upper = array([b[1] for b in bounds], dtype=c_double)
    result = lib.direct(OBJECTIVE(objective), c_int(len(lower)), lower.ctypes.data_as(POINTER(c_double)), upper.ctypes.data_as(POINTER(c_double)), c_int(maxiter), c_int(maxtime), c_int(maxsample))
    # print '[python] direct returned "%s"' % result
    # try:
    #     result = [double(s) for s in result.strip().split()[:N+1]]
    # except ValueError, e:
    #     print e
    #     print 's = "%s"' % s
    #     for z in result:
    #         print z.decode()
    #     raise
    return result[0], array([x for x in result[1:len(bounds)+1]])



def demoCDIRECT(maxiter=25):
    """
    Test and visualize cDIRECT on a 2D function.  This will draw the contours
    of the target function, the final set of rectangles and mark the optimum
    with a red dot.
    """
    import matplotlib.pyplot as plt
    
    def foo(x):
        """
        Code for the Shekel function S5.  The dimensionality 
        of x is 2.  The  minimum is at [4.0, 4.0].
        """
        # return min(-.5, -sum(1./(dot(x-a, x-a)+c) for a, c in SHEKELPARAMS))
        return sin(x[0]*2)+abs(x[0]-15) + sin(x[1])+.2*abs(x[1]-6)
    # def foo(x):
    #     # several local minima, global minimia is at bottom left
    #     return 2.5 + sin((x[0]-.4)*8)+sin((x[1]+.5)*5) + .1* sum(sin(x[0]*50)) + .1* sum(sin(x[1]*50))+ x[0]*.1 - x[1] * .1

    bounds = [(1.2, 28.), (0.1, 13.)]
    optv, optx = cdirect(foo, bounds, maxiter=maxiter)
    print '***** opt val =', optv
    print '***** opt x   =', optx
    
    plt.figure(2)
    plt.clf()
    
    # plot rectangles
    c0 = [(i/100.)*(bounds[0][1]-bounds[0][0])+bounds[0][0] for i in xrange(101)]
    c1 = [(i/100.)*(bounds[1][1]-bounds[1][0])+bounds[1][0] for i in xrange(101)]
    z = array([[foo([i, j]) for i in c0] for j in c1])
    
    ax = plt.subplot(111)
    B = [array([1.2, 0.1]), array([28., 13.])]
    for line in open('finalrecs.txt').readlines():
        dat = line.strip().split(',')
        lb = array([double(x) for x in dat[0].split()])*(B[1]-B[0])+B[0]
        ub = array([double(x) for x in dat[1].split()])*(B[1]-B[0])+B[0]
        ax.add_artist(plt.Rectangle(lb, ub[0]-lb[0], ub[1]-lb[1], fc='y', ec='k', lw=1, alpha=0.25, fill=True))

    ax.plot(optx[0], optx[1], 'ro')
    cs = ax.contour(c0, c1, z, 10)
    ax.clabel(cs)
    plt.jet()
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_title('final optimum')

def demoDIRECT(maxiter=25):
    """
    Test and visualize DIRECT on a 2D function.  This will draw the contours
    of the target function, the final set of rectangles and mark the optimum
    with a red dot.
    """

    def foo(x):
        """
        Code for the Shekel function S5.  The dimensionality 
        of x is 2.  The  minimum is at [4.0, 4.0].
        """
        return sin(x[0]*2)+abs(x[0]-15) + sin(x[1])+.2*abs(x[1]-6)
        
    bounds = [(1.2, 28.), (0.1, 13.)]
    optimum, report = direct(foo, bounds, debug=True, maxiter=maxiter)
    
    plt.figure(1)
    plt.clf()
    
    # plot rectangles
    c0 = [(i/50.)*(bounds[0][1]-bounds[0][0])+bounds[0][0] for i in xrange(51)]
    c1 = [(i/50.)*(bounds[1][1]-bounds[1][0])+bounds[1][0] for i in xrange(51)]
    z = array([[foo([i, j]) for i in c0] for j in c1])
    
    ax = plt.subplot(111)
    for rect in report['rectangles']:
        ax.add_artist(plt.Rectangle(rect.lb, rect.ub[0]-rect.lb[0], rect.ub[1]-rect.lb[1], fc='y', ec='k', lw=1, alpha=0.25, fill=True))
        # ax.plot([x[0] for _,x in report['fmin evolution']], [x[1] for _,x in report['fmin evolution']], 'go')
        ax.plot([optimum[1][0]], [optimum[1][1]], 'ro')
        # ax.text(rect.center[0], rect.center[1], '%.3f'%rect.y)
    cs = ax.contour(c0, c1, z, 10)
    ax.clabel(cs)
    plt.jet()
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_title('final rectangles')
    
    # ax = plt.subplot(122)
    # fminevol = [y for y,_ in report['fmin evolution']]
    # ax.plot(range(len(fminevol)), fminevol, 'k-', lw=2)
    # ax.set_ylim(min(fminevol)-0.01, max(fminevol)+0.01)
    # ax.grid()
    # ax.set_title('optimization evolution')
    # ax.set_xlabel('iteration')
    # ax.set_ylabel('fmin')
    # plt.show()


if __name__ == "__main__":
    demoDIRECT()
    
    
    