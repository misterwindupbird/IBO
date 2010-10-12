#!/usr/bin/env python
# encoding: utf-8
"""
performance.py

Created by Eric on 2010-02-15.
Copyright (c) 2010 Eric Brochu. All rights reserved.
"""

import sys
import os

from numpy import *
from scipy.special import erfinv

from ego.utils.optimize import cdirect

class MuBound(object):

    def __init__(self, GP, delta):
        super(MuBound, self).__init__()
        self.GP = GP
        self.vscale = sqrt(2) * erfinv(2*delta-1)   # probit of delta
    
    def objective(self, x):
        """
        negative value of Gdelta for a given point in the GP domain
        """
        mu, sig2 = self.GP.posterior(x)
        return -(mu + sig2 * self.vscale)

        
def Gdelta(GP, testfunc, firstY, delta=0.01, maxiter=10, **kwargs):
    """
    given a GP, find the max and argmax of G_delta, the confidence-bounded
    prediction of the max of the response surface
    """
    assert testfunc.maximize
    mb = MuBound(GP, delta)
    _, optx = cdirect(mb.objective, testfunc.bounds, maxiter=maxiter, **kwargs)
    opt = max(testfunc.f(optx), firstY)
    
    return (opt-firstY) / (-testfunc.minimum-firstY)
