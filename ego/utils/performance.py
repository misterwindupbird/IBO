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
performance.py

Created by Eric on 2010-02-15.
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
