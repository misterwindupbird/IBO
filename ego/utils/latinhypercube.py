#!/usr/bin/python

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

from __future__ import division
from numpy import array, arange, vstack
from numpy.random import RandomState

def lhcSample(bounds, N, seed=None):
    """
    Perform latin hypercube sampling.
    
    @param bounds:  sequence of [min, max] bounds for the space
    @param N:       number of samples
    
    @return: list of samples points (represented as arrays)
    """
    rs = RandomState(seed)
    samp = []
    for bmin, bmax in bounds:
        if bmin==bmax:
            dsamp = array([bmin]*N)
        else:
            dsamp = (bmax-bmin) * rs.rand(N) / N + arange(bmin, bmax, (bmax-bmin)/N)
        rs.shuffle(dsamp)
        samp.append(dsamp)
    
    return list(vstack(samp).T)
        