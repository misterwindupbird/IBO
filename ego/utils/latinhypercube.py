#!/usr/bin/python
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
        