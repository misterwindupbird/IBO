#!/usr/bin/env python
# encoding: utf-8

# Copyright (C) 2009, 2010, 2011 by Eric Brochu
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
prefutil.py

Created by Eric on 2009-12-28.

Helper functions for the preferences.
"""
from __future__ import division

import pdb

from numpy import argmax


def query2prefs(query, f, bestDegree=0):
    """
    Given a sequence of query points and a function to evaluate, return a set 
    of preferences.
    """
    try:
        # for now, find the biggest gap and use that as the preference point
        D = zip([f(q) for q in query], query)
        D.sort(key=lambda x:x[0])
    
        bpoint = argmax([D[i+1][0]-D[i][0] for i in xrange(len(query)-1)]) + 1
    
        # everything on the right of the break point is preferred to 
        # everything on the left
        prefs = []
        for i, (yu, xu) in enumerate(D[:bpoint]):
            for j, (yv, xv) in enumerate(D[bpoint:]):
                if i==0 and j==len(D)-bpoint-1:
                    prefs.append((xv, xu, bestDegree))
                else:
                    prefs.append((xv, xu, 0))
    except e:
        print e
        pdb.set_trace()
    return prefs
    


    