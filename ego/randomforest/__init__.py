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

from ego.gaussianprocess import CDF, PDF

from numpy import *
from matplotlib.pylab import *
import numpy.random as numpyrandom
from random import sample, random
from time import time
import pickle


class Node(object):

    def __init__(self, X, Y, feature=None, value=None, m=None, label=None):
        
        self.X = X
        self.Y = Y
        self.feature = feature
        self.value = value
        self.label = label
        self.m = m
        self.leftChild = None
        self.rightChild = None
        
    def __repr__(self):
        s = 'label = ' 
        s += 'None' if self.label is None else str(self.label)
        s += '\nfeature = ' 
        s += 'None' if self.feature is None else str(self.feature)
        s += '\nvalue = '
        s += 'None' if self.value is None else str(self.value)
        return s
        
    def isTerminal(self):
        return not self.leftChild
        
        
class RandomForest(object):

    def __init__(self, ntrees=20, m=2, ndata=3, pRetrain=0.1, width=1):
        
        self.ntrees = ntrees
        self.m = m
        self.ndata = ndata
        self.pRetrain = pRetrain
        self.forest = []
        self.name = 'RF'    # for legend
        self.doBagging = False
        self.X = None
        self.Y = None
        self.maxY = None
        
        self.width = width
        
    
    def addData(self, X, Y):
        
        if isscalar(Y):
            X = [X]
            Y = [Y]
        X = array(X)
        Y = array(Y)
        
        if self.X is None:
            # create forest
            self.X = X
            self.Y = Y
            self.maxY = max(Y)
            
            for x in xrange(self.ntrees):

                if self.doBagging:
                    thisX = []
                    thisY = []
                    for _ in xrange(len(X)):
                        s = randint(0, len(X))
                        thisX.append(X[s])
                        thisY.append(Y[s])
                else:
                    thisX = self.X
                    thisY = self.Y

                tree = self._trainNode(thisX, thisY, 0, '[%d] root'%x)
                self.forest.append(tree)
        
        else:
            # add to forest
            self.X = r_[self.X, X]
            self.Y = r_[self.Y, Y]
            for x, y in zip(X, Y):
                self._addDatum(x, y)

        
    def posterior(self, x):
        """
        compute the empirical mean and variance of x (that is, for all the data 
        in all the leaves x is assigned to, compute mean and variance)
        """
        values = set()
        data = []
        labels = []
        for tree in self.forest:
            node = tree
            while node.label is None:
                node = node.leftChild if x[node.feature] <= node.value else node.rightChild
                
            values.update(node.Y)
            data.extend(node.X)
            labels.append(node.label)
        
        # penalty = None
        # for dx in data:
        #     p = clip(norm((x-dx)/self.width), 0.0, 1.0)
        #     if penalty is None or p < penalty:
        #         penalty = p**len(x)
        
        mu = mean(labels)
        values = list(values)
        sig2 = var(values)
        print mu, sig2, p
        assert isscalar(mu)
        assert isscalar(sig2)
        # assert isscalar(p)
        return mu, sig2
    
    def mu(self, x):
        return self.posterior(x)[0]
        

    def sigma2(self, x):
        return self.posterior(x)[1]

        
    def _addDatum(self, x, y):
        """
        Add a datum to the forest, updating and splitting nodes as needed.
        """
        if y > self.maxY: self.maxY = y
        for t, tree in enumerate(self.forest):
            if self.doBagging and random() < 1./e: continue
            
            # with some probability, retrain instead of updating
            if random() <= self.pRetrain:
                xtrain = r_[tree.X[:], array(x, copy=False, ndmin=2)]
                ytrain = r_[tree.Y[:], y]
                newtree = self._trainNode(xtrain, ytrain, 0, 'added')
                tree.label = newtree.label
                tree.feature = newtree.feature
                tree.value = newtree.value
                tree.leftChild = newtree.leftChild
                tree.rightChild = newtree.rightChild
                # print '\t[%d] retrained' % t
                # checkTree(tree)
                continue
            
            node = tree
            while True:
                node.X.append(x)
                node.Y.append(y)
                # print 'added',
                
                if node.isTerminal():
                    break
                    
                node = node.leftChild if x[node.feature] < node.value else node.rightChild
            # print node
            # got the leaf, split if necessary
            xtrain = node.X[:]
            ytrain = node.Y[:]
            nnode = self._trainNode(xtrain, ytrain, 0, '')
            node.label = nnode.label
            node.feature = nnode.feature
            node.value = nnode.value
            node.leftChild = nnode.leftChild
            node.rightChild = nnode.rightChild
            

    def _trainNode(self, X, Y, depth, nodestring='root'):
    
        # print nodestring, 'being trained with', X
        # terminal node?
        if len(Y) <= self.ndata:
            # print 'Y = %s, which is len %d' % (Y, len(Y))
            label = sum(Y) / len(Y)
            return Node(X, Y, label=label)
        if all([X[0]==x for x in X[1:]]):
            label = sum(Y) / len(Y)
            return Node(X, Y, label=label)

        # bestImpurity = 1.
        bestDeviance = 10e10
        bestFeature = -1
        bestValue = -1.
        NX = len(X)
        NA = len(X[0])
        
        for _ in xrange(self.m):
            feature = numpyrandom.randint(NA)
            values = sorted(list(set(x[feature] for x in X)))
        
            if len(values) < 2:
                val = values[0]
            else:
                i = randint(len(values)-1)
                val = (values[i+1] - values[i]) * random() + values[i]

            group0 = []
            group1 = []
            for x, y in zip(X, Y):
                if x[feature] <= val:
                    group0.append(y)
                else:
                    group1.append(y)
                
            # now, we want to minimize the sum of squared deviances
            # TODO:  really?  is this the best thing to do?
        
            if len(group0)==0 or len(group1)==0:
                # worst-case: a node that doesn't actually split
                # TODO: this node should not be added
                deviance = 10e10 -1.
            else:
                deviance = 0.
                for group in (group0, group1):
                    mean = sum(group) / len(group)
                    # deviance -= log(len(group), 2)
                    for y in group:
                        deviance += (mean-y)**2
                    # print '\tdeviance =', deviance
        
            if deviance < bestDeviance:
                bestFeature = feature
                bestValue = val
                bestDeviance = deviance
        
        if bestDeviance > 10e9:
            # print '%s: unable to find deviance better than %f' % (nodestring, bestDeviance)
            label = sum(Y) / len(Y)
            return Node(X, Y, label=label)
    
        # okay, couldn't find perfect node, so divide up the data and send it along
        node = Node(X, Y, bestFeature, bestValue, self.m)
        leftX = []
        leftY = []
        rightX = []
        rightY = []
        for x, y in zip(X, Y):
            if x[bestFeature] <= bestValue:
                leftX.append(x)
                leftY.append(y)
            else:
                rightX.append(x)
                rightY.append(y)
        
        # print nodestring
        # print '\tbest feature =', bestFeature
        # print '\tbest value =', bestValue
        # print '\tsend %d data left, %d right' % (len(leftX), len(rightY))
        node.leftChild = self._trainNode(leftX, leftY, depth+1, nodestring+'->left')
        node.rightChild = self._trainNode(rightX, rightY, depth+1, nodestring+'->right')
    
        return node
        
        
def checkTree(tree):
    """
    sanity test
    """
    def checkNode(node, s):
        if node.isTerminal():
            assert node.label == sum(node.Y)/len(node.Y)
            print '%s: label = %.3f'%(s, node.label)
        else:
            checkNode(node.leftChild, s+', F%d < %.2f '% (node.feature, node.value))
            checkNode(node.rightChild, s+', F%d > %.2f '% (node.feature, node.value))
    
    checkNode(tree, 'ROOT')

        

        