/*
Copyright (C) 2010, 2011 by Eric Brochu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sstream>
#include <set>

#include "direct.h"

using namespace std;

typedef struct 
{
    int feature;
    double value;
    double label;
    int leftChild;
    int rightChild;
    int ndata;
    int* dataind;
} node_t;


class RF_Maximizer
{
public:
    static double* lb;
    static double* ub;
    
    static double* X;
    static double* Y;
    static double maxY;
    
    static int NA;
    static int NX;
    static int NT;      // number of trees
    static int NN;      // total number of nodes
    static node_t* nodes;
    
    static double parm;
    
    static void posterior(int ndim, double* x, double &mu, double &sigma)
    {
        double y = 0.0;
        set<double> ys;
        
        for (int i = 0; i < NT; i++)
        {
            int cursor = i;
            while (1)
            {
                node_t node = nodes[cursor];     // FIXME
                if (node.leftChild > -1)
                {
                    // cout << "\tis x[" << node.feature << "] < " << node.value << "?\t";
                    if (x[node.feature] < node.value)
                    {
                        // cout << "yes" << endl;
                        cursor = node.leftChild;
                    }
                    else
                    {
                        // cout << "no" << endl;
                        cursor = node.rightChild;
                    }
                }
                else
                {
                    // cout << "[C++] label " << node.label << endl;
                    // cout << "[C++] node.Y = ";
                    for (int d = 0; d < node.ndata; d++)
                    {
                        // cout << Y[node.dataind[d]] << " ";
                        ys.insert(Y[node.dataind[d]]);
                    }
                    // cout << endl;
                    y += node.label;
                    break;
                }
            }
        }
        mu = y/NT;
        
        double m = 0.0;
        for (set<double>::iterator it = ys.begin(); it != ys.end(); it++)
        {
            m += *it;
        }
        m /= ys.size();
        
        double s = 0.0;
        for (set<double>::iterator it = ys.begin(); it != ys.end(); it++)
        {
            s += pow(*it - m, 2);
        }
        sigma = sqrt(s / ys.size());
        
        // now, add a kernel factor based on distance to closest point
        if (true)
        {
            double dmin = 100000000.0;
            for (int i = 0; i < NX; i++)
            {
                double d = 0.0;
                for (int j = 0; j < NA; j++)
                {
                    d += pow(x[j]-X[i*NA+j], 2.0);
                }
                if (d < dmin) 
                {
                    dmin = d;
                }
            }
            sigma += dmin;
        }
        // if (sigma > 1.0) sigma = 1.0;
        if (sigma < 0.0) sigma = 0.0;
        return;
    }
    
    
    static double negei(int ndim, double* x)
    {
        double mu;
        double sigma;
        
        posterior(ndim, x, mu, sigma);
        double ydiff = mu - maxY - parm;
        double Z = ydiff / sigma;
        double cdf = 0.5 * (1. + erf(Z / sqrt(2.)));
        double pdf = exp(-(Z*Z / 2.)) / (sqrt(2. * M_PI));
        double EI = ydiff * cdf + sigma * pdf;
        // 
        // cout << "[C++ RF] x = ";
        // for (int i=0; i < ndim; i++) cout << x[i] << " ";
        // cout << endl;
        // cout << "         mu = " << mu << endl;
        // cout << "         sigma = " << sigma << endl;
        // cout << "         EI = " << EI << endl;
        return -EI;
    }


    static double negpi(int ndim, double* x)
    {
        double mu;
        double sigma;
        
        posterior(ndim, x, mu, sigma);
        double ydiff = mu - maxY - parm;
        double Z = ydiff / sigma;
        double cdf = 0.5 * (1. + erf(Z / sqrt(2.)));
        return -cdf;
    }


    static double negucb(int ndim, double* x)
    {
        double mu;
        double sigma;
        
        posterior(ndim, x, mu, sigma);
        // cout << "[C++ RF] x = ";
        // for (int i=0; i < ndim; i++) cout << x[i] << " ";
        // cout << endl;
        // cout << "         mu = " << mu << endl;
        // cout << "         sigma = " << sigma << endl;
        // cout << "         UCB = " << (mu + parm * sigma) << endl;
        return -(mu + parm * sigma);
    }
};


int RF_Maximizer::NA = 0;
double* RF_Maximizer::lb = NULL;
double* RF_Maximizer::ub = NULL;
int RF_Maximizer::NT = 0;
int RF_Maximizer::NN = 0;
int RF_Maximizer::NX = 0;
node_t* RF_Maximizer::nodes = NULL;
double* RF_Maximizer::X = NULL;
double* RF_Maximizer::Y = NULL;
double RF_Maximizer::parm = 0.1;
double RF_Maximizer::maxY = MIN_DOUBLE;


extern "C" const double*
maxRF(int ndim,
        double* lb,
        double* ub,
        int ntrees,
        int nnodes,
        node_t* nodes,
        double* X,
        double* Y,
        int acqfunc,
        int nx,
        double parm,
        int maxiter, 
        int maxtime, 
        int maxsample)
{
    RF_Maximizer::NA = ndim;
    RF_Maximizer::lb = lb;
    RF_Maximizer::ub = ub;
    RF_Maximizer::NN = nnodes;
    RF_Maximizer::NT = ntrees;
    RF_Maximizer::NX = nx;
    RF_Maximizer::nodes = nodes;
    RF_Maximizer::X = X;
    RF_Maximizer::Y = Y;
    RF_Maximizer::parm = parm;    


    RF_Maximizer::maxY = Y[0];
    for (int i = 0; i < RF_Maximizer::NX; i++) 
    {
        if (Y[i] > RF_Maximizer::maxY) RF_Maximizer::maxY = Y[i];
    }
    
    const double* dout;
    switch (acqfunc)
    {
    case 0:
        dout = direct(&(RF_Maximizer::negei), ndim, lb, ub, maxiter, maxtime, maxsample);
        break;
    case 1:
        dout = direct(&(RF_Maximizer::negpi), ndim, lb, ub, maxiter, maxtime, maxsample);
        break;
    case 2:
        // cout << "[C++] maximize UCB" << endl;
        dout = direct(&(RF_Maximizer::negucb), ndim, lb, ub, maxiter, maxtime, maxsample);
        break;
    default:
        dout = NULL;
        break;
    }
    return dout;
}
