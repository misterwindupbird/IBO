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

#ifndef DIRECT_H
#define DIRECT_H

#include <vector>
#include <utility>
#include <limits>

typedef std::vector<double> fvec;
typedef unsigned int uint;
typedef std::pair<uint, double> ind_val;
typedef double(*objective_t)(int, double*);



const double MAX_DOUBLE = std::numeric_limits<double>::max();
const double MIN_DOUBLE = std::numeric_limits<double>::min();

class Direct;


class Rectangle {
public:
    Rectangle(const fvec& ln, const fvec& ub, Direct* D);
    Rectangle(const Rectangle& rec);
    ~Rectangle();
    void display() const;
    
    fvec  lb;
    fvec  ub;
    double   y;
    fvec  center;
    double   d;
};


class Direct
{
public:
    Direct(const fvec& lb, const fvec& ub, objective_t ob);
    ~Direct();
    double samplef(const fvec& x);
    std::vector<Rectangle> divrec(const Rectangle& rec);
    
    objective_t objective;
    fvec lowerb;
    fvec upperb;
    uint N;
    double FMIN;
    fvec XMIN;
    uint nsamples;
    std::vector<bool> fixed;
};

extern "C" const double* direct(objective_t objective, int ndim, double* lb, double* ub, int maxiter, int maxtime, int maxsample);

#endif
