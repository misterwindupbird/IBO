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

#include <iostream>
#include <cmath>

#include "helpers.h"

using namespace std;

extern "C" const double logCDFs(int nprefinds, int* prefinds, double* x)
{
    double lcdf = 0.;
    double Z = sqrt(2);
    
    // cout << "[C] erf(0) = "  << erf(0) << endl;
    // cout << "[C] erf(0.5) = "  << erf(0.5) << endl;
    // cout << "[C] erf(1) = "  << erf(1.0) << endl;
    // for (int i = 0; i < nprefinds; i++)
    // {
    //     cout << "[C] " << prefinds[i] << ", " << x[prefinds[i]] << endl;
    // }
    // cout << endl;
    
    for (int i = 0; i < nprefinds; i++)
    {
        double q = 0.5 * (1 + erf((x[prefinds[i]]-x[prefinds[i+1]]) / Z));
        // cout << "[C] CDF of " << (x[prefinds[i]]-x[prefinds[i+1]]) / Z << " = " << q << endl;
        i++;
        if (q/Z != 0.0)
        {
            lcdf += log(q/Z);
        }
    }
    // cout << "[C] lcdf = " << lcdf << endl;
    return lcdf;
}