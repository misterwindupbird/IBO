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
#include "direct.h"


using namespace std;

class FooClass{
    
public:
    static void setTarget(float targ) 
    {
        target = targ;
    }
    static float foo(int n, float* x)
    {
        float z = 0;
        for (uint i = 0; i < n; i++)
        {
            z += fabs(x[i]-target);
        }
        return z;
    }

    static float target;
};

float FooClass::target = 0;

int main()
{
    fvec lb = fvec(3);
    fvec ub = fvec(3);
    for (uint i = 0; i < 3; i++)
    {
        lb[i] = 0;
        ub[i] = 7;
    }
    FooClass::target = 5;
    const double* dout = direct(&(FooClass::foo), 3, &lb[0], &ub[0], 10, 10, 1000);
    
    cout << "result = " << dout << endl;
    return 0;
}