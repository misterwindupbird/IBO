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