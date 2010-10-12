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