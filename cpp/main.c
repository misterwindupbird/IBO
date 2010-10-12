#include <cstdlib>
#include <cmath>
#include <iostream>

extern "C" const char* 
maxEI(int ndim, 
         double* lb, 
         double* ub, 
         double* invR, 
         double* X, 
         double* Y,
         int nx,
         int kerneltype,
         double* hyperparams,
         int npbases,
         double* pbasismeans,
         double* pbasisbeta,
         double pbasistheta,
         double* pbasislowerb,
         double* pbasiswidth,
         double xi,
         double noise,
         int maxiter, 
         int maxtime, 
         int maxsample);
                     
                     
int main()
{
    int ndim = 2;
    double lb[2] = {0, 0};
    double ub[2] = {1, 1};
    double invR[9] = {-45.66176471,  34.26470588,  28.30882353, 34.26470588, -24.70588235, -21.32352941, 28.30882353, -21.32352941, -16.54411765};
    double Y[3] = {1,2,3};
    double X[6] = {.5, .6, .2, .3, .01, .3};
    int nx = 3;
    int kerneltype = 0;
    double hyperparams[2] = {.8, .1};
    int npbasis = 0;
    double* pbasismeans = NULL;
    double* pbasisbeta = NULL;
    double pbasistheta = 0;
    double* pbasislowerb = NULL;
    double* pbasisupperb = NULL;
    double xi = 0.1;
    double noise = 0.01;
    int maxiter = 1000;
    int maxtime = 10;
    int maxsample = 10000;
    
    const char* z = maxEI(ndim, lb, ub, invR, X, Y, nx, kerneltype, hyperparams, npbasis, pbasismeans, pbasisbeta, pbasistheta, pbasislowerb, pbasisupperb, xi, noise, maxiter, maxtime, maxsample);
    std::cout << z << std::endl;
    return 0;
}
