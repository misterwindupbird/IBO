#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sstream>

#include "direct.h"
// functions to optimize GPs

using namespace std;

class GP_Maximizer
{
public:
    static double* lb;
    static double* ub;
    static double* invR;
    static int NA;
    static int NX;
    static double* X;
    static double* Y;
    static int acqfunc;
    static int kerneltype;
    static double* hyperparams;
    static int npbases;
    static double* pbasismeans;
    static double* pbasisbeta;
    static double pbasistheta;
    static double* pbasislowerb;
    static double* pbasiswidth;
    static double sf2;
    static double noise;
    static double maxY;
    static double parm;
    
    static void posterior(int ndim, double* x, double &mu, double &sigma)
    {
        // compute r, the distance from x to each datum
        // cout << "x = ";
        // for (uint i = 0; i < ndim; i++)
        // {
        //     cout << x[i] << " ";
        // }
        // cout << endl;
        double* r = (double*)malloc(NX*sizeof(double));
        for (int i = 0; i < NX; i++)
        {
            double z = 0;
            switch(kerneltype)
            {
            case 0:
                // Gaussian ARD
                for (int j = 0; j < NA; j++)
                {
                    z += 1/pow(hyperparams[j],2) * pow(X[NA*i+j]-x[j], 2);
                }
                // z = sqrt(z);
                r[i] = sf2 * exp(-.5 * z);
                break;
            case 1:
                // Gaussian ISO
                for (int j = 0; j < NA; j++)
                {
                    z += std::pow((X[NA*i+j]-x[j])/hyperparams[0], 2);
                }
                // z = sqrt(z);
                r[i] = sf2 * exp(-.5 * z);
                break;
            case 2:
                // Matern 3.   z = self.sf2 * (1 + self.sqrt3*r/self.theta) * exp(-self.sqrt3*r/self.theta)
                for (int j = 0; j < NA; j++)
                {
                    z += std::pow((X[NA*i+j]-x[j])/hyperparams[0], 2);
                }
                z = sqrt(3)*sqrt(z);
                r[i] = sf2 * (1.0 + z) * exp(-z);
                break;
            case 3:
                // Matern 5
                
                // self.sf2 * exp(-sqrt(z)) * (1.0 + sqrt(z) + z/3.0)
                for (int j = 0; j < NA; j++)
                {
                    z += std::pow(X[NA*i+j]-x[j], 2);
                }
                z = sqrt(z);
                r[i] = sf2 * (1.0 + sqrt(5)*z / hyperparams[0] + 5*z*z/(3*hyperparams[0]*hyperparams[0])) * exp(-(sqrt(5)*z/hyperparams[0]));
                cout << r[i] << " ";
                break;
                
            }
        }
        
        double ypred;
        if (npbases > 0)
        {
            double mu = 0.0;
            for (int i = 0; i < npbases; i++)
            {
                double d = 0;
                for (int j = 0; j < NA; j++) 
                {
                    d += pow((x[j]-pbasislowerb[j])/pbasiswidth[j] - pbasismeans[i*NA+j], 2);
                    // cout << pbasismeans[i*NA+j] << " ";
                    // cout << (ub[j]-lb[j])*(x[j]-lb[j]) << "-" << pbasismeans[i*NA+j] << " ";
                }
                // cout << endl;
                // cout << "d = " << d << endl;
                // cout << "beta = " << pbasisbeta[i] << endl;
                // cout << "theta = " << pbasistheta << endl;
                mu += pbasisbeta[i] * exp(-pbasistheta * d);
            }
            // cout << "mu = " << mu << endl;
            double* ymu = (double*)malloc(NX*sizeof(double));
            for (int i = 0; i < NX; i++) ymu[i] = Y[i]-mu;
            ypred = mu + aMb(NX, r, invR, ymu);
            free(ymu);
        }
        else
        {
            // prediction is r * invR * y
            ypred = aMb(NX, r, invR, Y);
        }
        
        // cout << "ypred = " << ypred << endl;
        
        // variance is 1+reg - r * invR * r
        double sig2 = 1. + noise - aMb(NX, r, invR, r);
        if (sig2 < 1e-8)
        {
            sig2 = 1e-8;
        }
        else if (sig2 > 10.)
        {
            sig2 = 10.;
        }
        // cout << "sig2 = " << sig2 << endl;
        free (r);
        
        // and now, the EI
        // ydiff = ypred - double(max(self.Y)*.9)
        // ynorm = double(ydiff / s)
        // 
        // EI = self.alpha * (ydiff * CDF(ynorm)) + (1-self.alpha) * (s * PDF(ynorm))
        sigma = sqrt(sig2);
        mu = ypred;
        
        return;
    }
    

    // multiply a * M * b.T
    static double aMb(int& N, double* a, double* M, double* b)
    {
        double* Mb = (double*)malloc(N*sizeof(double));
        for (int i = 0; i < N; i++)
        {
            Mb[i] = 0;
            for (int j = 0; j < N; j++)
            {
                Mb[i] += M[i*N+j] * b[j];
            }
        }
        
        double x = 0;
        for (int i = 0; i < N; i++) x += Mb[i] * a[i];
        free(Mb);
        
        return x;
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
        // cout << parm << " ";


        // cout << "ydiff = " << ydiff << endl;
        // cout << "ynorm = " << ynorm << endl;
        // cout << "cdf = " << cdf << endl;
        // cout << "pdf = " << pdf << endl;
        // 
        // cout << "EI = " << EI << endl;
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
        return -(mu + parm * sigma);
    }
    
};

double* GP_Maximizer::lb = NULL;
double* GP_Maximizer::ub = NULL;
double* GP_Maximizer::invR = NULL;
int GP_Maximizer::NA = 0;
int GP_Maximizer::NX = 0;
double* GP_Maximizer::X = NULL;
double* GP_Maximizer::Y = NULL;
int GP_Maximizer::kerneltype = 0;
int GP_Maximizer::acqfunc = 0;
double* GP_Maximizer::hyperparams = NULL;
int GP_Maximizer::npbases = 0;
double* GP_Maximizer::pbasismeans = NULL;
double* GP_Maximizer::pbasisbeta = NULL;
double GP_Maximizer::pbasistheta = 0;
double* GP_Maximizer::pbasislowerb = NULL;
double* GP_Maximizer::pbasiswidth = NULL;
double GP_Maximizer::sf2 = 0;
double GP_Maximizer::noise = 0.0001;
double GP_Maximizer::parm = 0.0;
double GP_Maximizer::maxY = MIN_DOUBLE;


extern "C" const double* 
acqmaxGP(int ndim, 
                     double* lb, 
                     double* ub, 
                     double* invR, 
                     double* X, 
                     double* Y,
                     int nx,
                     int acqfunc,
                     int kerneltype,
                     double* hyperparams,
                     int npbases,
                     double* pbasismeans,
                     double* pbasisbeta,
                     double pbasistheta,
                     double* pbasislowerb,
                     double* pbasiswidth,
                     double parm,
                     double noise,
                     int maxiter, 
                     int maxtime, 
                     int maxsample)
{
    GP_Maximizer::lb = lb;
    GP_Maximizer::ub = ub;    
    GP_Maximizer::invR = invR;
    GP_Maximizer::NA = ndim;
    GP_Maximizer::NX = nx;
    GP_Maximizer::X = X;
    GP_Maximizer::Y = Y;
    GP_Maximizer::acqfunc = acqfunc;
    GP_Maximizer::parm = parm;
    GP_Maximizer::noise = noise;
    GP_Maximizer::kerneltype = kerneltype;
    GP_Maximizer::hyperparams = hyperparams;
    GP_Maximizer::npbases = npbases;
    GP_Maximizer::pbasismeans = pbasismeans;
    GP_Maximizer::pbasisbeta = pbasisbeta;
    GP_Maximizer::pbasistheta = pbasistheta;
    GP_Maximizer::pbasislowerb = pbasislowerb;
    GP_Maximizer::pbasiswidth = pbasiswidth;
    switch (kerneltype)
    {
    case 0:
    case 1:
    case 2:
        // non-SV kernel
        GP_Maximizer::sf2 = 1.0;
        break;
    default:
        // to be used if & when we add SV kernels
        GP_Maximizer::sf2 = exp(2.0 * log(hyperparams[ndim]));
        break;
    
    }
    GP_Maximizer::maxY = Y[0];
    
    for (int i = 0; i < nx; i++) 
    {
        if (Y[i] > GP_Maximizer::maxY) GP_Maximizer::maxY = Y[i];
    }
    // double* x = (double*)malloc(sizeof(double)*2);
    // x[0] = 1.0;
    // x[1] = 0.0;
    // cout << "[C++] negei(1, 0) =" << GP_Maximizer::negei(2, x) << endl;
    const double* dout;
    switch (acqfunc)
    {
    case 0:
        // cout << "[C++] maximize EI" << endl;
        dout = direct(&(GP_Maximizer::negei), ndim, lb, ub, maxiter, maxtime, maxsample);
        break;
    case 1:
        // cout << "[C++] maximize PI" << endl;
        dout = direct(&(GP_Maximizer::negpi), ndim, lb, ub, maxiter, maxtime, maxsample);
        break;
    case 2:
        // cout << "[C++] maximize UCB" << endl;
        dout = direct(&(GP_Maximizer::negucb), ndim, lb, ub, maxiter, maxtime, maxsample);
        break;
    default:
        cout << "[C++] unknown acquisition function" << endl;
        dout = NULL;
        break;
    }
    // cout << "[optimizeGP] dout = " << dout << endl;
    return dout;
}

extern "C" const double * testreturn(double* z, const int nz, double* res)
{
    double* d = (double*)malloc(sizeof(double)*nz);
    
    ostringstream s;
    cout << "... " << z[0] << endl;
    for (int i = 0; i < nz; i++)
    {
        s << z[i] << " ";
        cout << z[i] << endl;
        d[i] = z[i];
    }
    cout << s.str().c_str() << endl;
    // res = d;
    return d;
}