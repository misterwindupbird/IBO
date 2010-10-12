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
