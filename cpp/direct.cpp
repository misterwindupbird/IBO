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
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <fstream>
#import <cassert>
#include "direct.h"

using namespace std;

const bool DEBUG = false;


bool operator<(const ind_val& a, const ind_val& b)
{
    return a.second < b.second;
}

bool sortByVal(const ind_val& a, const ind_val& b)
{
    return a.second < b.second;
}

Rectangle::Rectangle(const fvec& lower, const fvec& upper, Direct* D)
{
    lb = fvec(lower);
    ub = fvec(upper);

    center = fvec(lower.size(), 0.0);
    d = 0.0;
    for (unsigned int i = 0; i < lower.size(); i++)
    {
        center[i] = lb[i] + (ub[i]-lb[i]) / 2.;
        d += pow((lb[i]-center[i]), 2);
    }
    d = sqrt(d);
    // if (DEBUG) cout << "***** d = " << d << endl;
    
    y = D->samplef(center);
}

Rectangle::Rectangle(const Rectangle& rec)
{
    lb = fvec(rec.lb);
    ub = fvec(rec.ub);
    center = fvec(rec.center);
    y = rec.y;
    d = rec.d;    
}

Rectangle::~Rectangle()
{
    // if (DEBUG) cout << "[cdirect] \tdestructing rectangle" << endl;
}



void Rectangle::display() const
{
    for (unsigned int i = 0; i < center.size(); i++)
    {
        cout << "\t" << lb[i] << "\t" << center[i] << "\t" << ub[i] << endl;
    }
    cout << "y = " << y << endl;
}


Direct::Direct(const fvec& lb, const fvec& ub, objective_t ob)
{
    lowerb = fvec(lb);
    upperb = fvec(ub);
    objective = ob;
    N = lowerb.size();
    FMIN = MAX_DOUBLE;
    nsamples = 0;
    for (uint i = 0; i < N; i++)
        fixed.push_back(lowerb[i]==upperb[i]);
}

Direct::~Direct()
{
    if (DEBUG) cout << "[cdirect] destructing DIRECT" << endl;
}


double Direct::samplef(const fvec& x)
{
    fvec sample(N);
    for (uint i = 0; i < N; i++)
    {
        if (fixed[i])
            sample[i] = lowerb[i];
        else
            sample[i] = x[i] * (upperb[i]-lowerb[i]) + lowerb[i];
    }
    double y = objective(sample.size(), &sample[0]);
    nsamples += 1;
    
    if (y < FMIN) 
    {
        FMIN = y;
        XMIN = fvec(x);
        for (uint i = 0; i < N; i++)
            XMIN[i] = lowerb[i] + (upperb[i]-lowerb[i]) * x[i];
        if (DEBUG) 
        {
            cout << "[cdirect] new fmin" << endl;
            cout << "\tFMIN = " << FMIN << endl;
            cout << "\tXMIN = ";
            for (uint i = 0; i < N; i++)
                cout << XMIN[i] << " ";
            cout << endl;
        }
    }
    return y;
}




vector<Rectangle> Direct::divrec(const Rectangle& rec)
{
    if (DEBUG) 
    {
        cout << "***** chop rectangle centered at [";
        for (uint i = 0; i < N; i++) cout << " " << rec.center[i];
        cout << " ]" << endl;
    }
    
    // divide the rectangle, returning vect of divided rectangles
    double maxlength = rec.ub[0]-rec.lb[0];
    
    for (uint i = 1; i < N; i++)
    {
        if (!fixed[i] && rec.ub[i]-rec.lb[i] > maxlength)
        {
            maxlength = rec.ub[i]-rec.lb[i];
        }
    }
    
    if (DEBUG) cout << "maxlength = " << maxlength << endl;
    
    vector<Rectangle> newrecs;
    vector<ind_val> I;
    for (uint i = 0; i < N; i++)
    {
        // if (DEBUG) cout << "fixed[i] = " << fixed[i] << endl;
        // if (DEBUG) cout << "rec.ub[i]-rec.lb[i] = " << rec.ub[i]-rec.lb[i] << endl;
        if (!fixed[i] && rec.ub[i]-rec.lb[i] == maxlength)
        {
            fvec s1 = fvec(rec.center);
            fvec s2 = fvec(rec.center);
            s1[i] = rec.lb[i] + maxlength / 3.;
            s2[i] = rec.lb[i] + 2. * maxlength / 3.;
            
            double sf1 = samplef(s1);
            double sf2 = samplef(s2);
            if (sf1 < sf2)
            {
                I.push_back(ind_val(i, sf1));
            }
            else
            {
                I.push_back(ind_val(i, sf2));
            }
        }
    }
    
    sort(I.begin(), I.end(), sortByVal);
    if (DEBUG) cout << "\talong " << I.size() << " axes: ";
    if (DEBUG) for (uint i = 0; i < I.size(); i++) cout << "(" << I[i].first << ", " << I[i].second << ")  ";
    if (DEBUG) cout << endl;
    
    Rectangle oldrect = Rectangle(rec);
    
    for (uint i = 0; i < I.size(); i++)
    {
        uint d = I[i].first;
        double dwidth = oldrect.ub[d] - oldrect.lb[d];
        double split1 = oldrect.lb[d] + dwidth / 3.;
        double split2 = oldrect.lb[d] + 2. * dwidth / 3.;
        
        fvec lb1 = fvec(oldrect.lb);
        fvec ub1 = fvec(oldrect.ub);
        // lb2 = fvec(oldrect.lb);
        // ub2 = fvec(oldrect.ub);
        fvec lb3 = fvec(oldrect.lb);
        fvec ub3 = fvec(oldrect.ub);

        ub1[d] = split1;
        newrecs.push_back(Rectangle(lb1, ub1, this));

        oldrect.lb[d] = split1;
        oldrect.ub[d] = split2;
        
        
        lb3[d] = split2;
        newrecs.push_back(Rectangle(lb3, ub3, this));
        
    }
    double d = 0.0;
    for (uint i = 0; i < N; i++)
    {
        d += pow((oldrect.lb[i]-oldrect.center[i]), 2);
    }
    oldrect.d = sqrt(d);
    newrecs.push_back(oldrect);

    return newrecs;
}


void writerecs(const char* fname, const vector<Rectangle>& recs)
{
    // write the rectangles out to disk, one per line, format LB, UB, f
    ofstream fout;
    fout.open(fname);
    for (uint i = 0; i < recs.size(); i++)
    {
        const Rectangle& r = recs[i];
        for (uint j = 0; j < r.lb.size(); j++) fout << r.lb[j] << " ";
        fout << ", ";
        for (uint j = 0; j < r.ub.size(); j++) fout << r.ub[j] << " ";
        fout << ", ";
        fout << r.y << endl;
    }
    fout.close();
}


// SANITY TEST
void test_coverage(const fvec& lower, const fvec& upper, const vector<Rectangle>& recs, uint nsamples)
{
    // first, test the rectangles
    uint N = lower.size();
    for (uint r = 0; r < recs.size(); r++)
    {
        for (uint k = 0; k < N; k++)
        {
            if (recs[r].center[k] != recs[r].lb[k] + (recs[r].ub[k]-recs[r].lb[k])/2)
            {
                cout << "***** rec " << r << " expected center at " << recs[r].center[k] << ", got " << recs[r].lb[k] + (recs[r].ub[k]-recs[r].lb[k])/2 << endl;
            }
            
            if (recs[r].lb[k] < 0.0)
                cout << "***** rec " << r << " invalid lower bound " << recs[r].lb[k] << "!" << endl;
            if (recs[r].ub[k] > 1.0)
                cout << "***** rec " << r << " invalid upper bound " << recs[r].ub[k] << "!" << endl;
        }
    }
    
    for (uint i = 0; i < nsamples; i++)
    {
        uint hits = 0;
        
        fvec rvec = fvec(N, 0.0);
        srand (i);
        for (uint j = 0; j < N; j++) rvec[j] = double(rand()) / double(RAND_MAX);
        
        for (uint r = 0; r < recs.size(); r++)
        {
            if (DEBUG) 
            {
                cout << "\t check [";
                for (uint k = 0; k < N; k++) cout << " " << recs[r].lb[k];
                cout << " ]" << endl;
                for (uint k = 0; k < N; k++) cout << " " << recs[r].ub[k];
                cout << " ]" << endl;
            }
            
            bool found = true;
            for (uint k = 0; k < N; k++)
            {
                if (rvec[k] < recs[r].lb[k] || rvec[k] > recs[r].ub[k])
                {
                    found = false;
                    break;
                }
            }
            if (found) hits += 1;
        }
        
        if (hits != 1)
        {
            cout << "random vector [";
            for (uint k = 0; k < N; k++) cout << " " << rvec[k];
            cout << " ] has " << hits << " hits!" << endl;
        }
    }
}

/*
 *  Minimize the objective functions over the dimensions and range defined by 
 *  ndim, lb, up.  Will terminate when the first of these happens:
 *
 *    - maxiter iterations reached
 *    - maxtime seconds elapsed
 *    - maxsample samples performed on objective
 *
 * Note that for the latter two termination conditions, we allow DIRECT to
 * finish it's current rectangle division, so there will still be some 
 * time/samples that will be used before it returns.
 */
extern "C" const double* direct(objective_t objective, int ndim, double* lb, double* ub, int maxiter, int maxtime, int maxsample)
{
    time_t start = time(NULL);
    time_t lasttic = time(NULL);
    
    if (DEBUG) cout << "[C++] maxiter = " << maxiter << ",  ";;
    if (DEBUG) cout << "maxtime = " << maxtime << ",  ";
    if (DEBUG) cout << "maxsample = " << maxsample << endl;
    // assert(maxsample <= 101);
    
    // setup
    vector<Rectangle> rectangles;
    fvec lower = fvec(ndim, 0.0);
    fvec upper = fvec(ndim, 0.0);
    for (uint i = 0; i < lower.size(); i++)
    {
        lower[i] = lb[i];
        upper[i] = ub[i];
    }
    
    
    fvec flower = fvec(ndim, 0.0);
    fvec fupper = fvec(ndim, 1.0);

    Direct* D = new Direct(lower, upper, objective);
    
    Rectangle first = Rectangle(flower, fupper, D);
    
    vector<Rectangle> recs = D->divrec(first);

    if (DEBUG)
    { 
        for (uint i = 0; i < recs.size(); i++)
        {
            cout << "\nnew rectangle " << i << endl;
            recs[i].display();
        }
    }
    
    // okay, let's get choppy!
    int iteration = 0; 
    const double epsilon = 10e-10;
    bool done = false;
    while (iteration < maxiter and !done)
    {
        iteration++;
        if (DEBUG) cout << "[cdirect] iteration " << iteration << ", samples = " << D->nsamples << endl;
            
        // find potentially-optimal rectangles
        vector<uint> potopts;
        for (uint j = 0; j < recs.size(); j++)
        {
            // don't split really tiny rectangles...
            if (DEBUG) 
                if (recs[j].d < 10e-6) cout << "[cdirect] rectangle ws tiny..." << endl;
                
            double maxI1 = MIN_DOUBLE;
            double minI2 = MAX_DOUBLE;
            bool breaked = false;
            for (uint i = 0; i < recs.size(); i++)
            {
                if (i==j) continue;
                if (recs[i].d < recs[j].d)
                {
                    // I1
                    double val = (recs[j].y - recs[i].y) / (recs[j].d - recs[i].d);
                    if (val > maxI1) maxI1 = val;
                }
                else if (recs[i].d > recs[j].d)
                {
                    // I2
                    double val = (recs[i].y - recs[j].y) / (recs[i].d - recs[j].d);
                    if (val < minI2)
                    {
                        minI2 = val;
                        if (minI2 <= 0.) 
                        {
                            breaked = true;
                            break;
                        }
                    }
                }
                else
                {
                    // I3
                    if (recs[j].y > recs[i].y) 
                    {
                        breaked = true;
                        break;
                    }
                }
            
                if (maxI1 != MIN_DOUBLE && minI2 != MAX_DOUBLE && minI2 < maxI1)
                {
                    breaked = true;
                    break;
                }
            } // for i
            if (!breaked)
            {
                if (DEBUG) cout << "j = " << j << endl;
                if (minI2==MAX_DOUBLE)
                {
                    potopts.push_back(j);
                    if (DEBUG) cout << "push [1]" << endl;
                }
                else if (D->FMIN == 0.0)
                {
                    if (recs[j].y <= recs[j].d * minI2)
                    {
                        potopts.push_back(j);
                        if (DEBUG) cout << "push [2]" << endl;
                    }
                }
                else if (epsilon <= (D->FMIN-recs[j].y)/abs(D->FMIN) + (recs[j].d/abs(D->FMIN)) * minI2)
                {
                    potopts.push_back(j);
                    if (DEBUG) 
                    {
                        cout << "push [3]: " << epsilon << " < " << (D->FMIN-recs[j].y)/abs(D->FMIN) + (recs[j].d/abs(D->FMIN)) * minI2 << endl;
                        cout << "\tepsilon\t" <<  epsilon << endl;
                        cout << "\tFMIN[0]\t" << D->FMIN << endl;
                        cout << "\tRj.y\t" << recs[j].y << endl;
                        cout << "\tRj.d\t" << recs[j].d << endl;
                        cout << "\tminI2\t" << minI2 << endl;
                    }
                }
            }
        
            // timing
            if (DEBUG && time(NULL)-lasttic > 30)
            {
                time_t mins = time_t(time(NULL)-start) / 60;
                time_t secs = time_t(time(NULL)-start) % 60;
            
                cout << "[cdirect] elapsed time " << mins << ":" << setfill('0') << setw(2) << secs << ".  ";
                cout << "on iteration " << iteration+1 << "/" << maxiter << ", samples = " << D->nsamples;
                cout << " (fmin = " << D->FMIN << ")" << endl;
            
                lasttic = time(NULL);
            }
        
        } // for j
    
        if (potopts.size()==0)
        {
            cout << "[cdirect] could not divide any more" << endl;
            break;
        }
        // cout << "\t" << potopts.size() << " potopt found" << endl;
        for (int ind = potopts.size()-1; ind >= 0; ind--)
        {
            uint j = potopts[ind];
            vector<Rectangle> newrecs = D->divrec(recs[j]);
            recs.insert(recs.end(), newrecs.begin(), newrecs.end());
            recs.erase(recs.begin()+j);
            // test_coverage(lower, upper, recs, 100);
        
            if (D->nsamples > uint(maxsample))
            {
                if (DEBUG) cout << "[cdirect] exceeded " << maxsample << " samples"<< endl;
                done = true;
                break;
            }
            if (time(NULL)-start > maxtime)
            {
                done = true;
                break;
            }
        }
        if (DEBUG) cout << "\tended with " << recs.size() << " rectangles" << endl;
    
        // print rectangles and do sanity test
        // for (uint i = 0; i < recs.size(); i++)
        // {
        //     cout << "\nrectangle " << i << endl;
        //     recs[i].display();
        // }
        // cout << "iteration " << iteration << endl;
    
        if (time(NULL)-start > maxtime)
        {
            if (DEBUG) cout << "[cdirect] timing out" << endl;
            break;
        }
        if (D->nsamples > uint(maxsample))
        {
            if (DEBUG) cout << "[cdirect] exceeded " << maxsample << " samples (" << D->nsamples << ")" << endl;
            break;
        }
    } // while iters
    
    
    // fvec res;
    ostringstream s;
    if (DEBUG) cout << "[C++] valmin = " << D->FMIN << endl;
    // res.push_back(D->FMIN);
    s << D->FMIN << " ";
    if (DEBUG) cout << "[C++] argmin = ";
    for (uint i = 0; i < D->N; i++)
    {
        if (DEBUG) cout << "'" << D->XMIN[i] << "' ";
        s << D->XMIN[i] << " ";
    }
    if (DEBUG) cout << endl;
    s << " ";
    if (DEBUG) cout << "[C++] sending " << (s.good() ? "GOOD" : "BAD") << " string"<< endl;
    if (DEBUG) cout << "[C++] sending " << s.str() << endl;
    if (DEBUG) cout << "[C++] sending " << s.str().c_str() << endl;
    
    // time_t mins = time_t(time(NULL)-start) / 60;
    time_t secs = time_t(time(NULL)-start);
    
    if (DEBUG)
    {
        cout << "[C++] " << D->nsamples << " samples in ";
        if (secs==0)
        {
            cout << "< 1";
        }
        else
        {
            cout << secs;
        }
        cout << "s .  fmin = " << D->FMIN << "." << endl;
        if (DEBUG) cout << "          samples = " << D->nsamples << endl;
        if (DEBUG) cout << "          fmin    = " << setfill('0') << setw(4) << D->FMIN << endl;
    }
    
    if (0)
    {
        cout << "\nwriting out final rectangles..." << endl;
        writerecs("finalrecs.txt", recs);
    }

    double* res = (double*)malloc(sizeof(double)*(ndim+1));
    res[0] = D->FMIN;
    for (uint i = 0; i < D->N; i++)
    {
        res[i+1] = D->XMIN[i];
    }


    delete D;
    
    if (DEBUG) cout << "[cdirect] DIRECT destructed" << endl;
    if (DEBUG) cout << "[cdirect] s = " << s << endl;
    if (DEBUG) cout << "[cdirect] s.str() = " << s.str() << endl;
    if (DEBUG) cout << "[cdirect] s.str().c_str() = " << s.str().c_str() << endl;
    
    // return s.str().c_str();
    return res;
}

