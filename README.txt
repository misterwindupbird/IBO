Interactive Bayesian Optimization with Gaussian Processes.  All files copyright Eric Brochu 2010.

This is an experimental pre-release version and may contain bugs.  It also has lots of code that you don't really need but I do.


=== INSTALLATION ===

This package expects you to have Python 2.6.x, NumPy and SciPy installed.  An easy way to get it all at once it to download the Enthought Python Distribution from enthought.com.

After cloning the repository, you will need to add the package path to your PYTHONPATH.  On OS X, I added the following lines to .bash_login:

    PYTHONPATH=${PYTHONPATH}:/Users/eric/projects/EGOcode/
    export PYTHONPATH
    
The package contains C++ code for an OSX dynamic library, though it should be pretty simple to compile it on Linux if need be.  This library is optional.  It adds no functionality, but it speeds up the algorithm substantially.  To build it, jut run

    make depend
    make
    
You will then need to add the library path (/path/to/EGOcode/cpp/libs/) to your DYLD_LIBRARY_PATH.

To make sure the system is set up properly, run the unittests in EGOcode/ego.  There may be warnings if the C++ library is not found, but all tests should pass.


=== USING THE PACKAGE ===

The file demo.py provides examples of calling the package.  Basically, you create a GaussianProcess if you have direct observations, or PrefGaussianProcess if you have preference observations.  You can then add data to the GP, and call maximizeEI() or fastUCBGallery() to find informative query points.  

The 'bound' argument controls the ranges of the parameters that the queries will return on.  If they are set so that the lower bound is equal to the upper bound, that parameter is fixed at the indicated value.  For example:

>>> fastUCBGallery(GP, [[-1, 1.5], [0.5, 0.5]], 4)

will get a gallery of 4 query points over the range [-1, 1.5] for the first parameter, fixing the second parameter at 0.5.

Other use cases can be found in the unit tests.

