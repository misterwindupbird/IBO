#!python
from numpy import array, log, zeros, exp, sqrt, sum, vstack, ones, arange, clip
from numpy.linalg import norm
# from scipy.special import gamma, yv


class Kernel(object):
    """
    base class for kernels.
    """
    def __init__(self, hyperparams):
        self._hyperparams = array(hyperparams)
        self._hyperparams.setflags(write=False)

    # this needs to be read-only, or bad things will happen
    def getHyperparams(self):
        return self._hyperparams

    hyperparams = property(getHyperparams)

    def cov(self, x1, x2):
        raise NotImplementedError('kernel-derived class does not have cov method')
    
    
    def covMatrix(self, X):
        NX, _ = vstack(X).shape
        K = ones((NX, NX))
        for i in xrange(NX):
            for j in xrange(i+1):
                K[i, j] = K[j, i] = self.cov(X[i], X[j])
        
        return K
        
        
    def derivative(self, X, hp):
        raise NotImplementedError('kernel-derived class does not have derivative method')


class SVKernel(object):
    
    def __init__(self, mag):
        self._magnitude = mag
        self._sf2 = exp(2.0*log(self._magnitude))  # signal variance
        
    def covScale(self, k):
        
        return self._sf2 * k
    
    
class GaussianKernel_iso(Kernel):
    """
    Isotropic Gaussian (aka "squared exponential") kernel.  Has 2
    non-negative hyperparameters:
    
        hyperparams[0]      kernel width parameter
        hyperparams[1]      noise magnitude parameter
    """
    
    def __init__(self, hyperparams, **kwargs):
        super(GaussianKernel_iso, self).__init__(hyperparams)
        self._itheta2 = 1 / hyperparams[0]**2
        # self._magnitude = hyperparams[1]
        # self._sf2 = exp(2.0*log(self._magnitude))  # signal variance
        

    def cov(self, x1, x2):
        
        return exp(-.5 * norm(x1-x2)**2 * self._itheta2)
        

    def derivative(self, X, hp):
        
        NX, _ = vstack(X).shape
        K = self.covMatrix(X)
        
        if hp == 0:
            C = zeros(K.shape)
            for i in xrange(NX):
                for j in xrange(i):
                    C[i, j] = C[j, i] = sum((X[i]-X[j])**2) * self._itheta2
            return K * C
        # elif hp == 1:
        #     return 2.0 * K
        else:
            raise ValueError


class SVGaussianKernel_iso(SVKernel, GaussianKernel_iso):

    def __init__(self, hyperparams, **kwargs):

        GaussianKernel_iso.__init__(self, hyperparams[:-1])
        SVKernel.__init__(self, hyperparams[-1])
        self._hyperparams = array(hyperparams)
        self._hyperparams.setflags(write=False)
        
    def cov(self, x1, x2):
        
        return self.covScale(GaussianKernel_iso.cov(self, x1, x2))
        
    def derivative(self, X, hp):
        
        if hp==0:
            return GaussianKernel_iso.derivative(self, X, hp)
        elif hp==1:
            return 2.0 * self.covMatrix(X)
        

class GaussianKernel_ard(Kernel):
    """
    Anisotropic Gaussian (aka "squared exponential") kernel.  Has D+1
    non-negative hyperparameters, where D is dimensionality.  The first
    D are the length-scale hyperparameters for the dimensions, and the
    D+1th is the noise magnitude.
    """
    
    def __init__(self, hyperparams, **kwargs):
        
        super(GaussianKernel_ard, self).__init__(hyperparams)
        self._theta = clip(hyperparams, 1e-4, 1e4)
        self._itheta2 = array([1.0/t**2 for t in self._theta])
        # self._magnitude = hyperparams[-1]
        # self._sf2 = exp(2.0*log(self._magnitude))  # signal variance
        

    def cov(self, x1, x2):
        
        return exp(-.5 * sum(self._itheta2 * (x1-x2)**2))
        

    def derivative(self, X, hp):
        
        NX, NA = vstack(X).shape
        K = self.covMatrix(X)
        
        if hp < NA:
            C = zeros(K.shape)
            for i in xrange(NX):
                for j in xrange(i):
                    C[i, j] = C[j, i] = sum(self._itheta2[hp]*(X[i][hp]-X[j][hp])**2.0)
            return K * C
        # elif hp == NA:
        #     return 2.0 * K
        else:
            raise ValueError


class SVGaussianKernel_ard(SVKernel, GaussianKernel_ard):

    def __init__(self, hyperparams, **kwargs):

        GaussianKernel_ard.__init__(self, hyperparams[:-1])
        SVKernel.__init__(self, hyperparams[-1])
        self._hyperparams = array(hyperparams)
        self._hyperparams.setflags(write=False)
        
        
    def cov(self, x1, x2):
        
        return self.covScale(GaussianKernel_ard.cov(self, x1, x2))
        
    def derivative(self, X, hp):
        
        if hp < len(self._theta):
            return GaussianKernel_ard.derivative(self, X, hp)
        elif hp==len(self._theta):
            return 2.0 * self.covMatrix(X)


class MaternKernel3(Kernel):
    """
    Matern kernel for nu=3/2.  Distance measure is isotropic.  Exact formulation
    is based on Rasmussen & Williams. There are 2 non-negative hyperparameters:
    
        hyperparams[0]      kernel width parameter
        hyperparams[1]      noise magnitude parameter
    """
    
    def __init__(self, hyperparams, **kwargs):
        super(MaternKernel3, self).__init__(hyperparams)
        self._theta = hyperparams[0]
        self._magnitude = hyperparams[1]
        self._sf2 = exp(2.0*log(self._magnitude))
        self.sqrt3 = sqrt(3)

    def cov(self, x1, x2):
        
        z = self.sqrt3 * norm(x1-x2) / self._theta
        return self._sf2 * (1.0 + z) * exp(-z)

    def derivative(self, X, hp):
        
        NX, _ = vstack(X).shape
        K = self.covMatrix(X)
        
        if hp == 0:
            C = zeros(K.shape)
            for i in xrange(NX):
                for j in xrange(i):
                    r = norm(X[i]-X[j])
                    C[i, j] = C[j, i] = self._sf2 * r**2 * exp(-r)
            return C
        elif hp == 1:
            return 2.0 * K
        else:
            raise ValueError
        

class MaternKernel5(Kernel):
    """
    Matern kernel for nu=5/2.  Distance measure is isotropic.  Exact formulation
    is based on Rasmussen & Williams. There are 2 non-negative hyperparameters:
    
        hyperparams[0]      kernel width parameter
        hyperparams[1]      noise magnitude parameter
    """
    
    def __init__(self, hyperparams, **kwargs):
        super(MaternKernel5, self).__init__(hyperparams)
        self._theta = hyperparams[0]
        self._magnitude = hyperparams[1]
        self._sf2 = exp(2.0*log(self._magnitude))
        

    def cov(self, x1, x2):
        z = sum((sqrt(5.0) * array(x1-x2) / self._theta)**2.0)
        z = self._sf2 * exp(-sqrt(z)) * (1.0 + sqrt(z) + z/3.0)
        print z

    def derivative(self, X, hp):
        
        NX, _ = vstack(X).shape
        K = self.covMatrix(X)
        
        if hp == 0:
            C = zeros(K.shape)
            for i in xrange(NX):
                for j in xrange(i):
                    z = sum((sqrt(5.0) * array(X[i]-X[j])/self._theta)**2.0)
                    C[i, j] = C[j, i] = self._sf2 * (z + sqrt(z)**3.0) * exp(-sqrt(z))/3.0
            return C
        elif hp == 1:
            return 2.0 * K
        else:
            raise ValueError
        
        

# class LinearTestKernel(Kernel):
#     """
#     just for debugging: don't use this for real
#     """
#     
#     def __init__(self, scale=1.0, **kwargs):
#         super(LinearKernel, self).__init__()
#         self.scale = scale
#         
#     def cov(self, x1, x2):
#         
#         return max(1.0 - sum(abs(x1-x2))/self.scale, 0.0)

if __name__=="__main__":
    
    kernel = SVGaussianKernel_iso(array([1., 3.]))
    print kernel._itheta2
    print kernel._sf2
    X = arange(0.0, 8.0).reshape(2,-1)
    print kernel.covMatrix(X)
    print kernel.derivative(X, 0)
    print kernel.derivative(X, 1)
    kernel.hyperparams[1] = 10000
    print kernel.hyperparams
    kernel.hyperparams = None
    print kernel.hyperparams
        
