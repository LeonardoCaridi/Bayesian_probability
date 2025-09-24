import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

def FindHeightForLevel(inArr, adLevels):
    """
    Computes the height of a :math:`N Dim` function for given levels
    
    :param inArr: function values
    :type inArr: array
    :param adLevels: levels
    :type adLevels: list or array
    
    :return: function values with levels closest to *levels*
    :rtype: array
    """
    
    # flatten the array
    oldshape = np.shape(inArr)
    adInput  = np.reshape(inArr, np.prod(oldshape))
    
    # get array specifics
    nLength  = np.size(adInput)

    # create reversed sorted list
    adTemp   = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted

    # create the normalised cumulative distribution
    adCum    = np.zeros(nLength)
    adCum[0] = adSorted[0]
    
    for i in range(1,nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
        
    adCum    = adCum - adCum[-1]

    # find the values closest to levels
    adHeights = []
    for item in adLevels:
        idx = (np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])

    adHeights = np.array(adHeights)
    return np.sort(adHeights)

def log_prior(theta):
    return 0.0

def log_likelihood(x, y, model, theta, sigma = 1.0):
    
    prediction = model(x, theta)
    residuals  = (y - prediction)/sigma
    logL = -0.5*np.sum(residuals**2)
    
    return logL

def log_posterior(x, y, model, theta, sigma = 1.0):
    return log_prior(theta)+log_likelihood(x, y, model, theta, sigma = sigma)

def line(x, theta):
    return theta[0]*np

def parabola(x, theta):
    return theta[0]+theta[1]*x**2

def generate_data(x, model, theta, rng = None, sigma = 1.0):
    
    if rng is None:
        rng = np.random
    
    noise  = rng.normal(0.0, sigma, size=(x.shape[0]))
    signal = model(x, theta)
    return noise + signal

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng(1234)
    x = np.linspace(0.,1.,10000)
    
    sigma_noise = 0.5
    nbins       = 256
    
    a = 0.3
    b = 1.2
    y = generate_data(x, line, (a,b), rng = rng, sigma=sigma_noise)
    
    print('logL (simulation)= ', log_likelihood(x, y, line, (a,b), sigma = sigma_noise))
    
    # case 1: the line
    
    intercept = np.linspace(-2,2,nbins)
    slope     = np.linspace(-5,5,nbins)
    
    logP      = np.zeros((nbins,nbins), dtype = np.float64)
    
    for i in range(nbins):
        for j in range(nbins):
            logP[i,j] = log_posterior(x,y,line,(intercept[i],slope[j]), sigma = sigma_noise)

    
    print('logZ (line)= {}'.format(logsumexp(logP)*np.diff(x)[0]*np.diff(y)[0]) )
    
    levels = np.sort(FindHeightForLevel(logP,[0.5,0.9]))

    X,Y = np.meshgrid(intercept, slope)
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    C   = ax.contourf(X,Y, logP.T, 100)
    ax.contour(X,Y,logP.T,levels, linestyles='-', colors='k')
    ax.axhline(b)
    ax.axvline(a)
    ax.set_xlabel('intercept')
    ax.set_ylabel('slope')
    plt.colorbar(C)
    
    # compute the intercept marginal
    
    fig = plt.figure(2)
    ax  = fig.add_subplot(121)
    pdf = np.exp(logP).sum(axis=1)*np.diff(y)[0]
    ax.plot(intercept, pdf,'k')
    ax.axvline(a)
    ax.set_xlabel('intercept')
    
    ax  = fig.add_subplot(122)
    pdf = np.exp(logP).sum(axis=0)*np.diff(x)[0]
    ax.plot(slope, pdf,'k')
    ax.axvline(b)
    ax.set_xlabel('slope')
    
    # case 2: the parabola

    intercept = np.linspace(-2,2,nbins)
    slope     = np.linspace(-5,5,nbins)
    
    logP      = np.zeros((nbins,nbins), dtype = np.float64)
    
    for i in range(nbins):
        for j in range(nbins):
            logP[i,j] = log_posterior(x,y,parabola,(intercept[i],slope[j]), sigma = sigma_noise)

    
    print('logZ (parabola)= {}'.format(logsumexp(logP)*np.diff(x)[0]*np.diff(y)[0]) )
    
    levels = np.sort(FindHeightForLevel(logP,[0.5,0.9]))

    X,Y = np.meshgrid(intercept, slope)
    fig = plt.figure(4)
    ax  = fig.add_subplot(111)
    C   = ax.contourf(X,Y, logP.T, 100)
    ax.contour(X,Y,logP.T,levels, linestyles='-', colors='k')
    ax.axhline(b)
    ax.axvline(a)
    ax.set_xlabel('intercept')
    ax.set_ylabel('slope')
    plt.colorbar(C)
    
    # compute the intercept marginal
    
    fig = plt.figure(5)
    ax  = fig.add_subplot(121)
    pdf = np.exp(logP).sum(axis=1)*np.diff(y)[0]
    ax.plot(intercept, pdf,'k')
    ax.axvline(a)
    ax.set_xlabel('intercept')
    
    ax  = fig.add_subplot(122)
    pdf = np.exp(logP).sum(axis=0)*np.diff(x)[0]
    ax.plot(slope, pdf,'k')
    ax.axvline(b)
    ax.set_xlabel('slope')
    
#    plt.show()
#    exit()
    
    fig = plt.figure(6)
    ax  = fig.add_subplot(111)
    ax.plot(x,y,'ob',label = 'data')
    ax.errorbar(x,y,yerr = sigma_noise)
    ax.plot(x,line(x,(a,b)),'-g', label='simulation')
    plt.legend()
    plt.show()
    
    

