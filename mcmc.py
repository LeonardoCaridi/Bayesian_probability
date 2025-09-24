import numpy as np
from scipy.special import logsumexp

def log_prior(theta, bounds):
    """
    Prior uniformi dentro i confini definiti da bounds
    """
    for i in range(theta.shape[0]):
        if theta[i] < bounds[i][0] or theta[i] > bounds[i][1]:
            return -np.inf
    
    return 0.0

def log_T90_distribution(x, theta, sigma_logT90 = 0.0):
    """
    Parametri:
    - x: dati log(T90) (logaritmo naturale)
    - theta: parametri distribuzione
    - sigma_logT90: eventuale errore gaussiano su log(T90)
    """

    w, mu1, delta, sigma1, sigma2 = theta
    if (sigma1 <= 0) or (sigma2 <= 0) or (w <= 0) or (w >= 1) or (delta < 0): return -np.inf
    mu2    = mu1 + delta
    var1   = sigma1**2 + sigma_logT90**2
    var2   = sigma2**2 + sigma_logT90**2

    logN1 = -0.5*(np.log(2*np.pi*var1) + ((x-mu1)**2)/var1)
    logN2 = -0.5*(np.log(2*np.pi*var2) + ((x-mu2)**2)/var2)
    stacked = np.vstack([np.log(w) + logN1, np.log(1-w) + logN2]) 
    # logsumexp(logN1, logN2) = log(N1 + N2), indica la distribuzione di un solo dato
    return logsumexp(stacked, axis=0)     

def log_likelihood(x, theta, sigma_logT90 = 0.0):
    # Assumo i T90 indipendenti e sommo su tutte le probabilità: np.sum()
    return np.sum(log_T90_distribution(x, theta, sigma_logT90), axis=0)

def proposal_distribution(x,  rng = None):
    
    d = x.shape[0]
    covariance = np.eye(d)
    
    #per valori simulati {"w": 0.3, "mu1": -0.5, "mu2": 3.5, "sigma1": 0.8, "sigma2": 1.0}
    covariance = np.array(([ 0.01545509,  0.08896731, -0.02650015, -0.12120278, -0.05484408],
                           [ 0.08896731,  0.55401367, -0.19294877, -0.70861588, -0.31702254],
                           [-0.02650015, -0.19294877,  0.08722617,  0.21468223,  0.09385355],
                           [-0.12120278, -0.70861588,  0.21468223,  0.99010538,  0.43217997],
                           [-0.05484408, -0.31702254,  0.09385355,  0.43217997,  0.19659419]))    
    """
    # per valori reali
    covariance = np.array(([ 0.00273077,  0.01525256, -0.01151554,  0.0149962,  -0.00317278],
                           [ 0.01525256,  0.10883794, -0.08290754,  0.08776148, -0.0157554 ],
                           [-0.01151554, -0.08290754,  0.06957072, -0.07510057,  0.00860551],
                           [ 0.0149962,   0.08776148, -0.07510057,  0.11197837, -0.01227969],
                           [-0.00317278, -0.0157554,   0.00860551, -0.01227969,  0.00767069]))
    """
    if rng is None:
        rng = np.random
    
    return rng.multivariate_normal(np.zeros(d,dtype=np.float64), covariance)

def log_posterior(x, theta, bounds, sigma_logT90 = 0.0):
    return log_prior(theta, bounds)+log_likelihood(x, theta, sigma_logT90 = sigma_logT90)

def generate_data(theta_true, N, rng = None, sigma_logT90 = 1.0):
    """
    Parametri:
    - theta_true: dizionario contenente i parametri noti
    - N: numero sample generati
    - sigma_logT90: rumore gaussiano della misura logT90
    """
    if rng is None:
        rng = np.random

    # Generate synthetic data
    # Genera una maschera: z = 1 seleziona N1 (p = w), z = 0 seleziona N2 (p = 1-w)
    z = rng.random(N) < theta_true["w"] 
    logT90_gen = np.where(z, 
                rng.normal(theta_true["mu1"], theta_true["sigma1"], size=N),
                rng.normal(theta_true["mu2"], theta_true["sigma2"], size=N))

    return logT90_gen + rng.normal(0.0, sigma_logT90, size=N)

def metropolis_hastings(theta0, x, bounds, rng = None, sigma_logT90 = 0.0, n = 1000):
    """
    Parametri: 
    - theta0: initial parameters numpy array
    - x: logT90
    - bounds: intervalli della prior
    - sigma_logT90: eventale errore gaussiano sui dati logT90
    - n: numero di iterazioni
    """
    if rng is None:
        rng = np.random
    
    accepted = 0
    rejected = 1
    
    theta0 = np.atleast_1d(theta0)

    d = theta0.shape[0]
    
    logP0   = log_posterior(x, theta0, bounds, sigma_logT90 = sigma_logT90)
    samples = np.zeros((n,d), dtype=np.float64)
    
    for i in range(n):
        
        theta_t = theta0 + proposal_distribution(theta0, rng = rng)
        logP_t = log_posterior(x, theta_t, bounds, sigma_logT90 = sigma_logT90)
        
#        print('logP proposal = ',logP_t,'logP0 = ',logP0)
        
        if logP_t - logP0 > np.log(rng.uniform(0,1)):
            theta0       = theta_t
            logP0        = logP_t
            samples[i,:] = theta_t
            accepted    += 1
        else:
            samples[i,:] = theta0
            rejected    += 1
        
        print("iteration {0}: acceptance {1}".format(i,accepted/float(accepted+rejected)))
    overall_rate = accepted / float(n)
    print("Adaptive Metropolis finished. Acceptance rate = {:.4f}".format(overall_rate))
    
    return samples

def next_pow_two(n):
    """
    find the next power of 2 given n
    """
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorrelation(x, norm=True):
    """
    compute the autocorrelation function of an array
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf
