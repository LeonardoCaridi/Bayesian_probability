import numpy as np
from scipy.special import logsumexp


def log_prior(theta, bounds):
    """
    Uniform prior inside the bounds 
    """

    # Controlla parametri da 1 a 4 (esclude w)
    for i in range(1, theta.shape[0]):
        if theta[i] < bounds[i][0] or theta[i] > bounds[i][1]:
            return -np.inf
    
    # Controllo esplicito su w per non avere valori 0 o 1     
    if not (0.0 < theta[0] < 1.0):
        return -np.inf

    # w prior = Beta(2,2) = 6*x*(1-x) 
    # La normalizzazione non è necessaria (viene levato il 6)
    logp = np.log(theta[0]) + np.log(1-theta[0])
    
    return logp  

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

def proposal_distribution(x, init_cov = None, rng = None):
    
    d = x.shape[0]
    if init_cov is None:
        covariance = 0.01*np.eye(d)
    else:
        covariance = init_cov
    """
    covariance = np.array(([ 1.16001274e-04, 4.34444634e-05, -8.75870281e-06, 3.74859215e-05, -2.68890398e-05],
                           [ 4.34444634e-05, 1.30336403e-03, -1.12672730e-03, 2.51966524e-04, -1.73614797e-04],
                           [-8.75870281e-06, -1.12672730e-03, 1.90314140e-03, -9.62649007e-05, 8.67849825e-06],
                           [ 3.74859215e-05, 2.51966524e-04, -9.62649007e-05, 8.14646898e-04, -1.35645581e-04],
                           [-2.68890398e-05, -1.73614797e-04, 8.67849825e-06, -1.35645581e-04, 5.28118669e-04]))
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

def metropolis_hastings(theta0, x, bounds, init_cov=None, rng = None, sigma_logT90 = 0.0, n = 1000):
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
        
        theta_t = theta0 + proposal_distribution(theta0, init_cov=init_cov, rng = rng)
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

        if(i % (n*0.05) == 0):
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

def init_theta_from_data(x, bounds):
    # percentili utili
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    mu1_0 = p25
    delta_0 = max(0.5, p75 - p25)   # almeno 0.5
    w0 = 0.5
    s1_0 = 1.0
    s2_0 = 1.0
    # clip to bounds
    theta0 = np.array([np.clip(w0, bounds[0][0]+1e-6, bounds[0][1]-1e-6),
                       np.clip(mu1_0, bounds[1][0], bounds[1][1]),
                       np.clip(delta_0, bounds[2][0]+1e-6, bounds[2][1]),
                       np.clip(s1_0, bounds[3][0], bounds[3][1]),
                       np.clip(s2_0, bounds[4][0], bounds[4][1])], dtype=np.float64)
    return theta0

def autocorr(x, lag):
    return np.corrcoef(np.array([x[:-lag], x[lag:]]))[0,1]

def ess(x):
    n = len(x)
    # calcolo autocorrelazione fino a quando diventa negativa
    rho = []
    for k in range(1, n//2):
        r = autocorr(x, k)
        if r <= 0: 
            break
        rho.append(r)
    tau = 1 + 2*np.sum(rho)
    return n / tau