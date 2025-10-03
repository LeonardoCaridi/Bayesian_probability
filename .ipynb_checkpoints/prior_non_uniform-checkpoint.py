import numpy as np

def log_prior(theta, bounds):
    
    Prior A:
      - w ~ Beta(2,2)
      - mu1 ~ Normal(0, 3)
      - delta ~ HalfNormal(sd=3)  (delta > 0)
      - sigma1, sigma2 ~ HalfNormal(sd=2)  (sigma > 0)
    bounds = [[w_min,w_max], [mu1_min,mu1_max], [delta_min,delta_max],
              [sigma1_min,sigma1_max], [sigma2_min,sigma2_max]]
    

    theta = np.atleast_1d(theta)
    try:
        w, mu1, delta, s1, s2 = theta.tolist()
    except Exception:
        # se theta non decomponibile correttamente
        return -np.inf

    # Bounds check (uso gli stessi bounds passati)
    if not (bounds[0][0] <= w <= bounds[0][1]):
        return -np.inf
    if not (bounds[1][0] <= mu1 <= bounds[1][1]):
        return -np.inf
    if not (bounds[2][0] <= delta <= bounds[2][1]):
        return -np.inf
    if not (bounds[3][0] <= s1 <= bounds[3][1]):
        return -np.inf
    if not (bounds[4][0] <= s2 <= bounds[4][1]):
        return -np.inf

    # Additional validity checks (mi assicuro che w sia strettamente tra 0 e 1,
    # e che delta e sigma siano positivi)
    if (w <= 0.0) or (w >= 1.0):
        return -np.inf
    if (delta <= 0.0) or (s1 <= 0.0) or (s2 <= 0.0):
        return -np.inf

    # --- Calcolo log-prior (inclusi termini di normalizzazione) ---
    logp = 0.0

    # Beta(2,2) on w: pdf ∝ w^(1) * (1-w)^(1)
    # normalization Beta(2,2) = 1/6, quindi logpdf = log(w) + log(1-w) - log(1/6)
    # possiamo includere la costante, ma non è necessario per MH: la includo per completezza
    log_beta22_const = -np.log(1.0/6.0)  # = log(6)
    logp += np.log(w) + np.log(1.0 - w) + log_beta22_const

    # mu1 ~ Normal(0, 3)
    sd_mu1 = 3.0
    logp += -0.5 * ((mu1 - 0.0)**2) / (sd_mu1**2) - 0.5 * np.log(2.0 * np.pi * sd_mu1**2)

    # delta ~ HalfNormal(sd=3): pdf(x) = sqrt(2/pi)/sd * exp(-x^2/(2 sd^2))  for x>=0
    sd_delta = 3.0
    if delta < 0:
        return -np.inf
    log_halfnorm_const = 0.5 * (np.log(2.0) - np.log(np.pi)) - np.log(sd_delta)
    logp += log_halfnorm_const - 0.5 * (delta**2) / (sd_delta**2)

    # sigma1, sigma2 ~ HalfNormal(sd=2)
    sd_sigma = 2.0
    log_halfnorm_const_sigma = 0.5 * (np.log(2.0) - np.log(np.pi)) - np.log(sd_sigma)
    logp += log_halfnorm_const_sigma - 0.5 * (s1**2) / (sd_sigma**2)
    logp += log_halfnorm_const_sigma - 0.5 * (s2**2) / (sd_sigma**2)

    return logp
