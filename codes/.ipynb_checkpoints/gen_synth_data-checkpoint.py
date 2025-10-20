import numpy as np

def gen_synth_data(sample, N, model='H0', sigma_logT90=1.0, rng=None, verbose=True):
    """
    Generate synthetic logT90 data from a parameter vector or dict for model H0 or H1.
    
    Parameters
    ----------
    sample : array-like or dict
        If array-like: for H0 expects [v, mu1, delta, ln_s1, ln_s2]
                       for H1 expects [v1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3]
        If dict: contain explicit keys
            H0: {'w': [w1,w2] , 'mus': [mu1, mu2], 'sigmas': [s1, s2]}
            H1: {'w': [w1,w2,w3], 'mus':[mu1, mu2, mu3], 'sigmas':[s1, s2, s3]} 
    N : int
        Number of synthetic samples to generate.
    model : {'H0','H1'}
        Which model to use to interpret `sample`.
    sigma_logT90 : float or array-like
        Measurement noise (std). If array-like and length==N it is used per datum,
        otherwise its median is used as scalar noise level.
    rng : numpy.random.Generator or None
        RNG to use (default: np.random.default_rng()).
    verbose : bool
        Whether to print summary info.
    
    Returns
    -------
    synth_obs : ndarray, shape (N,)
        Synthetic observed logT90 values (components + measurement noise).
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    # --- interpret sample into weights, mus, sigs ---
    param_array = None  # canonical parameter array to print later
    
    # helper to accept dict or array
    if isinstance(sample, dict):
        s    = sample
        w    = np.asarray(s['w'], dtype=float)
        mus  = np.asarray(s['mus'], dtype=float)
        sigs = np.asarray(s['sigmas'], dtype=float)

        # build canonical param_array from dict (depending on model)
        if model == 'H0':
            # v from w[0], delta from mus
            v = float(w[0]) if w.size >= 1 else 0.0
            mu1 = float(mus[0]) if mus.size >= 1 else 0.0
            mu2 = float(mus[1]) if mus.size >= 2 else mu1
            delta = mu2 - mu1
            ln_s1 = float(np.log(sigs[0])) if sigs.size >= 1 else 0.0
            ln_s2 = float(np.log(sigs[1])) if sigs.size >= 2 else 0.0
            param_array = np.array([v, mu1, delta, ln_s1, ln_s2], dtype=float)
        elif model == 'H1':
            # derive v1, v2 from w if possible: v1 = w1, v2 = w2/(1-w1) if denom>0
            w = np.asarray(w, dtype=float)
            w1 = float(w[0]) if w.size >= 1 else 0.0
            if w.size >= 2 and (1.0 - w1) > 0:
                v2 = float(w[1]) / float(1.0 - w1)
            else:
                v2 = 0.0
            mu1 = float(mus[0]) if mus.size >= 1 else 0.0
            mu2 = float(mus[1]) if mus.size >= 2 else mu1
            mu3 = float(mus[2]) if mus.size >= 3 else mu2
            delta12 = mu2 - mu1
            delta23 = mu3 - mu2
            ln_s1 = float(np.log(sigs[0])) if sigs.size >= 1 else 0.0
            ln_s2 = float(np.log(sigs[1])) if sigs.size >= 2 else 0.0
            ln_s3 = float(np.log(sigs[2])) if sigs.size >= 3 else 0.0
            param_array = np.array([w1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3], dtype=float)
        else:
            raise ValueError("model must be 'H0' or 'H1'")
            
    else:
        # array-like path: interpret the parameterization used in the class
        arr = np.asarray(sample, dtype=float)
        if model == 'H0':
            if arr.size != 5:
                raise ValueError("H0 sample array must have length 5: [v, mu1, delta, ln_s1, ln_s2]")
            v, mu1, delta, ln_s1, ln_s2 = arr
            w = np.array([v, 1.0 - v], dtype=float)
            mus = np.array([mu1, mu1 + delta], dtype=float)
            sigs = np.exp(np.array([ln_s1, ln_s2], dtype=float))
        elif model == 'H1':
            if arr.size != 8:
                raise ValueError("H1 sample array must have length 8: [v1,v2,mu1,delta12,delta23,ln_s1,ln_s2,ln_s3]")
            v1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3 = arr
            w1 = v1
            w2 = (1.0 - v1) * v2
            w3 = (1.0 - v1) * (1.0 - v2)
            w = np.array([w1, w2, w3], dtype=float)
            mu2 = mu1 + delta12
            mu3 = mu2 + delta23
            mus = np.array([mu1, mu2, mu3], dtype=float)
            sigs = np.exp(np.array([ln_s1, ln_s2, ln_s3], dtype=float))
        else:
            raise ValueError("model must be 'H0' or 'H1'")

    # basic sanity on weights
    # put negative weight to 0 and normalize if sum(w)>1
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        raise ValueError("bad mixture weights (all nonpositive)")
    w = w / s

    mus = np.asarray(mus, dtype=float)
    sigs = np.asarray(sigs, dtype=float)
    if np.any(sigs <= 0):
        raise ValueError("sigs must be positive")

    # --- generate component indices and component draws ---
    k = len(w)
    comps = rng.choice(k, size=N, p=w)
    # vectorized gaussian draws: one per sample from the chosen component
    synth_components = rng.normal(loc=mus[comps], scale=sigs[comps])

    # --- measurement noise handling ---
    sigma_err = np.asarray(sigma_logT90)
    if sigma_err.ndim == 0:
        meas_sigma = float(sigma_err)
        meas_noise = rng.normal(loc=0.0, scale=meas_sigma, size=N)
    else:
        # if provided vector and matches N, use per-sample; else use median fallback
        if sigma_err.size == N:
            meas_noise = rng.normal(loc=0.0, scale=sigma_err)
        else:
            meas_sigma = float(np.median(sigma_err))
            meas_noise = rng.normal(loc=0.0, scale=meas_sigma, size=N)

    synth_obs = synth_components + meas_noise

    if verbose:
        info = {
            'weights': w,
            'mus': mus,
            'sigmas': sigs,
            'N': N
        }
        print("Generated synthetic mixture:")
        print(info)
        
        # print canonical parameter arry
        if param_array is not None:
            if model == 'H0':
                print("\nCanonical params (H0) [v, mu1, delta, ln_s1, ln_s2]:\n", np.asarray(param_array, dtype=float))
            else:
                print("\nCanonical params (H1) [v1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3]:\n", np.asarray(param_array, dtype=float))
        
        return synth_obs, np.asarray(param_array, dtype=float)

    return synth_obs
