# Gamma-ray burst classification

A threshold for the completion of the exercise is to complete points 1a), 2), 3) and 4).  
Gamma-ray bursts (GRBs) are usually classified into two different categories, according to their
duration: short GRBs and long GRBs. The distribution of the burst duration, $T$, is usually
modelled as the weighted sum of two log-normal distributions:

\[
p(T_{90}) = w_1\,\mathcal{N}\big(\log(T_{90})\mid\mu_1,\sigma_1\big) \, + \, (1 - w_1)\,\mathcal{N}\big(\log(T_{90})\mid\mu_2,\sigma_2\big)
\]

We will make use of Fermi/GBM data available here (LINK). Unless stated, we will neglect
measurement uncertainties.

1. **Determine:**
   a. The parameters of the distribution.  
   b. As above, assuming Gaussian uncertainties on each $\log T_{90}$.

Assuming the parameters inferred in the previous point, we now turn our attention to the problem
of classifying each GRB.

2. Compute the probability of GRB170817A (T = 2.0 s) of being a short GRB or a long GRB.

3. Decide a figure of merit and determine the threshold value for $T_{90}$ to
   discriminate between short and long GRBs.

Some authors propose a third class of GRBs, *intermediate*.

4. Which of the two hypothesis is favored according to the available data?

Another possibility is to classify GRBs in **soft** and **hard**, according to the hardness ratio

\[
HR = \frac{F_{50-100\,\mathrm{keV}}}{F_{20-50\,\mathrm{keV}}}
\]

where $F$ is the measured flux in a certain energy interval.

5. Study the bimodal distribution in the $\log(T_{90})$ - $\log(HR)$ space. (Suggestion: specialize
https://dp.tdhopper.com/collapsed-gibbs/ to a fixed number of components.)
