Gamma-ray burst classification

A threshold for the completion of the exercise is to complete points 1a), 2), 3) and 4).
Gamma-ray bursts (GRBs) are usually classified into two different catagories, according to their
duration: short GRBs and long GRBs. The distribution of the burst duration, T, is usually
modelled as the weighted sum of two log-normal distributions:

p(T_90) = w1 N(log(T_90) | mu1, sigma1) + (1 - w1) N(log(T_90) | mu2, sigma2)

We will make use of Fermi/BATSE data available in GRBs folder. Unless stated, we will neglect
measurement uncertainties.

1. Determine:
   a. The parameters of the distribution.
   b. As above, assuming Gaussian uncertainties on each log T90

Assuming the parameters inferred in the previous point, we now turn our attention to the problem
of classifying each GRB.

2. Compute the probability of GRB170817A (T = 2.0 s) of being a short GRB or a long GRB.

3. Decide a figure of merit and determine the threshold value for T90 to
   discriminate between short and long GRBs.

Some authors propose a third class of GRBs, intermediate.

4. Which of the two hypothesis is favored according to the available data?

Another possibility is to classify GRBs in soft and hard, according to the hardness ratio

HR = F_{50-100 keV} / F_{20-50 keV}

where F is the measured flux in a certain energy interval.

5. Study the bimodal distribution in the log(T90) - log(HR) space. (Suggestion: specialize
   https://dp.tdhopper.com/collapsed-gibbs/ to a fixed number of components.)
