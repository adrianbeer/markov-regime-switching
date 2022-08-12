# markov-regime-switching
Modelling regime switching index price movements with markov chains. 

# Specifications
1. **Forecast horizons:** for quarterly GDP growth try a nowcast (h=0, all indicators of that quarter are known) 
and forecast horizons h=1 to 4 quarters. For the year-on-year inflation rate, try forecast horizons h=1 to h=12 months. 
2. Use a **recursive out-of-sample forecast experiment.** 
For each horizon h, use such an experiment to generate forecasts of the first quarter/month of 2000 to the last 
quarter/month of 2021. This will make your forecasts comparable to each other. In addition, think about sensibly 
defined subsamples.
3. Use at least the following three **loss functions**: 
   - mean forecast error
   - mean absolute forecast error
   - root mean squared forecast error.

# TODO
Use MS over MS-DFM ?

(+) Can be used to construct direct density forecasts. However in the MS-DFM the estimated factor and transition 
probabilities can be used in a subsequent model, like an ADL to construct forecasts. 

(+) Simpler. MS doesn't require a Kalman-Filter or the like to estimate probabilities. 

(+) Avoids multivariate distr. modelling/normality assumption

(-) Can't use external information, i.e. dynamic factor (except if we use two-dimensional target vector)
to estimate the regime.

(?) Can we use a custom loss function, where only the density of one indicator (S&P) is considered?

# Data Source
Data set provided by Robert Shiller: [Stock market data](http://www.econ.yale.edu/~shiller/data.htm)

# Useful commands:
activate markov-switching
conda install

conda list -e > requirements.txt