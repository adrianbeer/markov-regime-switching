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

# Usage
1. Download the data specified in [Data Source](#Data-Source) to some folder.
2. Copy [the sample config file](config-sample.py) to `config.py` and set value appropriately.


# Data Sources (with Great Depression)
- S&P data set by Robert Shiller: [Website](http://www.econ.yale.edu/~shiller/data.htm),
[Download](http://www.econ.yale.edu/~shiller/data/ie_data.xls). **The stock price data are monthly averages of daily closing prices.**
## Fred Data Sources
All FRED data sources were downloaded in `.csv` format.
- Industrial Production Total Index: [Website](https://fred.stlouisfed.org/series/INDPRO), downloaded as `.csv`.
- Moody's Seasoned Baa Corporate Bond Yield (BAA) [Website](https://fred.stlouisfed.org/series/BAA)
- Moody's Seasoned Aaa Corporate Bond Yield (AAA) [Website](https://fred.stlouisfed.org/series/AAA)
- Producer Price Index by Commodity: All Commodities (PPIACO)

# Useful commands:
activate markov-switching
conda install

conda list -e > requirements.txt
