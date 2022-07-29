# markov-regime-switching
Modelling regime switching index price movements with markov chains. 

# Usage
Donwload the data specified in [Data Source](#Data-Source) to some folder.
Go to the [configuration file](config.py) and set `data_dir` to that folder.


# Data Source (with Great Depression)
- S&P data set by Robert Shiller: [Website](http://www.econ.yale.edu/~shiller/data.htm),
[Download](http://www.econ.yale.edu/~shiller/data/ie_data.xls). **The stock price data are monthly averages of daily closing prices.**
- Industrial Production Total Index: [Website](https://fred.stlouisfed.org/series/INDPRO), downloaded as `.csv`.
- Moody's Seasoned Baa Corporate Bond Yield (BAA) [Website](https://fred.stlouisfed.org/series/BAA)
- Moody's Seasoned Aaa Corporate Bond Yield (AAA) [Website](https://fred.stlouisfed.org/series/AAA)
- Producer Price Index by Commodity: All Commodities (PPIACO)

# Useful commands:
activate markov-switching
conda install

conda list -e > requirements.txt