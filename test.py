import optimisation.discretisation as discretisation
import scipy.stats as stats
import data_analysis.price_tests as price_tests
import price_forecaster.lear_forecast as lear_forecast

path = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Raw Data.xlsx'
output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Regression Analysis Flat Volatility.xlsx'

dfs = price_tests.process_raw_data(path)
price_tests.fit_gpd_and_compare_aic(dfs)