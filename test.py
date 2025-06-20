import optimisation.discretisation as discretisation
import scipy.stats as stats
import data_analysis.price_tests as price_tests
import price_forecaster.lear_forecast as lear_forecast

path = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Raw Data.xlsx'
output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Regression Analysis.xlsx'

price_tests.perform_volatility_regression_analysis(
    raw_data_filepath=path,
    output_data_filepath=output_filepath
)