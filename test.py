import optimisation.discretisation as discretisation
import scipy.stats as stats
import data_analysis.price_tests as price_tests
import price_forecaster.lear_forecast as lear_forecast

path = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Raw Data.xlsx'

dfs = price_tests.process_raw_data(path)
price_tests.generate_qq_plots(dfs)

lear_forecast.run_lear_forecast(
    '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/gb_dk1_data.csv',
    364,
    '2024-01-01',
    '2024-01-31',
    'DK1',
    '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Output Data/LEAR'
)