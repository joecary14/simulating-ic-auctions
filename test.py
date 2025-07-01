import optimisation.discretisation as discretisation
import scipy.stats as stats
import data_analysis.price_tests as price_tests
import price_forecaster.lear_forecast as lear_forecast
import price_forecaster.data_collection_v5 as data_collection_v5

path = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Raw Data.xlsx'
output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Regression Analysis Flat Volatility with Seasonality.xlsx'
be_v2_input = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/v2 Inputs/gb_be_2023_2024_data.csv'
lear_output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Output Data/LEAR/v5'
dk1_v4_input = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/v4 Inputs/gb_dk1_2023_2024_data.csv'
elexon_actual_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Elexon INDO & Wind Hour Ahead Forecast.xlsx'
prices = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Prices.xlsx'
inputs = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/v4 Inputs/gb_be_2023_2024_data.csv'


lear_forecast.run_lear_forecast(
    inputs,
    364,
    '2024-01-01',
    '2024-07-01',
    'BE',
    '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Output Data/LEAR/v5'
)