import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection
import price_forecaster.lear_forecast as lear_forecast
import optimisation.optimisation_engine as engine
import simulation.simulation_engine as simulation_engine
import scipy.stats as stats

demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/FR D-2 Demand Forecast.xlsx'
be_demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/BE D-7 Demand Forecast.xlsx'
price_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/All Prices.xlsx'
output_directory = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Output Data/LEAR/v1'
output_filename = '/gb_dk1_2023_2024_data.csv'
input_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/gb_dk1_data.csv'
dk1_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/DK D-1 RES Forecasts.xlsx'
elexon_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Elexon TSDF & Wind Forecast 2021-2024.xlsx'
nl_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/NL D-1 RES Forecasts.xlsx'
country_code = 'BE'

central_price_spread_estimate = 10
price_spread_stdev = 5
min_prior_spread = -5
max_prior_spread = 25 
number_of_bins = 10
min_observation = -10
max_observation = 30
max_bid_price = 20
max_total_quantity_demanded = 10
number_of_price_levels = 3
number_of_quantity_levels = 3
capacity_offered = 10
number_of_bidders = 10
number_of_mc_simulations = 100
number_of_auction_simulations = 1000
soda_tolerance = 5e-3
lp_relative_tolerance = 0.1
max_iterations = 1000

async def main():
    prior_price_spread_distribution = stats.norm(
        loc=central_price_spread_estimate, 
        scale=price_spread_stdev
    )
    simulation_engine.simulate_auction(
        prior_price_spread_distribution,
        min_prior_spread,
        max_prior_spread,
        number_of_bins,
        min_observation,
        max_observation,
        max_bid_price,
        max_total_quantity_demanded,
        number_of_price_levels,
        number_of_quantity_levels,
        capacity_offered,
        number_of_bidders,
        number_of_mc_simulations,
        number_of_auction_simulations,
        soda_tolerance,
        lp_relative_tolerance,
        max_iterations
    )
     
asyncio.run(main())