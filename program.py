import asyncio
import model.engine as engine
import price_forecaster.data_collection_v1 as data_collection_v1
import price_forecaster.data_collection_v2 as data_collection_v2
import price_forecaster.data_collection_v3 as data_collection_v3
import price_forecaster.lear_forecast as lear_forecast
import optimisation.optimisation_engine as engine
import simulation.simulation_engine as simulation_engine

from optimisation.solver_parameters import SolverParameters

demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/BE ATL.xlsx'
be_demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/BE ATL.xlsx'
price_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Prices.xlsx'
output_directory = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Output Data/LEAR/v3'
output_filename = '/gb_be_2023_2024_data.csv'
fr_lear_input_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/v2 Inputs/gb_fr_2023_2024_data.csv'
dk1_lear_input_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/v2 Inputs/gb_dk1_2023_2024_data.csv'
be_lear_input_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/v3 Inputs/gb_be_2023_2024_data.csv'
dk1_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/DK RES Production.xlsx'
elexon_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Elexon INDO & Wind Hour Ahead Forecast.xlsx'
nl_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/NL D-1 RES Forecasts.xlsx'
country_code = 'DK1'
fr_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/FR ATL - Nuclear.xlsx'
years = [2023, 2024]

number_of_price_bins = 10
number_of_price_levels = 10
number_of_quantity_levels = 1
number_of_mc_simulations = 100
number_of_auction_simulations = 1000
soda_tolerance = 1e-4
lp_relative_tolerance = 0.01
max_iterations = 1000
read_in_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Code Testing/Auction Simulation Testing/24-1-24 Forecast Test.xlsx'
output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Code Testing/Auction Simulation Testing/24-1-24 Forecast Test Results.xlsx'

async def main():
    solver_params = SolverParameters(
        number_of_mc_simulations,
        soda_tolerance,
        lp_relative_tolerance,
        max_iterations
    )
    simulation_engine.simulate_auction(
        read_in_filepath=read_in_filepath,
        number_of_price_bins=number_of_price_bins,
        number_of_bid_price_bins=number_of_price_levels,
        number_of_bid_quantity_bins=number_of_quantity_levels,
        number_of_auction_simulations=number_of_auction_simulations,
        solver_parameters=solver_params,
        write_out_filepath=output_filepath
    )
     
asyncio.run(main())