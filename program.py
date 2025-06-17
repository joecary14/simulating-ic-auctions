import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection
import price_forecaster.lear_forecast as lear_forecast
import optimisation.optimisation_engine as engine
import simulation.simulation_engine as simulation_engine

from optimisation.solver_parameters import SolverParameters

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

number_of_price_bins = 12
number_of_price_levels = 3
number_of_quantity_levels = 3
number_of_mc_simulations = 100
number_of_auction_simulations = 1000
soda_tolerance = 5e-3
lp_relative_tolerance = 0.1
max_iterations = 1000
read_in_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Code Testing/Auction Simulation Testing/27-1-24 Forecast Test.xlsx'
output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Code Testing/Auction Simulation Testing/27-1-24 Forecast Test Results.xlsx'

async def main():
    solver_parameters = SolverParameters(
        number_of_mc_simulations,
        soda_tolerance,
        lp_relative_tolerance,
        max_iterations
    )
    simulation_engine.simulate_auction(
        read_in_filepath,
        number_of_price_bins,
        number_of_price_levels,
        number_of_quantity_levels,
        number_of_auction_simulations,
        solver_parameters,
        output_filepath
    )
     
asyncio.run(main())