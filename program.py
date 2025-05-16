import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection

read_in_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Raw Data/One Year Test.xlsx'
rolling_window_days = 30
number_of_simulations = 10
number_of_generators = 10
generator_marginal_cost = 40
generator_capacity = 1000
risk_aversion = 1
optimisation_tolerance = 0.1
initial_random_evaluations = 10
number_of_optimisation_iterations = 10
output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Code Testing/BO Test.xlsx'

async def main():
    await data_collection.get_elexon_lear_data_for_year(2023)
     
asyncio.run(main())