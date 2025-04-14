import model.engine as engine

read_in_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Raw Data/One Year Test.xlsx'
rolling_window_days = 30
number_of_simulations = 1000
number_of_generators = 10
generator_marginal_cost = 40
generator_capacity = 1000
risk_aversion = 1
optimisation_tolerance = 0.2
initial_random_evaluations = 10
number_of_optimisation_iterations = 10
output_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Raw Data/One Year Test 2.xlsx'

def main():
    engine.run(
        read_in_filepath,
        rolling_window_days,
        number_of_simulations,
        number_of_generators,
        generator_marginal_cost,
        generator_capacity,
        risk_aversion,
        optimisation_tolerance,
        initial_random_evaluations,
        number_of_optimisation_iterations,
        output_filepath
    )
    
    
main()