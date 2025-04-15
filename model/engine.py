import data_handler.excel_interaction as excel_interaction
import price_forecaster.naive_forecast as naive
import optimisation.optimisation_engine as optimisation_engine

def run(
    read_in_filepath : str,
    rolling_window_days : int,
    number_of_simulations : int,
    number_of_generators : int,
    generator_marginal_cost : float,
    generator_capacity : float,
    risk_aversion : float,
    optimisation_tolerance : float,
    initial_random_evaluations : int,
    number_of_optimisation_iterations : int,
    output_filepath : str
) -> None:
    raw_data_dfs = excel_interaction.read_in_excel_data(read_in_filepath)
    naive_forecasts = naive.get_naive_forecasts(
        raw_data_dfs,
        rolling_window_days
    )
    clearing_prices_by_ic = {}
    for str, naive_forecast in naive_forecasts.items():
        clearing_prices = optimisation_engine.run_optimisation(
            number_of_simulations,
            number_of_generators,
            naive_forecast,
            generator_marginal_cost,
            generator_capacity,
            risk_aversion,
            optimisation_tolerance,
            initial_random_evaluations,
            number_of_optimisation_iterations
        )
        clearing_prices_by_ic[str] = clearing_prices
        print(f"Clearing prices for {str} calculated.")
    
    excel_interaction.write_data_to_excel(
        clearing_prices_by_ic,
        output_filepath
    )
    
    