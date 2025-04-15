import numpy as np
import polars as pl
import constants as ct
import optimisation.optimiser as optimiser
import auction_simulation.day_simulation as day_simulation

def run_optimisation(
    number_of_simulations: int,
    number_of_generators: int,
    forecasts: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    risk_aversion: float,
    optimisation_tolerance: float,
    initial_random_evaluations: int,
    number_of_optimisation_iterations: int
) -> pl.DataFrame:
    clearing_prices_by_day = []
    for date in forecasts[ct.ColumnNames.DATE.value].unique():
        forecast_one_ic = forecasts.filter(pl.col(ct.ColumnNames.DATE.value) == date)
        clearing_prices = get_results_one_day(
            date,
            number_of_simulations,
            number_of_generators,
            forecast_one_ic,
            generator_marginal_cost,
            generator_capacity,
            risk_aversion,
            optimisation_tolerance,
            initial_random_evaluations,
            number_of_optimisation_iterations
        )
        delivery_periods = forecast_one_ic[ct.ColumnNames.DELIVERY_PERIOD.value]
        clearing_prices_by_day.append(clearing_prices)
        print(f"Clearing prices for {date} calculated.")
        
        clearing_prices_df = pl.DataFrame(
            {
                ct.ColumnNames.DATE.value: [date] * len(clearing_prices),
                ct.ColumnNames.DELIVERY_PERIOD.value: delivery_periods,
                ct.ColumnNames.CLEARING_PRICE.value: clearing_prices
            }
        )
        
        clearing_prices_by_day.append(clearing_prices_df)
    
    clearing_prices_df = pl.concat(clearing_prices_by_day)
    
    return clearing_prices_df  

def get_results_one_day(
    date: str,
    number_of_simulations: int,
    number_of_generators: int,
    forecast_one_ic: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    risk_aversion: float,
    optimisation_tolerance: float,
    initial_random_evaluations: int,
    number_of_optimisation_iterations: int
) -> np.ndarray:
    br_alpha_by_generator, br_beta_by_generator = optimiser.run_optimisation_for_day(
        date,
        number_of_simulations,
        forecast_one_ic,
        generator_marginal_cost,
        generator_capacity,
        number_of_generators,
        risk_aversion,
        optimisation_tolerance,
        initial_random_evaluations,
        number_of_optimisation_iterations
    )
    
    covariance_matrix_by_period = day_simulation.get_covariance_matrix_from_df(forecast_one_ic)
    initial_generator_capacity = [generator_capacity/5 for _ in range(len(forecast_one_ic[ct.ColumnNames.DELIVERY_PERIOD.value]))]
    initial_capacity_bids = {str(i) : initial_generator_capacity for i in range(number_of_generators)}
    initial_capacity_bids[ct.ColumnNames.DELIVERY_PERIOD.value] = forecast_one_ic[ct.ColumnNames.DELIVERY_PERIOD.value]
    initial_capacity_bids = pl.DataFrame(initial_capacity_bids)
    auction_information_one_day = day_simulation.get_auction_information_one_sim(
        forecast_one_ic,
        covariance_matrix_by_period,
        number_of_generators,
        br_alpha_by_generator,
        br_beta_by_generator,
        initial_capacity_bids,
        generator_marginal_cost,
    )
    
    auction_results, clearing_prices = auction_information_one_day.run_auction()
    
    return clearing_prices