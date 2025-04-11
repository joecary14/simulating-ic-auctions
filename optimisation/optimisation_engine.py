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
    optimisation_tolerance: float
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
            optimisation_tolerance
        )
        delivery_periods = forecast_one_ic[ct.ColumnNames.DELIVERY_PERIOD.value]
        clearing_prices_by_day.append(clearing_prices)
        
        clearing_prices_df = pl.DataFrame(
            {
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
) -> np.ndarray:
    br_alpha_by_generator, br_beta_by_generator, br_bid_capacity_by_generator = optimiser.run_optimisation_for_day(
        date,
        number_of_simulations,
        forecast_one_ic,
        generator_marginal_cost,
        generator_capacity,
        number_of_generators,
        risk_aversion,
        optimisation_tolerance
    )
    
    covariance_matrix_by_period = day_simulation.get_covariance_matrix_by_period(forecast_one_ic)
    auction_information_one_day = day_simulation.get_auction_information_one_sim(
        forecast_one_ic,
        covariance_matrix_by_period,
        number_of_generators,
        br_alpha_by_generator,
        br_beta_by_generator,
        br_bid_capacity_by_generator,
        generator_marginal_cost,
    )
    
    auction_results, clearing_prices = auction_information_one_day.run_auction()
    
    return clearing_prices