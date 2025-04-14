import polars as pl
import numpy as np
import constants as ct
import auction_simulation.day_simulation as day_simulation

def run_simulations(
    date: str,
    number_of_simulations: int,
    number_of_generators: int,
    forecast_one_ic: pl.DataFrame,
    alpha_by_generator: dict[str, float],
    beta_by_generator: dict[str, float],
    bid_capacity_by_generator: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    generator_id: int,
    risk_aversion: float
):
    daily_returns_by_sim = run_day_simulations(
        date,
        number_of_simulations,
        number_of_generators,
        forecast_one_ic,
        alpha_by_generator,
        beta_by_generator,
        bid_capacity_by_generator,
        generator_marginal_cost,
        generator_capacity,
        generator_id
    )
    
    utility = calculate_utility(
        daily_returns_by_sim,
        risk_aversion
    )
    
    return utility

def calculate_utility(
    daily_returns_by_sim: np.ndarray,
    risk_aversion: float
) -> float:
    mean_return = daily_returns_by_sim.mean()
    variance_return = daily_returns_by_sim.var()
    
    utility = mean_return - risk_aversion * variance_return
    
    return utility
    
def run_day_simulations(
    date : str,
    number_of_simulations : int,
    number_of_generators : int,
    forecast_one_ic : pl.DataFrame,
    alpha_by_generator : dict[str, float],
    beta_by_generator : dict,
    bid_capacity_by_generator : pl.DataFrame,
    generator_marginal_cost : float,
    generator_capacity : float,
    generator_id : int
) -> np.ndarray:
    
    forecast_one_day = forecast_one_ic.filter(pl.col(ct.ColumnNames.DATE.value) == date)
    covariance_matrix = day_simulation.get_covariance_matrix(forecast_one_day)
    daily_returns = []
    for i in range(number_of_simulations):
        daily_return_one_sim = day_simulation.simulate_day(
            forecast_one_day,
            covariance_matrix,
            number_of_generators,
            alpha_by_generator,
            beta_by_generator,
            bid_capacity_by_generator,
            generator_marginal_cost,
            generator_capacity,
            generator_id
        )
        daily_returns.append(daily_return_one_sim)
    
    daily_returns_array = np.array(daily_returns)
    
    return daily_returns_array