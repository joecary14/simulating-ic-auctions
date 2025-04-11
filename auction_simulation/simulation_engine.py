import polars as pl
import constants as ct
import auction_simulation.day_simulation as day_simulation

def run_simulations(
    date: str,
    number_of_simulations: int,
    forecast_one_ic: pl.DataFrame,
    alpha_by_generator: pl.DataFrame,
    beta_by_generator: pl.DataFrame,
    bid_capacity_by_generator: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    generator_id: int,
    risk_aversion: float
):
    profits_by_sim = run_day_simulations(
        date,
        number_of_simulations,
        forecast_one_ic,
        alpha_by_generator,
        beta_by_generator,
        bid_capacity_by_generator,
        generator_marginal_cost,
        generator_capacity
    )
    
    utility = calculate_utility(
        profits_by_sim,
        risk_aversion,
        generator_id
    )
    
    return utility

def calculate_utility(
    day_simulation_results: pl.DataFrame,
    risk_aversion: float,
    generator_id: int
) -> float:
    generator_results = day_simulation_results[generator_id].to_numpy()
    mean_profit = generator_results.mean()
    variance_profit = generator_results.var()
    
    utility = mean_profit - risk_aversion * variance_profit
    
    return utility
    
def run_day_simulations(
    date : str,
    number_of_simulations : int,
    forecast_one_ic : pl.DataFrame,
    alpha_by_generator : pl.DataFrame,
    beta_by_generator : pl.DataFrame,
    bid_capacity_by_generator : pl.DataFrame,
    generator_marginal_cost : float,
    generator_capacity : float,
) -> pl.DataFrame:
    
    forecast_one_day = forecast_one_ic.filter(pl.col(ct.ColumnNames.DATE.value) == date)
    covariance_matrix_by_period = day_simulation.get_covariance_matrix_by_period(forecast_one_day)
    profits = []
    for i in range(number_of_simulations):
        profits_by_generator = day_simulation.simulate_day(
            forecast_one_day,
            covariance_matrix_by_period,
            number_of_simulations,
            alpha_by_generator,
            beta_by_generator,
            bid_capacity_by_generator,
            generator_marginal_cost,
            generator_capacity
        )
        profits.append(profits_by_generator)
    
    profits_df = pl.concat(profits)
    
    return profits_df