import polars as pl
import numpy as np
import constants as ct
import auction_simulation.simulation_engine as simulation_engine

from scipy.optimize import minimize

def run_optimisation_for_day(
    date: str,
    number_of_simulations: int,
    forecast_one_ic_one_day: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    number_of_generators: int,
    risk_aversion: float,
    optimisation_tolerance: float
) -> np.ndarray:
    initial_alpha = {i : 0 for i in range(number_of_generators)}
    initial_beta = {i : 1 for i in range(number_of_generators)}
    periods_in_day = forecast_one_ic_one_day[ct.ColumnNames.DELIVERY_PERIOD.value].unique().to_list().count()
    initial_generator_capacity = [generator_capacity/10 for _ in range(periods_in_day)]
    initial_capacity_bids = pl.DataFrame({i : initial_generator_capacity for i in range(number_of_generators)})
    converged = False
    alpha_by_generator = initial_alpha.copy()
    beta_by_generator = initial_beta.copy()
    bid_capacity_by_generator = initial_capacity_bids.copy()
    
    while not converged:
        utility_changes = []
        for i in range(number_of_generators):
            alpha = alpha_by_generator[i]
            beta = beta_by_generator[i]
            bid_capacity = bid_capacity_by_generator[i]
            
            utility = simulation_engine.run_simulations(
                date,
                number_of_simulations,
                forecast_one_ic_one_day,
                alpha_by_generator,
                beta_by_generator,
                bid_capacity_by_generator,
                generator_marginal_cost,
                generator_capacity,
                i,
                risk_aversion
            )
        
            new_strategy = optimise_strategy(
                date,
                number_of_simulations,
                alpha_by_generator,
                beta_by_generator,
                bid_capacity_by_generator,
                forecast_one_ic_one_day,
                generator_marginal_cost,
                generator_capacity,
                i,
                risk_aversion
            )
            
            candidate_alpha_by_generator = alpha_by_generator.copy()
            candidate_beta_by_generator = beta_by_generator.copy()
            candidate_bid_capacity_by_generator = bid_capacity_by_generator
            candidate_alpha_by_generator[i] = new_strategy[0]
            candidate_beta_by_generator[i] = new_strategy[1]
            candidate_bid_capacity_by_generator[i] = new_strategy[2]
            
            new_utility = simulation_engine.run_simulations(
                date,
                number_of_simulations,
                forecast_one_ic_one_day,
                candidate_alpha_by_generator,
                candidate_beta_by_generator,
                candidate_bid_capacity_by_generator,
                generator_marginal_cost,
                generator_capacity,
                i,
                risk_aversion
            )
            
            if new_utility > utility:
                alpha_by_generator[i] = new_strategy[0]
                beta_by_generator[i] = new_strategy[1]
                bid_capacity_by_generator[i] = new_strategy[2]
            
            utility_changes.append(utility)

        if all(change < optimisation_tolerance for change in utility_changes):
            converged = True
    
    return alpha_by_generator, beta_by_generator, bid_capacity_by_generator

def objective_function(
    strategy_vector: np.ndarray,
    date: str,
    number_of_simulations: int,
    forecast_one_ic: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    generator_id: int,
    risk_aversion: float,
    alpha_by_generator: dict[int, float],
    beta_by_generator: dict[int, float],
    bid_capacity_by_generator : pl.DataFrame
) -> float:
    candidate_alpha_by_generator = alpha_by_generator.copy()
    candidate_beta_by_generator = beta_by_generator.copy()
    candidate_bid_capacity_by_generator = bid_capacity_by_generator.copy()
    candidate_alpha_by_generator[generator_id] = strategy_vector[0]
    candidate_beta_by_generator[generator_id] = strategy_vector[1]
    candidate_bid_capacity_by_generator[generator_id] = strategy_vector[2:]
    
    utility = simulation_engine.run_simulations(
        date,
        number_of_simulations,
        forecast_one_ic,
        candidate_alpha_by_generator,
        candidate_beta_by_generator,
        candidate_bid_capacity_by_generator,
        generator_marginal_cost,
        generator_capacity,
        generator_id,
        risk_aversion
    )
    
    return -utility  # Minimize the negative utility to maximize the utility
        
def optimise_strategy(
    date: str,
    number_of_simulations: int,
    alpha_by_generator: dict[int, float],
    beta_by_generator: dict[int, float],
    bid_capacity_by_generator: pl.DataFrame,
    forecast_one_ic: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    generator_id: int,
    risk_aversion: float,
) -> np.ndarray:
    initial_strategy = (
        alpha_by_generator[generator_id],
        beta_by_generator[generator_id],
        bid_capacity_by_generator[generator_id].to_numpy()
    )
    
    result = minimize(
        objective_function,
        x0 = initial_strategy,
        args=(date, number_of_simulations, forecast_one_ic, generator_marginal_cost, generator_capacity, generator_id, risk_aversion),
        method="Nelder-Mead"
    )
    
    if not result.success:
        return None
    else:
        return result.x
        