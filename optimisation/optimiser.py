import polars as pl
import numpy as np
import constants as ct
import auction_simulation.simulation_engine as simulation_engine

from bayes_opt import BayesianOptimization

def run_optimisation_for_day(
    date: str,
    number_of_simulations: int,
    forecast_one_ic_one_day: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    number_of_generators: int,
    risk_aversion: float,
    optimisation_tolerance: float,
    initial_random_evaluations: int,
    number_of_optimisation_iterations: int
) -> np.ndarray:
    initial_alpha = {str(i) : 0 for i in range(number_of_generators)}
    initial_beta = {str(i) : 1 for i in range(number_of_generators)}
    initial_generator_capacity = [generator_capacity/5 for _ in range(len(forecast_one_ic_one_day[ct.ColumnNames.DELIVERY_PERIOD.value]))]
    initial_capacity_bids = {str(i) : initial_generator_capacity for i in range(number_of_generators)}
    initial_capacity_bids[ct.ColumnNames.DELIVERY_PERIOD.value] = forecast_one_ic_one_day[ct.ColumnNames.DELIVERY_PERIOD.value]
    initial_capacity_bids = pl.DataFrame(initial_capacity_bids)
    converged = False
    alpha_by_generator = initial_alpha.copy()
    beta_by_generator = initial_beta.copy()
    bid_capacity_by_generator = initial_capacity_bids.clone()
    utility_by_generator = {str(i) : ct.NumericalConstants.DEFAULT_UTILITY.value for i in range(number_of_generators)}
    
    while not converged:
        for i in range(number_of_generators):
            utility = simulation_engine.run_simulations(
                date,
                number_of_simulations,
                number_of_generators,
                forecast_one_ic_one_day,
                alpha_by_generator,
                beta_by_generator,
                bid_capacity_by_generator,
                generator_marginal_cost,
                generator_capacity,
                str(i),
                risk_aversion
            )
        
            new_alpha, new_beta = optimise_strategy(
                date,
                number_of_simulations,
                number_of_generators,
                alpha_by_generator,
                beta_by_generator,
                bid_capacity_by_generator,
                forecast_one_ic_one_day,
                generator_marginal_cost,
                generator_capacity,
                str(i),
                risk_aversion,
                initial_random_evaluations,
                number_of_optimisation_iterations
            )
            
            candidate_alpha_by_generator = alpha_by_generator.copy()
            candidate_beta_by_generator = beta_by_generator.copy()
            candidate_alpha_by_generator[str(i)] = new_alpha
            candidate_beta_by_generator[str(i)] = new_beta
            
            new_utility = simulation_engine.run_simulations(
                date,
                number_of_simulations,
                number_of_generators,
                forecast_one_ic_one_day,
                candidate_alpha_by_generator,
                candidate_beta_by_generator,
                bid_capacity_by_generator,
                generator_marginal_cost,
                generator_capacity,
                str(i),
                risk_aversion
            )
            
            if new_utility > utility:
                alpha_by_generator[str(i)] = new_alpha
                beta_by_generator[str(i)] = new_beta
            
        new_utility_by_generator = simulation_engine.get_utility_by_generator(
            date,
            number_of_simulations,
            number_of_generators,
            forecast_one_ic_one_day,
            alpha_by_generator,
            beta_by_generator,
            bid_capacity_by_generator,
            generator_marginal_cost,
            generator_capacity,
            risk_aversion
        )
        
        utility_changes_by_generator = [new_utility_by_generator[str(i)] - utility_by_generator[str(i)] for i in range(number_of_generators)]
            
        if all(abs(change) < optimisation_tolerance for change in utility_changes_by_generator):
             converged = True
        else:
            utility_by_generator = new_utility_by_generator.copy()
    
    return alpha_by_generator, beta_by_generator

#For now, only optimising the values alpha and beta, assuming a fixed capacity bid into the auction. May relax this later
def objective_function(
    alpha: float,
    beta: float,
    date: str,
    number_of_simulations: int,
    number_of_generators: int,
    forecast_one_ic: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    generator_id: str,
    risk_aversion: float,
    alpha_by_generator: dict[int, float],
    beta_by_generator: dict[int, float],
    bid_capacity_by_generator : pl.DataFrame
) -> float:
    candidate_alpha_by_generator = alpha_by_generator.copy()
    candidate_beta_by_generator = beta_by_generator.copy()
    candidate_alpha_by_generator[generator_id] = alpha
    candidate_beta_by_generator[generator_id] = beta
    
    utility = simulation_engine.run_simulations(
        date,
        number_of_simulations,
        number_of_generators,
        forecast_one_ic,
        candidate_alpha_by_generator,
        candidate_beta_by_generator,
        bid_capacity_by_generator,
        generator_marginal_cost,
        generator_capacity,
        generator_id,
        risk_aversion
    )
    
    return utility  #BayesianOptimization maxmises the objective
        
def optimise_strategy(
    date: str,
    number_of_simulations: int,
    number_of_generators: int,
    alpha_by_generator: dict[str, float],
    beta_by_generator: dict[str, float],
    bid_capacity_by_generator: pl.DataFrame,
    forecast_one_ic: pl.DataFrame,
    generator_marginal_cost: float,
    generator_capacity: float,
    generator_id: str,
    risk_aversion: float,
    initial_random_evaluations: int,
    number_of_optimisation_iterations: int
) -> tuple[float, float]:
    
    pbounds = {
        'alpha': (-5, 5),
        'beta': (0, 2)
    }
    
    def bo_objective(alpha, beta):
        return objective_function(
            alpha,
            beta,
            date,
            number_of_simulations,
            number_of_generators,
            forecast_one_ic,
            generator_marginal_cost,
            generator_capacity,
            generator_id,
            risk_aversion,
            alpha_by_generator,
            beta_by_generator,
            bid_capacity_by_generator
        )
        
    optimizer = BayesianOptimization(
        f=bo_objective,
        pbounds=pbounds,
        random_state=42,
        verbose=1
    )
    
    optimizer.maximize(
        init_points=initial_random_evaluations,
        n_iter=number_of_optimisation_iterations
    )
    
    best_params = optimizer.max['params']
    best_alpha = best_params['alpha']
    best_beta = best_params['beta']
    
    return best_alpha, best_beta
        