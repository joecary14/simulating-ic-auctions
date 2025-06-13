import numpy as np
from scipy.stats import norm

import optimisation.discretisation as discretisation
import optimisation.optimiser as optimiser
import optimisation.visualisation as visualisation
import optimisation.validation as validation

def run_optimisation_one_period(
    price_spread_prior_distribution,
    min_prior_spread: float,
    max_prior_spread: float,
    number_of_value_bins: int,
    min_observation: float,
    max_observation: float,
    max_bid_price: float,
    max_total_quantity_demanded: float,
    number_of_price_levels: int,
    number_of_quantity_levels: int,
    capacity_offered: float,
    number_of_bidders: int,
    number_of_simulations: int
):
    discrete_price_spread = discretisation.discretise_distribution(
        price_spread_prior_distribution,
        number_of_value_bins,
        min_prior_spread,
        max_prior_spread
    )
    possible_prior_values = discrete_price_spread[:, 0]
    prior_value_probabilities = discrete_price_spread[:, 1]
    
    possible_observations = np.linspace(
        min_observation,
        max_observation,
        number_of_value_bins
    )
    conditional_probabilities = []
    for prior_value in possible_prior_values:
        mean = prior_value
        std_dev = 1.0
        conditional_distribution = norm(loc=mean, scale=std_dev) #Assume normal for now
        discrete_conditional_distribution = discretisation.discretise_distribution(
            conditional_distribution,
            number_of_value_bins,
            min_observation,
            max_observation
        )
        possible_observation_probabilities = discrete_conditional_distribution[:, 1]
        conditional_probabilities.append(possible_observation_probabilities)
    
    conditional_observation_probabilities = np.column_stack(conditional_probabilities)
    marginal_observation_probabilities = conditional_observation_probabilities @ prior_value_probabilities  # shape (L,)
    marginal_observation_probabilities /= np.sum(marginal_observation_probabilities)
    posterior_value_probabilities = (conditional_observation_probabilities * prior_value_probabilities[None, :]).T / marginal_observation_probabilities[None, :]  # shape (K, L)
    
    possible_demand_schedules = discretisation.generate_demand_schedules(
        max_bid_price,
        max_total_quantity_demanded,
        number_of_price_levels,
        number_of_quantity_levels
    )
        
    conditional_sigma = solve_for_equilibrium(
        possible_prior_values,
        possible_observations,
        possible_demand_schedules,
        marginal_observation_probabilities,
        conditional_observation_probabilities,
        posterior_value_probabilities,
        number_of_bidders,
        number_of_simulations,
        capacity_offered
    )    
    
    visualisation.visualise_as_heatmap(conditional_sigma)
    
    return conditional_sigma, prior_value_probabilities, conditional_observation_probabilities, possible_demand_schedules
    
def solve_for_equilibrium(
    possible_prior_values: np.ndarray,
    possible_observations: np.ndarray,
    possible_demand_schedules: tuple[tuple[tuple[float, float], ...]],
    marginal_observation_probabilities: np.ndarray,
    conditional_observation_probabilities: np.ndarray,
    posterior_value_probabilities: np.ndarray,
    number_of_bidders: int,
    number_of_simulations: int,
    capacity_offered: float,
    number_of_attempts: int = 10,
    max_iter: int = 1000000,
    lp_tol: float = 1
) -> np.ndarray:
    current_conditional_sigma = None
    current_dual_variables = None
    current_start_index = 0
    
    for attempt in range(number_of_attempts):
        print(f"Attempt {attempt + 1} of {number_of_attempts}")
        new_conditional_sigma, new_dual_variables, last_iter = optimiser.soda_algorithm(
            possible_prior_values,
            possible_observations,
            possible_demand_schedules,
            marginal_observation_probabilities,
            conditional_observation_probabilities,
            posterior_value_probabilities,
            number_of_bidders,
            number_of_simulations,
            capacity_offered,
            input_conditional_sigma=current_conditional_sigma,
            input_dual_variables=current_dual_variables,
            start_iteration_number=current_start_index,
            max_iter=max_iter
        )
    
        utility_loss = validation.check_for_equilibrium(
            new_conditional_sigma,
            marginal_observation_probabilities,
            conditional_observation_probabilities,
            posterior_value_probabilities,
            possible_demand_schedules,
            len(possible_prior_values),
            tuple(possible_prior_values),
            number_of_simulations,
            number_of_bidders,
            capacity_offered
        )
        
        if utility_loss <= lp_tol:
            print(f"Convergence achieved in {last_iter} iterations with utility loss: {utility_loss}")
            return new_conditional_sigma
        else:
            print(f"Utility loss {utility_loss} exceeds tolerance {lp_tol}. Retrying...")
            current_conditional_sigma = new_conditional_sigma
            current_dual_variables = new_dual_variables
            current_start_index = last_iter + 1
    
    print("Failed to converge within the specified number of attempts.")
    if current_conditional_sigma is not None:
        return current_conditional_sigma
    else:
        shape = (len(possible_observations), len(possible_demand_schedules))
        return np.zeros(shape)