import numpy as np
from scipy.stats import norm
from time import time

import optimisation.discretisation as discretisation
import optimisation.optimiser as optimiser
import optimisation.visualisation as visualisation
import optimisation.validation as validation

from simulation.auction_parameters import AuctionParameters
from optimisation.solver_parameters import SolverParameters

def run_optimisation_one_period(
    auction_parameters: AuctionParameters,
    solver_parameters: SolverParameters
):
    price_spread_prior_distribution = norm(auction_parameters.central_prior_value, auction_parameters.prior_standard_deviation) #Assume normal for now
    start_time = time()
    discrete_price_spread = discretisation.discretise_distribution(
        price_spread_prior_distribution,
        auction_parameters.number_of_price_bins,
        auction_parameters.min_prior_spread,
        auction_parameters.max_prior_spread
    )
    possible_prior_values = discrete_price_spread[:, 0]
    prior_value_probabilities = discrete_price_spread[:, 1]
    
    possible_observations = np.linspace(
        auction_parameters.min_observation,
        auction_parameters.max_observation,
        auction_parameters.number_of_price_bins
    )
    conditional_probabilities = []
    for prior_value in possible_prior_values:
        mean = prior_value
        conditional_distribution = norm(loc=mean, scale=auction_parameters.observation_standard_deviation) #Assume normal for now
        discrete_conditional_distribution = discretisation.discretise_distribution(
            conditional_distribution,
            auction_parameters.number_of_price_bins,
            auction_parameters.min_observation,
            auction_parameters.max_observation
        )
        possible_observation_probabilities = discrete_conditional_distribution[:, 1]
        conditional_probabilities.append(possible_observation_probabilities)
    
    conditional_observation_probabilities = np.column_stack(conditional_probabilities)
    marginal_observation_probabilities = conditional_observation_probabilities @ prior_value_probabilities  # shape (L,)
    marginal_observation_probabilities /= np.sum(marginal_observation_probabilities)
    posterior_value_probabilities = (conditional_observation_probabilities * prior_value_probabilities[None, :]).T / marginal_observation_probabilities[None, :]  # shape (K, L)
    
    possible_demand_schedules = discretisation.generate_demand_schedules(
        auction_parameters.max_bid_price,
        auction_parameters.max_quantity_demanded,
        auction_parameters.number_of_bid_price_bins,
        auction_parameters.number_of_bid_quantity_bins
    )
        
    conditional_sigma, final_utility_loss = solve_for_equilibrium(
        possible_prior_values,
        possible_observations,
        possible_demand_schedules,
        marginal_observation_probabilities,
        conditional_observation_probabilities,
        posterior_value_probabilities,
        auction_parameters.number_of_bidders,
        solver_parameters.number_of_monte_carlo_simulations,
        auction_parameters.capacity_offered,
        solver_parameters.optimisation_algorithm_tolerance,
        solver_parameters.utility_loss_tolerance,
        solver_parameters.max_iterations
    )    
    
    # visualisation.visualise_as_heatmap(conditional_sigma)
    end_time = time()
    
    print(f"Optimisation completed in {end_time - start_time:.2f} seconds.")
    print(f"K = {auction_parameters.number_of_price_bins}, L = {auction_parameters.number_of_price_bins}, M = {len(possible_demand_schedules)}")
    
    return conditional_sigma, prior_value_probabilities, conditional_observation_probabilities, possible_demand_schedules, final_utility_loss
    
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
    soda_tolerance: float,
    lp_relative_tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    current_conditional_sigma = None
    current_dual_variables = None
    current_start_index = 0
    final_utility_loss = 0
    
    while current_start_index < max_iter - 1:
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
            max_iter=max_iter,
            tol=soda_tolerance
        )
    
        utility_loss = validation.calculate_utility_loss(
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
        
        if utility_loss <= lp_relative_tolerance:
            print(f"Convergence achieved in {last_iter} iterations with utility loss: {utility_loss}")
            final_utility_loss = utility_loss
            break
        else:
            print(f"Utility loss {utility_loss} exceeds tolerance {lp_relative_tolerance}. Retrying...")
            current_conditional_sigma = new_conditional_sigma
            current_dual_variables = new_dual_variables
            current_start_index = last_iter + 1
            final_utility_loss = utility_loss
    
    if current_conditional_sigma is None:
        shape = (len(possible_observations), len(possible_demand_schedules))
        current_conditional_sigma = np.zeros(shape)
    
    return current_conditional_sigma, final_utility_loss