import numpy as np
from scipy.stats import norm

import optimisation.discretisation as discretisation
import optimisation.optimiser as optimiser


#TODO - this is where we bring it all together
def run_optimisation_one_period(
    price_spread_prior_distribution,
    conditional_observation_distribution,
    min_prior_spread: float,
    max_prior_spread: float,
    number_of_value_bins: int,
    min_observation: float,
    max_observation: float,
    max_bid_price: float,
    max_bid_quantity: float,
    number_of_price_levels: int,
    number_of_quantitiy_levels: int,
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
    possible_prior_probabilities = discrete_price_spread[:, 1]
    
    possible_observations = np.linspace(
        min_observation,
        max_observation,
        number_of_value_bins
    )
    conditional_probabilities = []
    for prior_value in possible_prior_values:
        mean = prior_value
        std_dev = 1.0
        conditional_distribution = norm(loc=mean, scale=std_dev)
        discrete_conditional_distribution = discretisation.discretise_distribution(
            conditional_distribution,
            number_of_value_bins,
            min_observation,
            max_observation
        )
        possible_observation_probabilities = discrete_conditional_distribution[:, 1]
        conditional_probabilities.append(possible_observation_probabilities)
    
    conditional_probabilities_matrix = np.transpose(np.column_stack(conditional_probabilities))
    
    discrete_action_space = discretisation.generate_demand_schedules(
        max_bid_price,
        max_bid_quantity,
        number_of_price_levels,
        number_of_quantitiy_levels
    )
        
    optimiser.soda_algorithm(
        possible_prior_values,
        possible_observations,
        discrete_action_space,
        possible_prior_probabilities,
        conditional_probabilities_matrix,
        number_of_bidders,
        number_of_simulations,
        capacity_offered
    )