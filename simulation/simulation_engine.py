import numpy as np
import optimisation.auction as auction
import optimisation.optimisation_engine as optimisation_engine

def simulate_auction(
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
    number_of_mc_simulations: int,
    number_of_auction_simulations: int,
    soda_tolerance: float,
    lp_relative_tolerance: float,
    max_iterations: int
) -> float:
    conditional_sigma, prior_value_probabilities, conditional_observation_probabilities, possible_demand_schedules = optimisation_engine.run_optimisation_one_period(
        price_spread_prior_distribution,
        min_prior_spread,
        max_prior_spread,
        number_of_value_bins,
        min_observation,
        max_observation,
        max_bid_price,
        max_total_quantity_demanded,
        number_of_price_levels,
        number_of_quantity_levels,
        capacity_offered,
        number_of_bidders,
        number_of_mc_simulations,
        soda_tolerance,
        lp_relative_tolerance,
        max_iterations
    )
    
    simulated_clearing_price = simulate_auction_clearing(
        number_of_auction_simulations,
        conditional_sigma,
        prior_value_probabilities,
        conditional_observation_probabilities,
        possible_demand_schedules,
        number_of_bidders,
        capacity_offered
    )
    print(f"Clearing Price: {simulated_clearing_price}")
    return simulated_clearing_price

def simulate_auction_clearing(
    number_of_outturn_values_to_draw: int,
    conditional_distributional_strategies: np.ndarray,
    prior_value_probabilities: np.ndarray,
    conditional_observation_probabilities: np.ndarray,
    bid_schedules: tuple[tuple[tuple[float, float], ...]],
    number_of_bidders: int,
    capacity_offered: float
) -> float:
    total_clearing_price = 0.0
    number_of_possible_observations = conditional_observation_probabilities.shape[0]
    number_of_possible_values = len(prior_value_probabilities)
    number_of_possible_bid_schedules = len(bid_schedules)
    for i in range(number_of_outturn_values_to_draw):
        # Simulate the auction clearing process here
        # For now, we will just use a random clearing price
        outturn_value_index = np.random.choice(
            number_of_possible_values,
            p=prior_value_probabilities)
        bid_schedules_by_participant = []
        for bidder in range(number_of_bidders):
            observation_index = np.random.choice(
                number_of_possible_observations,
                p=conditional_observation_probabilities[:, outturn_value_index])
            bid_schedule_probabilities = conditional_distributional_strategies[observation_index, :]
            bid_index = np.random.choice(
                number_of_possible_bid_schedules,
                p=bid_schedule_probabilities)
            bid_schedule = bid_schedules[bid_index]
            bid_schedules_by_participant.append(bid_schedule)
        
        clearing_price, allocations_by_participant = auction.clear_auction(
            bid_schedules_by_participant,
            capacity_offered
        )
        total_clearing_price += clearing_price
    
    average_clearing_price = total_clearing_price / number_of_outturn_values_to_draw
    return average_clearing_price