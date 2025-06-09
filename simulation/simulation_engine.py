import numpy as np
import optimisation.auction as auction
import optimisation.optimiser as optimiser

def simulate_auction_clearing(
    number_of_outturn_values_to_draw: int,
    optimum_distributional_strategies: np.ndarray,
    possible_prior_values: np.ndarray,
    possible_prior_probabilities: np.ndarray,
    possible_observations: np.ndarray,
    possible_observation_probabilities: np.ndarray,
    bid_schedules: list[list[tuple[float, float]]],
    number_of_bidders: int,
    capacity_offered: float
) -> float:
    total_clearing_price = 0.0
    for i in range(number_of_outturn_values_to_draw):
        # Simulate the auction clearing process here
        # For now, we will just use a random clearing price
        outturn_value_index = optimiser.sample_index(possible_prior_probabilities)
        outturn_value = possible_prior_values[outturn_value_index]
        bid_schedules_by_participant = {bidder: [] for bidder in range(number_of_bidders)}
        for bidder in range(number_of_bidders):
            observation_index = optimiser.sample_index(possible_observation_probabilities[outturn_value_index])
            bid_schedule_probabilities = optimum_distributional_strategies[observation_index, :]
            bid_index = optimiser.sample_index(bid_schedule_probabilities)
            bid_schedule = bid_schedules[bid_index]
            bid_schedules_by_participant[bidder] = bid_schedule
        
        clearing_price, allocations_by_participant = auction.clear_auction(
            bid_schedules_by_participant,
            capacity_offered
        )
        total_clearing_price += clearing_price
    
    average_clearing_price = total_clearing_price / number_of_outturn_values_to_draw
    return average_clearing_price