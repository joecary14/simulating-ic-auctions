import numpy as np
import pandas as pd
import optimisation.auction as auction
import optimisation.optimisation_engine as optimisation_engine
from optimisation.solver_parameters import SolverParameters
from simulation.settings import assign_auction_parameters

def simulate_auction(
    read_in_filepath: str,
    number_of_price_bins: int,
    number_of_bid_price_bins: int,
    number_of_bid_quantity_bins: int,
    number_of_auction_simulations: int,
    solver_parameters: SolverParameters,
    write_out_filepath: str
) -> None:
    auction_parameters_by_period, price_scale_factors = assign_auction_parameters(
        read_in_filepath,
        number_of_price_bins,
        number_of_bid_price_bins,
        number_of_bid_quantity_bins,
        number_of_auction_simulations
    )
    simulated_clearing_prices = []
    utility_losses = []
    for period_index, auction_parameters in enumerate(auction_parameters_by_period):    
        if auction_parameters.max_bid_price == 0 or auction_parameters.capacity_offered == 0:
            print(f"Skipping period {period_index + 1} due to zero max bid price or capacity offered.")
            simulated_clearing_prices.append(0.0)
            utility_losses.append(0.0)
            continue    
        conditional_sigma, prior_value_probabilities, conditional_observation_probabilities, possible_demand_schedules, final_utility_loss = optimisation_engine.run_optimisation_one_period(
            auction_parameters,
            solver_parameters
        )
        
        simulated_clearing_price = simulate_auction_clearing(
            price_scale_factors[period_index],
            auction_parameters.number_of_auction_simulations,
            conditional_sigma,
            prior_value_probabilities,
            conditional_observation_probabilities,
            possible_demand_schedules,
            auction_parameters.number_of_bidders,
            auction_parameters.capacity_offered
        )
        print(f"Clearing Price: {simulated_clearing_price}")
        print(f"Utility Loss: {final_utility_loss}")
        print(f"Period {period_index + 1} of {len(auction_parameters_by_period)}")
        simulated_clearing_prices.append(simulated_clearing_price)
        utility_losses.append(final_utility_loss)
        
    outturn_price_spreads = [auction_parameters.central_prior_value for auction_parameters in auction_parameters_by_period]
    actual_clearing_prices = [auction_parameters.actual_clearing_price for auction_parameters in auction_parameters_by_period]
    results_df = pd.DataFrame({
        'simulated_clearing_price': simulated_clearing_prices,
        'actual_clearing_price': actual_clearing_prices,
        'outturn_price_spread': outturn_price_spreads,
        'utility_loss': utility_losses
    })
    
    results_df.to_excel(write_out_filepath, index=False)

def simulate_auction_clearing(
    price_scale_factor: float,
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
    rescaled_average_clearing_price = average_clearing_price/price_scale_factor
    return rescaled_average_clearing_price