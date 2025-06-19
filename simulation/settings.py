import pandas as pd
import numpy as np
from simulation.auction_parameters import AuctionParameters

def assign_auction_parameters(
    read_in_filepath: str,
    number_of_price_bins: int,
    number_of_bid_price_bins: int,
    number_of_quantity_levels: int,
    number_of_auction_simulations: int
) -> tuple[list[AuctionParameters], list[float]]:
    central_prices_df = pd.read_excel(read_in_filepath)
    capacities = central_prices_df['capacities_offered']
    prices = central_prices_df['outturn_price_spread']
    number_of_bidders = central_prices_df['number_of_bidders']
    actual_clearing_prices = central_prices_df['actual_clearing_price']
    default_prior_stdev = 10
    default_observation_stdev = 5
    price_scale_factors = 1 / abs(prices)
    scaled_prices = prices * price_scale_factors
    scaled_prior_stdevs = default_prior_stdev * price_scale_factors
    scaled_conditional_stdevs = default_observation_stdev * price_scale_factors
    min_prior_spreads = 1 - 3 * scaled_prior_stdevs
    max_prior_spreads = 1 + 3 * scaled_prior_stdevs
    min_observations = min_prior_spreads - scaled_conditional_stdevs
    max_observations = max_prior_spreads + scaled_conditional_stdevs
    max_bids = np.maximum(max_observations, 0)
    scaled_capacities = [1 if capacity != 0 else 0 for capacity in capacities]
    auction_parameters = []
    max_quantities_demanded = scaled_capacities
    for period in range(len(prices)):
        auction_parameters_one_period = AuctionParameters(
            scaled_capacities[period],
            scaled_prices[period],
            number_of_bidders[period],
            default_prior_stdev,
            default_observation_stdev,
            number_of_price_bins,
            min_prior_spreads[period],
            max_prior_spreads[period],
            min_observations[period],
            max_observations[period],
            number_of_bid_price_bins,
            number_of_quantity_levels,
            max_bids[period],
            max_quantities_demanded[period],
            number_of_auction_simulations,
            actual_clearing_prices[period]
        )
        auction_parameters.append(auction_parameters_one_period)
    
    return auction_parameters, price_scale_factors