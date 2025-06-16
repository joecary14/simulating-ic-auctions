import pandas as pd
import numpy as np
from simulation.auction_parameters import AuctionParameters

def assign_auction_parameters(
    read_in_filepath: str,
    number_of_price_bins: int,
    number_of_bid_price_bins: int,
    number_of_quantity_levels: int,
    number_of_auction_simulations: int
) -> list[AuctionParameters]:
    central_prices_df = pd.read_excel(read_in_filepath)
    prices = central_prices_df['outturn_price_spread']
    capacities = central_prices_df['capacities_offered']
    number_of_bidders = central_prices_df['number_of_bidders']
    actual_clearing_prices = central_prices_df['actual_clearing_price']
    default_prior_stdev = 4
    default_observation_stdev = 2
    auction_parameters = []
    for period in range(len(prices)):
        central_price = prices[period]
        capacity_offered = capacities[period]
        auction_parameter = AuctionParameters(
            capacities[period],
            central_price,
            number_of_bidders[period],
            default_prior_stdev,
            default_observation_stdev,
            number_of_price_bins,
            central_price - 3*default_prior_stdev,
            central_price + 3*default_prior_stdev,
            central_price - 4*default_observation_stdev,
            central_price + 4*default_observation_stdev,
            number_of_bid_price_bins,
            number_of_quantity_levels,
            central_price + 4*default_observation_stdev,
            capacity_offered/4,
            number_of_auction_simulations,
            actual_clearing_prices[period]
        )
        auction_parameters.append(auction_parameter)
    
    return auction_parameters