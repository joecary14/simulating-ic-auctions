from dataclasses import dataclass

@dataclass
class AuctionParameters:
    capacity_offered: float
    central_prior_value: float
    number_of_bidders: int
    prior_standard_deviation: float
    observation_standard_deviation: float
    number_of_price_bins: int
    min_prior_spread: float
    max_prior_spread: float
    min_observation: float
    max_observation: float
    number_of_bid_price_bins: int
    number_of_bid_quantity_bins: int
    max_bid_price: float
    max_quantity_demanded: float
    number_of_auction_simulations: int
    actual_clearing_price: float
    
    
    