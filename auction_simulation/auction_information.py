import numpy as np
import polars as pl
import bid_information as bid_info

class AuctionInformation:
    def __init__(
        self, 
        actual_domestic_prices : np.ndarray, 
        actual_foreign_prices : np.ndarray,
        bids_by_generator_by_period : dict[str, np.ndarray],
        capacity_by_generator_by_period : dict[str, np.ndarray],
        capacity_offered : np.ndarray
    ):
        self.actual_domestic_prices = actual_domestic_prices
        self.actual_foreign_prices = actual_foreign_prices
        self.bids_by_generator_by_period = bids_by_generator_by_period
        self.capacity_by_generator_by_period = capacity_by_generator_by_period
        self.capacity_offered = capacity_offered
        
    def run_auction(
        self
    ):
        results = []
        clearing_prices = []

        for period in range(len(self.capacity_offered)):
            bids_by_generator = self.bids_by_generator_by_period[period]
            capacity_by_generator = self.capacity_by_generator_by_period[period]
            capacity_offered = self.capacity_offered[period]

            accepted_bids, clearing_price = self.run_auction_one_period(
            bids_by_generator,
            capacity_offered,
            capacity_by_generator
            )

            results.append(accepted_bids)
            clearing_prices.append(clearing_price)

        accepted_capacity = pl.DataFrame(results)
        clearing_prices_array = np.array(clearing_prices)

        return accepted_capacity, clearing_prices_array
    
    def run_auction_one_period(
        self,
        bids_by_generator : np.ndarray,
        capacity_offered : float,
        capacity_by_generator : np.ndarray
    ) -> tuple[dict[int, float], float]:
        if capacity_offered == 0:
            return {i : 0 for i in range(len(bids_by_generator))}
        bids = self.create_bids(
            bids_by_generator,
            capacity_by_generator
        )
        sorted_bids = sorted(bids, key=lambda x: x.bid_price, reverse=True)
        accepted_capacity = 0
        accepted_bids = {}
        clearing_price = 0
        
        for bid in sorted_bids:
            if accepted_capacity + bid.bid_capacity < capacity_offered:
                accepted_bids[bid.id] = bid.bid_capacity
            elif accepted_capacity == capacity_offered:
                accepted_bids[bid.id] = 0
            else:
                accepted_bids[bid.id] = capacity_offered - accepted_capacity
                clearing_price = bid.bid_price
        
        return accepted_bids, clearing_price
    
    def create_bids(
        bids_by_generator : np.ndarray,
        capacity_by_generator : np.ndarray,
    ) -> list[bid_info.BidInformation]:
        bids = []
        for i in range(len(bids_by_generator)):
            bid = bid_info.BidInformation(
                id = i,
                bid_price = bids_by_generator[i],
                bid_capacity = capacity_by_generator[i]
            )
            bids.append(bid)
        
        return bids
    
    