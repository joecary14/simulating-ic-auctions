import numpy as np
import polars as pl
import constants as ct
import auction_simulation.bid_information as bid_info

class AuctionInformation:
    def __init__(
        self, 
        actual_domestic_prices : np.ndarray, 
        actual_foreign_prices : np.ndarray,
        bids_by_generator_by_period : pl.DataFrame,
        capacity_by_generator_by_period : pl.DataFrame,
        capacity_offered : np.ndarray
    ):
        self.actual_domestic_prices = actual_domestic_prices
        self.actual_foreign_prices = actual_foreign_prices
        self.bids_by_generator_by_period = bids_by_generator_by_period
        self.capacity_by_generator_by_period = capacity_by_generator_by_period
        self.capacity_offered = capacity_offered
        
    def run_auction(
        self
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        periods = self.bids_by_generator_by_period[ct.ColumnNames.DELIVERY_PERIOD.value].unique().sort()
        num_periods = len(periods)
        first_period = periods[0]
        first_period_bids = self.bids_by_generator_by_period.filter(
            pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == first_period
        ).drop(ct.ColumnNames.DELIVERY_PERIOD.value)
        generator_ids = first_period_bids.columns
        results_by_generator = {str(generator_id): np.zeros(num_periods) for generator_id in generator_ids}
        clearing_prices = np.zeros(num_periods)
        
        for period_idx, period in enumerate(periods):
            bids_by_generator = self.bids_by_generator_by_period.filter(
                pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
            ).drop(ct.ColumnNames.DELIVERY_PERIOD.value)
            
            capacity_by_generator = self.capacity_by_generator_by_period.filter(
                pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
            ).drop(ct.ColumnNames.DELIVERY_PERIOD.value)
            
            capacity_offered = self.capacity_offered[period_idx]
            accepted_bids, clearing_price = self.run_auction_one_period(
                bids_by_generator,
                capacity_offered,
                capacity_by_generator
            )
            clearing_prices[period_idx] = clearing_price
            
            for generator_id, accepted_capacity in accepted_bids.items():
                results_by_generator[generator_id][period_idx] = accepted_capacity
        
        return results_by_generator, clearing_prices
    
    def run_auction_one_period(
        self,
        bids_by_generator : np.ndarray,
        capacity_offered : float,
        capacity_by_generator : np.ndarray
    ) -> tuple[dict[str, float], float]:
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
        if all(bid.bid_capacity == 0 for bid in sorted_bids) or all(bid.bid_price == 0 for bid in sorted_bids):
            return {bid.id: 0 for bid in sorted_bids}, 0
        
        for bid in sorted_bids:
            if accepted_capacity + bid.bid_capacity < capacity_offered:
                accepted_bids[bid.id] = bid.bid_capacity
                accepted_capacity += bid.bid_capacity
            elif accepted_capacity == capacity_offered:
                accepted_bids[bid.id] = 0
            else:
                accepted_bids[bid.id] = capacity_offered - accepted_capacity
                accepted_capacity += capacity_offered - accepted_capacity
                clearing_price = bid.bid_price
        
        return accepted_bids, clearing_price
    
    def create_bids(
        self,
        bids_by_generator : pl.DataFrame,
        capacity_by_generator : pl.DataFrame
    ) -> list[bid_info.BidInformation]:
        bids = []
        generator_ids = capacity_by_generator.columns
        for generator_id in generator_ids:
            bid_price = bids_by_generator[generator_id][0]
            bid_capacity = capacity_by_generator[generator_id][0]
            bid = bid_info.BidInformation(
                id = generator_id,
                bid_price = bid_price,
                bid_capacity = bid_capacity
            )
            bids.append(bid)
        
        return bids
    
    