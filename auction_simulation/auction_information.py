import numpy as np
import polars as pl
import constants as ct

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
        
        bids_by_period = {}
        capacities_by_period = {}
        
        for period in periods:
            bids_by_period[period] = self.bids_by_generator_by_period.filter(
                pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
            ).drop(ct.ColumnNames.DELIVERY_PERIOD.value)
            
            capacities_by_period[period] = self.capacity_by_generator_by_period.filter(
                pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
            ).drop(ct.ColumnNames.DELIVERY_PERIOD.value)
        
        generator_ids = bids_by_period[periods[0]].columns
        results_by_generator = {str(generator_id): np.zeros(num_periods) for generator_id in generator_ids}
        clearing_prices = np.zeros(num_periods)
        
        for period_idx, period in enumerate(periods):
            accepted_bids, clearing_price = self.run_auction_one_period(
                bids_by_period[period],
                self.capacity_offered[period_idx],
                capacities_by_period[period]
            )
            clearing_prices[period_idx] = clearing_price
            
            for generator_id, accepted_capacity in accepted_bids.items():
                results_by_generator[generator_id][period_idx] = accepted_capacity
        
        return results_by_generator, clearing_prices
    
    def run_auction_one_period(
        self,
        bids_by_generator : pl.DataFrame,
        capacity_offered : float,
        capacity_by_generator : pl.DataFrame
    ) -> tuple[dict[str, float], float]:
        if capacity_offered == 0:
            return {i : 0 for i in range(len(bids_by_generator))}, 0
        
        bid_prices = np.array([bids_by_generator[generator_id][0] for generator_id in bids_by_generator.columns])
        bid_capacities = np.array([capacity_by_generator[generator_id][0] for generator_id in capacity_by_generator.columns])
        bid_ids = [generator_id for generator_id in bids_by_generator.columns]
        
        if (bid_capacities == 0).all() or (bid_prices == 0).all():
            return {bid_id: 0 for bid_id in bid_ids}, 0
        
        sort_indices = np.argsort(-bid_prices)
        
        accepted_bids = {bid_id: 0 for bid_id in bid_ids}
        accepted_capacity = 0
        clearing_price = 0
        
        for idx in sort_indices:
            bid_id = bid_ids[idx]
            bid_price = bid_prices[idx]
            bid_capacity = bid_capacities[idx]
            
            remaining_capacity = capacity_offered - accepted_capacity
            if remaining_capacity <= 0:
                break
                
            if bid_capacity <= remaining_capacity:
                accepted_bids[bid_id] = bid_capacity
                accepted_capacity += bid_capacity
            else:
                accepted_bids[bid_id] = remaining_capacity
                accepted_capacity += remaining_capacity
                clearing_price = bid_price
                break
        
        return accepted_bids, clearing_price
    
    