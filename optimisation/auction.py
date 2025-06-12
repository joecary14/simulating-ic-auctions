from collections import defaultdict
from functools import lru_cache
import optimisation.bid as bid
#TODO - check the precise clearing rule
def clear_auction(
    bid_schedules_by_participant_id: list[tuple[tuple[float, float], ...]],
    capacity_offered: float
) -> tuple[float, list[float]]:
    sorted_bids = extract_bids_from_demand_schedules(bid_schedules_by_participant_id)
    
    quantity_acccepted = 0.0
    clearing_price = 0.0
    num_participants = len(bid_schedules_by_participant_id)
    allocations = [0.0] * num_participants
    
    i = 0
    while i < len(sorted_bids) and quantity_acccepted < capacity_offered:
        current_price = sorted_bids[i].price
        
        start_index = i
        total_quantity_at_price = 0.0
        
        while i < len(sorted_bids) and sorted_bids[i].price == current_price:
            total_quantity_at_price += sorted_bids[i].quantity
            i += 1
        
        remaining_capacity = capacity_offered - quantity_acccepted
        
        if total_quantity_at_price <= remaining_capacity:
            for bid_index in range(start_index, i):
                bid = sorted_bids[bid_index]
                allocations[bid.bidder_id] += bid.quantity
                quantity_acccepted += bid.quantity
            clearing_price = current_price
            
        else:
            multiplier = remaining_capacity / total_quantity_at_price
            for bid_index in range(start_index, i):
                bid = sorted_bids[bid_index]
                allocated_quantity = bid.quantity * multiplier
                allocations[bid.bidder_id] += allocated_quantity
                quantity_acccepted += allocated_quantity
            clearing_price = current_price
            break
    
    return clearing_price, allocations

def extract_bids_from_demand_schedules(
    demand_schedules_submitted: list[tuple[tuple[float, float], ...]]
) -> list[bid.Bid]:
    bids = []
    for participant_id, demand_schedule in enumerate(demand_schedules_submitted):
        previous_quantity = 0.0
        for price, quantity in demand_schedule:
            marginal_quantity = quantity - previous_quantity
            bids.append(bid.Bid(participant_id, price, marginal_quantity))
            previous_quantity = quantity
    
    bids.sort(key=lambda b: b.price, reverse=True)
    
    return bids

def generate_payoff_for_bidder(
    realised_capacity_value: float,
    demand_schedules_submitted: list[tuple[tuple[float, float], ...]],
    participant_id: int,
    capacity_offered: float
) -> float: #This accounts for capacity value being negative (i.e. FAPD)
    clearing_price, allocations_by_participant = clear_auction(demand_schedules_submitted, capacity_offered)
    allocated_quantity = allocations_by_participant[participant_id]
    utility = (max(realised_capacity_value, 0) - clearing_price) * allocated_quantity
    
    return utility

@lru_cache(maxsize=50000)
def cached_payoff(
    outturn_value_index: int, 
    bidder_action_index: int, 
    other_action_indices: tuple[int], 
    possible_values: tuple[float], 
    possible_demand_schedules: tuple[tuple[tuple[float, float], ...]],
    capacity_offered: float
) -> float:
    outturn_value = possible_values[outturn_value_index]
    schedules = [possible_demand_schedules[bidder_action_index]]
    for other_index in other_action_indices:
        schedules.append(possible_demand_schedules[other_index])
    
    payoff = generate_payoff_for_bidder(
        outturn_value,
        schedules,
        0,
        capacity_offered
    )
    
    return payoff