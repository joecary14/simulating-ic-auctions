from collections import defaultdict
from functools import lru_cache
import optimisation.bid as bid

def clear_auction(
    bid_schedules_by_participant_id: dict[int, list[tuple[float, float]]],
    capacity_offered: float
) -> tuple[float, dict[int, float]]:
    sorted_bids = extract_bids_from_demand_schedules(bid_schedules_by_participant_id)
    bids_by_price = defaultdict(list)
    for b in sorted_bids:
        bids_by_price[b.price].append(b)
    
    quantity_acccepted = 0
    clearing_price = 0
    allocations_by_participant = {}
    for price, bids_at_price in bids_by_price.items():
        total_quantity = sum(b.quantity for b in bids_at_price)
        if total_quantity + quantity_acccepted < capacity_offered:
            for bid in bids_at_price:
                allocations_by_participant[bid.participant_id] += bid.quantity
                quantity_acccepted += bid.quantity
        else:
            multiplier = (capacity_offered - quantity_acccepted) / total_quantity
            for bid in bids_at_price:
                allocated_quantity = bid.quantity * multiplier
                allocations_by_participant[bid.participant_id] += allocated_quantity
                quantity_acccepted += allocated_quantity
                clearing_price = price
            break
    
    if quantity_acccepted < capacity_offered:
        raise ValueError("Not enough bids to meet the capacity offered.")
    
    return clearing_price, allocations_by_participant
        
def extract_bids_from_demand_schedules(
    demand_schedules_by_participant_id: dict[int, list[tuple[float, float]]]
) -> list[bid.Bid]:
    bids = []
    for participant_id, demand_schedule in demand_schedules_by_participant_id.items():
        for i in range(len(demand_schedule)):
            price, quantity = demand_schedule[i]
            if i == 0:
                bids.append(bid.Bid(participant_id, price, quantity))
            else:
                previous_price, previous_quantity = demand_schedule[i - 1]
                marginal_quantity = quantity - previous_quantity
                bids.append(bid.Bid(participant_id, price, marginal_quantity))
    
    bids.sort(key=lambda b: b.price, reverse=True)
    
    return bids

def generate_payoff_for_bidder(
    realised_capacity_value: float,
    bid_schedules_by_participant_id: dict[int, list[tuple[float, float]]],
    participant_id: int,
    capacity_offered: float
) -> float: #This accounts for capacity value being bnegative (i.e. FAPD)
    clearing_price, allocations_by_participant = clear_auction(bid_schedules_by_participant_id, capacity_offered)
    allocated_quantity = allocations_by_participant[participant_id]
    utility = (max(realised_capacity_value, 0) - clearing_price) * allocated_quantity
    
    return utility

@lru_cache(maxsize=50000)
def cached_payoff(
    value_index: int, 
    bidder_action_index: int, 
    other_action_indices: list[int], 
    possible_values: tuple[int], 
    possible_demand_schedules: tuple[list[tuple[float, float]]],
    capacity_offered: float
) -> float:
    value = possible_values[value_index]
    schedules = {i: possible_demand_schedules[bidder_action_index] for i in range(len(other_action_indices) + 1)}
    for j, other_index in enumerate(other_action_indices):
        schedules[j + 1] = possible_demand_schedules[other_index]
    
    payoff = generate_payoff_for_bidder(
        value,
        schedules,
        0,
        capacity_offered
    )
    
    return payoff