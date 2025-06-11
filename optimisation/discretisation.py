import scipy.stats
import numpy as np
import pandas as pd

from typing import Tuple, List

default_min = -1e5
default_max = 1e5
spread_value = 10
spread_std = 5
test_dist = scipy.stats.norm(loc=spread_value, scale = spread_std)

def discretise_distribution(
    distribution,
    number_of_values: int,
    min_possible_value: float,
    max_possible_value: float
) -> np.ndarray:
    bins = np.linspace(min_possible_value, max_possible_value, number_of_values)
    centres = (bins[:-1] + bins[1:]) / 2
    pdf = []
    
    for i in range(number_of_values):
        if i == 0:
            prob = distribution.cdf(bins[i])
        elif i == 1:
            prob = distribution.cdf(centres[i]) - distribution.cdf(bins[0])
        elif i == number_of_values - 2:
            prob = distribution.cdf(bins[i+1]) - distribution.cdf(centres[i-1])
        elif i == number_of_values - 1:
            prob = 1 - distribution.cdf(bins[i])
        else:
            prob = distribution.cdf(centres[i]) - distribution.cdf(centres[i-1])
        
        pdf.append(prob)
    
    
    pdf = np.array(pdf)
    matrix = np.column_stack((bins, pdf))
    if np.sum(pdf) != 1:
        banana = 1
    
    return matrix

#TODO - may need to fix this to account for bucketing of floored values for v
def discretise_private_information_conditional_distribution_with_floor(
    distribution,
    number_of_values: int,
    min_possible_observation: float,
    max_possible_observation: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    negative_bins = np.linspace(min_possible_observation, 0, number_of_values)
    positive_bins = np.linspace(0, max_possible_observation, number_of_values)
    negative_centres = (negative_bins[:-1] + negative_bins[1:]) / 2
    positive_centres = (positive_bins[:-1] + positive_bins[1:]) / 2
    negative_pdf = []
    positive_pdf = []
    for i in range(number_of_values):
        if i == 0:
            negative_prob = distribution.cdf(negative_bins[i])
            positive_prob = distribution.cdf(positive_bins[i])
        elif i == 1:
            negative_prob = distribution.cdf(negative_centres[i]) - distribution.cdf(negative_bins[0])
            positive_prob = distribution.cdf(positive_centres[i]) - distribution.cdf(positive_bins[0])
        elif i == number_of_values - 1:
            negative_prob = 1 - distribution.cdf(negative_bins[i])
            positive_prob = 1 - distribution.cdf(positive_bins[i])
        elif i == number_of_values - 2:
            negative_prob = distribution.cdf(negative_bins[i+1]) - distribution.cdf(negative_centres[i-1])
            positive_prob = distribution.cdf(positive_bins[i+1]) - distribution.cdf(positive_centres[i-1])
        else:
            negative_prob = distribution.cdf(negative_centres[i]) - distribution.cdf(negative_centres[i-1])
            positive_prob = distribution.cdf(positive_centres[i]) - distribution.cdf(positive_centres[i-1])
        
        negative_pdf.append(negative_prob)
        positive_pdf.append(positive_prob)
    
    negative_forecast_error_distribution = pd.DataFrame({
        'value': negative_bins,
        'pdf': negative_pdf
    })
    positive_forecast_error_distribution = pd.DataFrame({
        'value': positive_bins,
        'pdf': positive_pdf
    })
        
    return negative_forecast_error_distribution, positive_forecast_error_distribution

def generate_demand_schedules(
    max_bid_price: float,
    max_total_quantity_demanded: float,
    number_of_possible_prices: int,
    number_of_possible_quantities: int
) -> Tuple[Tuple[Tuple[float, float], ...]]:
    """
    Generate all demand schedules where prices are non-increasing and quantities are strictly increasing.
    
    Output:
      - A list of demand schedules. Each schedule is represented as a list of length M of tuples:
        [(p1, q1), (p2, q2), ..., (pM, qM)], where p1 >= p2 >= ... >= pM and q1 < q2 < ... < qM.
    """
    #TODO - consider elimination of strictly dominated strategies
    possible_quantities = np.linspace(max_total_quantity_demanded/number_of_possible_quantities, max_total_quantity_demanded, number_of_possible_quantities)
    possible_prices = np.linspace(0, max_bid_price, number_of_possible_prices)
    prices_desc = sorted(possible_prices, reverse=True) 
    quantities_asc = sorted(possible_quantities)
    schedules = []

    def backtrack(i, prev_price_idx, prev_qty_idx, current_schedule: List[Tuple[float, float]]):
        if i == number_of_possible_prices: 
            schedules.append(tuple(current_schedule))
            return
        
        # Prices must be non-increasing (can be same or lower)
        for price_idx in range(prev_price_idx, len(prices_desc)):
            # Quantities must be strictly increasing
            for qty_idx in range(prev_qty_idx + 1, len(quantities_asc)):
                current_schedule.append((prices_desc[price_idx], quantities_asc[qty_idx]))
                backtrack(i + 1, price_idx, qty_idx, current_schedule)
                current_schedule.pop()
    
    backtrack(i=0, prev_price_idx=0, prev_qty_idx=-1, current_schedule=[])
    
    return tuple(schedules)
