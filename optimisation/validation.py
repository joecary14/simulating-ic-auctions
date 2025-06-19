import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import optimisation.optimiser as optimiser

def calculate_utility_loss(
    current_conditional_sigma: np.ndarray,
    marginal_observation_probabilities: np.ndarray,
    conditional_observation_probabilities: np.ndarray,
    posterior_probabilities: np.ndarray,
    demand_schedules: Tuple[Tuple[Tuple[float, float], ...]],
    number_of_possible_values: int,
    possible_values: tuple[float, ...],
    number_of_mc_simulations: int,
    number_of_participants: int,
    capacity_offered: float
) -> float:
    current_utility = optimiser.compute_expected_utilities(
        current_conditional_sigma,
        number_of_possible_values,
        posterior_probabilities,
        conditional_observation_probabilities,
        demand_schedules,
        possible_values,
        number_of_mc_simulations,
        number_of_participants,
        capacity_offered
    )
    
    current_sigma = marginal_observation_probabilities[:, None] * current_conditional_sigma
    best_value, current_value, relative_gap = equilibrium_utility_loss(
        current_utility,
        marginal_observation_probabilities,
        current_sigma
    )
    
    print(f"Best response value: {best_value}, Current value: {current_value}, Relative Gap: {relative_gap}")
    return relative_gap

def equilibrium_utility_loss(
    current_utility: np.ndarray,
    marginal_observation_probabilities: np.ndarray,
    current_sigma: np.ndarray
) -> tuple[float, float, float]:
    L, M = current_utility.shape
    c = -current_utility.flatten()              # we will minimize cᵀ x = -U⋅s' ⇒ max U⋅s'
    
    # Equality constraints: for each k, ∑_ℓ s'_{kℓ} = p_o[k]
    A_eq = np.zeros((L, L*M))
    for l in range(L):
        A_eq[l, l*M:(l+1)*M] = 1.0
    b_eq = marginal_observation_probabilities.copy()
    
    # Bounds: s'_{kℓ} ≥ 0
    bounds = [(0, None)] * (L*M)
    
    # Solve LP
    res = linprog(c=c,
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds,
                method="highs")
    if not res.success:
        raise RuntimeError(f"Best-response LP failed: {res.message}")
    
    # Extract best-response value
    best_value = -res.fun
    
    # Current strategy value
    current_value = np.sum(current_sigma * current_utility)
    
    relative_gap = np.abs(float(best_value - current_value)/float(current_value)) if  best_value != 0 else float('inf')
    return float(best_value), float(current_value), relative_gap