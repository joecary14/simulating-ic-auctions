import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import optimisation.optimiser as optimiser

def check_for_equilibrium(
    current_conditional_sigma: np.ndarray,
    marginal_observation_probabilities: np.ndarray,
    conditional_observation_probabilities: np.ndarray,
    posterior_probabilities: np.ndarray,
    demand_schedules: Tuple[Tuple[Tuple[float, float], ...]],
    number_of_possible_values: int,
    possible_values: tuple[float, ...],
    number_of_mc_simulations: int,
    number_of_participants: int,
    capacity_offered: float,
    tol: float = 1e-4
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
    best_value, current_value, gap = equilibrium_utility_loss(
        current_utility,
        marginal_observation_probabilities,
        current_sigma,
        tol=tol
    )
    
    print(f"Best response value: {best_value}, Current value: {current_value}, Gap: {gap}")
    if gap > tol:
        print("Warning: The current strategy is not an equilibrium strategy.")
    else:
        print("The current strategy is an equilibrium strategy.")
    
    return gap

def equilibrium_utility_loss(
    current_utility: np.ndarray,
    marginal_observation_probabilities: np.ndarray,
    current_sigma: np.ndarray,
    tol: float = 1e-4
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
    
    gap = float(best_value - current_value)
    return float(best_value), float(current_value), gap