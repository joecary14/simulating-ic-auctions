import numpy as np
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import optimisation.auction as auction

def soda_algorithm(
    V: np.ndarray, 
    O: np.ndarray, 
    B: Tuple[Tuple[Tuple[float, float], ...]], 
    pV: np.ndarray, 
    gO_given_V: np.ndarray, 
    number_of_bidders: int, 
    number_of_simulations: int,
    capacity_offered: float,
    max_iter: int = 10000, 
    tol: float=1e-6
):
    """
    SODA Algorithm for Computing Distributional Strategies in a Discrete Auction Game.

    Inputs:
      - V            : (K,) array of value grid points (v_k).
      - O            : (L,) array of signal grid points (o_l).
      - B            : List of actions (length M), where each b_m is a demand schedule.
      - pV           : (K,) array of prior probabilities p(v_k).
      - gO_given_V   : (L x K) array of conditional probabilities g(o_l | v_k).
      - n            : Number of bidders.
      - num_mc       : Number of Monte Carlo samples for approximating payoffs.
      - max_iter     : Maximum number of iterations.
      - tol          : Convergence tolerance for strategies.

    Outputs:
      - sigma        : (L x M) array of equilibrium probabilities sigma(l, m).
    """
    K = len(V)
    L = len(O)
    M = len(B)
    V_tuple = tuple(V) 
    
    # Precompute marginal probabilities p(o_ℓ)
    marginal_observation_probabilities = gO_given_V @ pV  # shape (L,)
    marginal_observation_probabilities /= np.sum(marginal_observation_probabilities)
    posterior_probabilities = np.zeros((K, L))  # shape (K, L)
    for k in range(K):
        for l in range(L):
            posterior_probabilities[k, l] = (gO_given_V[l, k] * pV[k]) / marginal_observation_probabilities[l]
    
    # Initialize dual variables Y and strategy matrix σ
    Y = np.zeros((L, M))  # Dual variables
    current_conditional_sigma = update_conditional_sigma(Y)
    current_sigma = marginal_observation_probabilities[:, None] * current_conditional_sigma

    # Main SODA loop
    for t in range(max_iter):
        print(f"Iteration {t+1} of {max_iter}")
        U = compute_expected_utilities(
            current_conditional_sigma,
            K,
            posterior_probabilities,
            gO_given_V,
            B,
            V_tuple,
            number_of_simulations,
            number_of_bidders,
            capacity_offered
        )
        
        # Update dual variables
        eta = 1 / np.sqrt(t + 1)  # Step size
        Y += eta * U
        
        new_conditional_sigma = update_conditional_sigma(Y)
        new_sigma = marginal_observation_probabilities[:, None] * new_conditional_sigma
        
        sigma_distance = np.max(np.abs(new_sigma - current_sigma))
        print(f"Max distance between current and new sigma: {sigma_distance:.6f}")

        if sigma_distance < tol:
            print(f"Converged after {t} iterations.")
            return new_sigma
        
        current_sigma = new_sigma.copy()
        current_conditional_sigma = new_conditional_sigma.copy()

    print("Reached maximum iterations without convergence.")
    return current_sigma

def compute_expected_utilities(
    conditional_sigma: np.ndarray,
    K: int,
    posterior_probabilities: np.ndarray,
    gO_given_V: np.ndarray,
    B: Tuple[Tuple[Tuple[float, float], ...]],
    V_tuple: Tuple[float, ...],
    num_mc: int,
    num_participants: int,
    capacity_offered: float
) -> np.ndarray:
    """
    Compute the LxM matrix U of expected payoffs.

    Inputs:
      - sigma         : (L x M) array of current strategy probabilities,
                        where sigma[l,m] = Pr[b_i = b_m|o_i = o_l].
      - K             : Number of discrete values (v_k).
      - posterior_probabilities : (K x L) array of posterior probabilities p(v_k | o_l). Columns should sum to 1; not necessarily rows
      - gO_given_V    : (L x K) array of conditional probs g(o_l | v_k). Columns should sum to 1; not necessarily rows
      - B_tuple       : tuple of length M, each entry is a demand-schedule object b_m.
      - V_tuple       : tuple of length K, the discrete values v_k.
      - num_mc        : number of Monte Carlo draws per (k,l,m) to approximate the
                        expectation over other bidders' signals and actions.
    
    Returns:
      - U             : (L x M) array where
                        U[l,m] ≈ E[u_i | o_i=o_l, b_i=b_m].
    """

    L, M = conditional_sigma.shape
    U = np.zeros((L, M))
    for l in range(L):
        for m in range(M):
            exp_utility = 0
            for k in range(K):
                weight_vk_given_ol = posterior_probabilities[k,l]
                mc_payoff = 0
                for _ in range(num_mc):
                    other_signal_indices = [
                        sample_index(gO_given_V[:, k]) for _ in range(num_participants-1)
                    ]
                    other_action_indices = [
                        sample_index(conditional_sigma[observation_row]) for observation_row in other_signal_indices
                    ]
                    
                    payoff = auction.cached_payoff(
                        k,
                        m,
                        tuple(other_action_indices),
                        V_tuple,
                        B,
                        capacity_offered
                    )
                    mc_payoff += payoff
                mc_payoff /= num_mc
                exp_utility += weight_vk_given_ol * mc_payoff
            U[l, m] = exp_utility
    
    return U

def update_conditional_sigma(
    Y: np.ndarray
) -> np.ndarray:
    row_max = Y.max(axis=1, keepdims=True)
    W = np.exp(Y - row_max)  # Subtract max from each row
    conditional_sigma = W / W.sum(axis=1, keepdims=True)
    
    return conditional_sigma

def sample_index(
    probabilities: np.ndarray
) -> int:
    return np.random.choice(len(probabilities), p=probabilities)