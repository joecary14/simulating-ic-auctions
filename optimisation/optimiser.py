import numpy as np
from typing import Tuple
import optimisation.auction as auction
#TODO - may want to transpose sigma for consistency, so that sigma[m, l] = Pr[b_i = b_m | o_i = o_l] for consistency with other probability matrix definitions
#TODO - check why sigma is currently set up so that the 
def soda_algorithm(
    V: np.ndarray, 
    O: np.ndarray, 
    B: list[list[tuple[float, float]]], 
    pV: np.ndarray, 
    gO_given_V: np.ndarray, 
    number_of_bidders: int, 
    number_of_simulations: int,
    capacity_offered: float,
    max_iter: int = 1000, 
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
    B_tuple = tuple(B)
    V_tuple = tuple(V) 
    
    # Precompute marginal probabilities p(o_ℓ)
    marginal_observation_probabilities = gO_given_V @ pV  # shape (L,)
    posterior_probabilities = np.zeros((K, L))  # shape (K, L)
    for k in range(K):
        for l in range(L):
            posterior_probabilities[k, l] = (gO_given_V[l, k] * pV[k]) / marginal_observation_probabilities[l]
    
    # Initialize dual variables Y and strategy matrix σ
    Y = np.zeros((L, M))  # Dual variables
    sigma = np.zeros((L, M))  # Strategies

    # Main SODA loop
    for t in range(max_iter):
        # Mirror projection: Update σ from Y
        # Vectorized update for all rows of σ
        row_max = Y.max(axis=1, keepdims=True)
        W = np.exp(Y - row_max)  # Subtract max from each row
        sigma = marginal_observation_probabilities[:, None] * (W / W.sum(axis=1, keepdims=True))
        conditional_sigma = W / W.sum(axis=1, keepdims=True)
        # Save σ for convergence check
        sigma_prev = sigma.copy()
        
        # Compute expected utilities
        U = compute_expected_utilities(
            conditional_sigma,
            K,
            posterior_probabilities,
            gO_given_V,
            B_tuple,
            V_tuple,
            number_of_simulations,
            number_of_bidders,
            capacity_offered
        )
        
        # Update dual variables
        eta = 1 / np.sqrt(t + 1)  # Step size
        Y += eta * U
        if np.max(np.abs(sigma - sigma_prev)) < tol:
            print(f"Converged after {t} iterations.")
            break
    
    return sigma

def compute_expected_utilities(
    conditional_sigma: np.ndarray,
    K: int,
    posterior_probabilities: np.ndarray,
    gO_given_V: np.ndarray,
    B_tuple: Tuple[list[tuple[float, float]], ...],
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
                        B_tuple,
                        capacity_offered
                    )
                    mc_payoff += payoff
                mc_payoff /= num_mc
                exp_utility += weight_vk_given_ol * mc_payoff
            U[l, m] = exp_utility
    
    return U

def sample_index(
    probabilities: np.ndarray
) -> int:
    return np.random.choice(len(probabilities), p=probabilities)