import numpy as np
from typing import Tuple
import optimisation.auction as auction

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
      - gO_given_V   : (K x L) array of conditional probabilities g(o_l | v_k).
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
    rho = gO_given_V.T @ pV  # shape (L,)
    
    # Initialize dual variables Y and strategy matrix σ
    Y = np.zeros((L, M))  # Dual variables
    sigma = np.zeros((L, M))  # Strategies

    # Main SODA loop
    for t in range(max_iter):
        # Mirror projection: Update σ from Y
        # Vectorized update for all rows of σ
        row_max = Y.max(axis=1, keepdims=True)
        W = np.exp(Y - row_max)  # Subtract max from each row
        sigma = rho[:, None] * (W / W.sum(axis=1, keepdims=True))
        # Save σ for convergence check
        sigma_prev = sigma.copy()
        
        # Compute expected utilities
        U = compute_expected_utilities(
            sigma,
            pV,
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
    sigma: np.ndarray,
    pV: np.ndarray,
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
                        where sigma[l,m] = Pr[b_i = b_m | o_i = o_l].
      - pV            : (K,)   array of prior probabilities p(v_k).
      - gO_given_V    : (K x L) array of conditional probs g(o_l | v_k).
      - B_tuple       : tuple of length M, each entry is a demand-schedule object b_m.
      - V_tuple       : tuple of length K, the discrete values v_k.
      - num_mc        : number of Monte Carlo draws per (k,l,m) to approximate the
                        expectation over other bidders' signals and actions.
    
    Returns:
      - U             : (L x M) array where
                        U[l,m] ≈ E[u_i | o_i=o_l, b_i=b_m].
    """

    L, M = sigma.shape
    K = len(pV)
    U = np.zeros((L, M))
    
    for l in range(L):
        marginal_p_ol = sum(pV[k] * gO_given_V[k, l] for k in range(K))
        for m in range(M):
            exp_utility = 0
            for k in range(K):
                weight_vk_given_ol = (pV[k] * gO_given_V[k, l])/ marginal_p_ol
                mc_payoff = 0
                
                for _ in range(num_mc):
                    other_signal_indices = [
                        sample_index(gO_given_V[k]) for _ in range(num_participants-1)
                    ]
                    other_action_indices = [
                        sample_index(sigma[l_j]) for l_j in other_signal_indices
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