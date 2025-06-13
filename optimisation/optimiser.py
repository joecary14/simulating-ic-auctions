import numpy as np
from typing import Optional, Tuple
import optimisation.auction as auction
import optimisation.visualisation as visualisation

def soda_algorithm(
    possible_values: np.ndarray, 
    possible_observations: np.ndarray, 
    possible_demand_schedules: Tuple[Tuple[Tuple[float, float], ...]], 
    marginal_observation_probabilities: np.ndarray, 
    conditional_observation_probabilities: np.ndarray,
    posterior_probabilities: np.ndarray,
    number_of_bidders: int, 
    number_of_simulations: int,
    capacity_offered: float,
    input_conditional_sigma: 'Optional[np.ndarray]',
    input_dual_variables: 'Optional[np.ndarray]',
    start_iteration_number: int = 0,
    max_iter: int = 1000000, 
    tol: float=1e-2
):
    K = len(possible_values)
    L = len(possible_observations)
    M = len(possible_demand_schedules)
    V_tuple = tuple(possible_values) 
    
    # Initialize dual variables Y and strategy matrix σ
    if input_conditional_sigma is None or input_dual_variables is None:
        Y = np.zeros((L, M))  # Dual variables
        current_conditional_sigma = update_conditional_sigma(Y)
        current_sigma = marginal_observation_probabilities[:, None] * current_conditional_sigma
    else:
        Y = input_dual_variables
        current_conditional_sigma = input_conditional_sigma
        current_sigma = marginal_observation_probabilities[:, None] * current_conditional_sigma
    convergence_history = []
    last_iteration = 0
    # Main SODA loop
    for t in range(start_iteration_number, max_iter + start_iteration_number):
        print(f"Iteration {t+1 - start_iteration_number} of {max_iter}")
        U = compute_expected_utilities(
            current_conditional_sigma,
            K,
            posterior_probabilities,
            conditional_observation_probabilities,
            possible_demand_schedules,
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
        mean_distance = np.mean(np.abs(new_sigma - current_sigma))
        convergence_history.append((sigma_distance, mean_distance))
        print(f"Max distance between current and new sigma: {sigma_distance:.6f}")
        three_point_average_sigma_distance = np.mean([h[0] for h in convergence_history[-3:]]) if len(convergence_history) >= 3 else 1

        if three_point_average_sigma_distance < tol:
            print(f"Converged after {t+1} iterations.")
            visualisation.plot_convergence(convergence_history)
            last_iteration = t + 1
            return new_conditional_sigma, Y, last_iteration
        
        current_conditional_sigma = new_conditional_sigma.copy()
        current_sigma = new_sigma.copy()

    print("Reached maximum iterations without convergence.")
    visualisation.plot_convergence(convergence_history)
    last_iteration = max_iter
    return current_conditional_sigma, Y, last_iteration

def compute_expected_utilities(
    conditional_sigma: np.ndarray,
    K: int,
    posterior_probabilities: np.ndarray,
    conditional_observation_probabilities: np.ndarray,
    B: Tuple[Tuple[Tuple[float, float], ...]],
    V_tuple: Tuple[float, ...],
    num_mc: int,
    num_participants: int,
    capacity_offered: float
) -> np.ndarray:
    """
    Compute the LxM matrix U of expected payoffs.

    Inputs:
      - conditional_sigma         : (L x M) array of current strategy probabilities,
                        where conditional_sigma[l,m] = Pr[b_i = b_m|o_i = o_l].
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
    signal_probabilties_by_vk = [conditional_observation_probabilities[:, k] for k in range(K)]
    conditional_strategy_cumulative_probabilities = np.cumsum(conditional_sigma, axis=1)
    for l in range(L):
        for m in range(M):
            exp_utility = 0
            for k in range(K):
                weight_vk_given_ol = posterior_probabilities[k,l]
                if weight_vk_given_ol == 0:
                    continue
                mc_payoffs = np.zeros(num_mc)
                signal_probabilities = signal_probabilties_by_vk[k]
                # Sample other participants' signals and actions
                all_other_signals = np.random.choice(
                    L,
                    size = (num_mc, num_participants-1),
                    p=signal_probabilities
                )
                for simulation_index in range(num_mc):
                    other_signal_indices = all_other_signals[simulation_index]
                    action_randoms = np.random.random(len(other_signal_indices))
                    other_action_indices = [
                        np.searchsorted(
                            conditional_strategy_cumulative_probabilities[observation_row],
                            rand
                        )
                        for observation_row, rand in zip(other_signal_indices, action_randoms)
                    ]
                    payoff = auction.cached_payoff(
                        k,
                        m,
                        tuple(other_action_indices),
                        V_tuple,
                        B,
                        capacity_offered
                    )
                    mc_payoffs[simulation_index] = payoff
                
                exp_utility += np.mean(mc_payoffs) * weight_vk_given_ol
            U[l, m] = exp_utility
    
    return U

def update_conditional_sigma(
    Y: np.ndarray
) -> np.ndarray:
    row_max = Y.max(axis=1, keepdims=True)
    W = np.exp(Y - row_max)  # Subtract max from each row
    conditional_sigma = W / W.sum(axis=1, keepdims=True)
    
    return conditional_sigma

