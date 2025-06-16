from dataclasses import dataclass

@dataclass
class SolverParameters:
    number_of_monte_carlo_simulations: int
    optimisation_algorithm_tolerance: float
    utility_loss_tolerance: float
    max_iterations: int