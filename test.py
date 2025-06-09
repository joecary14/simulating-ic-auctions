import optimisation.discretisation as discretisation
import scipy.stats as stats


test_dist = stats.norm(loc=0, scale=5)
discretisation.discretise_private_information_conditional_distribution_with_floor(test_dist, 10, -50, 50)