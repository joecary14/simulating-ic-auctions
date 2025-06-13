import optimisation.discretisation as discretisation
import scipy.stats as stats
import data_analysis.price_tests as price_tests

path = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Analysis/Bid Shading Analysis/Raw Data.xlsx'

dfs = price_tests.process_raw_data(path)
price_tests.generate_qq_plots(dfs)