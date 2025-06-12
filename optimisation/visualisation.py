import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualise_as_heatmap(
    current_sigma: np.ndarray
) -> None:
    """
    Visualise the current_sigma as a heatmap.
    
    Parameters:
        current_sigma (np.ndarray): The current sigma matrix to visualise.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(current_sigma, cmap="gray")
    plt.title("Heatmap of current_sigma")
    plt.xlabel("Action Index (m)")
    plt.ylabel("Signal Index (l)")
    plt.show()

def plot_convergence(convergence_history):    
    iterations = list(range(1, len(convergence_history) + 1))
    max_distances = [h[0] for h in convergence_history]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations, max_distances)
    plt.xlabel('Iteration')
    plt.ylabel('Max Strategy Distance (log scale)')
    plt.title('SODA Algorithm Convergence')
    plt.grid(True)
    plt.show()