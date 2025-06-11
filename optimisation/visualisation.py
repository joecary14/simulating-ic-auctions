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
    sns.heatmap(current_sigma, cmap="viridis")
    plt.title("Heatmap of current_sigma")
    plt.xlabel("Action Index (m)")
    plt.ylabel("Signal Index (l)")
    plt.show()