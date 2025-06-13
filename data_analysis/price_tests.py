import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Dict
from scipy.stats import shapiro, wilcoxon

def process_raw_data(
    raw_data_filepath: str
) -> Dict[str, pd.DataFrame]:
    raw_data_dfs = pd.read_excel(raw_data_filepath, sheet_name=None, dtype=None)
    processed_data_dfs = {}
    for sheet_name, raw_data_df in raw_data_dfs.items():
        
        processed_data_df = raw_data_df.copy()
        processed_data_df = processed_data_df.replace('-', 0)
        processed_data_df = processed_data_df.infer_objects()
        df_columns = processed_data_df.columns
        col2 = df_columns[1]
        col3 = df_columns[2]
        processed_data_df['market_expectation_price'] = processed_data_df.apply(
            lambda row: -row[col3] if row[col3] > row[col2] else row[col2],
            axis=1
        )
        processed_data_df['outturn_price_spread'] = processed_data_df[df_columns[3]] - processed_data_df[df_columns[4]]
        processed_data_df['difference'] = processed_data_df['outturn_price_spread'] - processed_data_df['market_expectation_price']
        processed_data_dfs[sheet_name] = processed_data_df[processed_data_df['Offered Capacity'] != 0]
    
    return processed_data_dfs

def test_for_normality(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, bool]:
    test_results = {}
    for sheet_name, df in dataframes.items():
        stat, p_value = shapiro(df['difference'])

        print(f"Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p_value:.4f}")

        if p_value > 0.05:
            test_results[sheet_name] = True
        else:
            test_results[sheet_name] = False
        
        mean = df['difference'].mean()
        std_dev = df['difference'].std()
        plt.figure(figsize=(10, 6))
        plt.hist(df['difference'], bins=30, alpha=0.7, color='blue', edgecolor='black', density=True)
        plt.title("Histogram of 'Difference' (Normal Distribution)", fontsize=14)
        plt.xlabel("Difference", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean:.2f}")
        plt.axvline(mean + std_dev, color='green', linestyle='dashed', linewidth=1, label=f"Std Dev: +{std_dev:.2f}")
        plt.axvline(mean - std_dev, color='green', linestyle='dashed', linewidth=1, label=f"Std Dev: -{std_dev:.2f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    return test_results

def wilcoxon_test(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, bool]:
    p_values = {}
    for sheet_name, df in dataframes.items():
        stat, p_value = wilcoxon(df['difference'])
        print(f"Wilcoxon Test Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value > 0.05: # type: ignore
            print("No significant bias detected (fail to reject H0).")
            p_values[sheet_name] = False
        else:
            print("Significant bias detected (reject H0).")
            p_values[sheet_name] = True
    return p_values

def bootstrap_resample_test_mean_bias(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, bool]:
    mean_biased = {}
    for sheet_name, df in dataframes.items():
        n_iterations = 1000
        bootstrap_means = [
            np.mean(np.random.choice(df['difference'], size=len(df['difference']), replace=True))
            for _ in range(n_iterations)
        ]

        lower_bound = np.percentile(bootstrap_means, 2.5)
        upper_bound = np.percentile(bootstrap_means, 97.5)
        mean_difference = df['difference'].mean()

        print(f"Mean Difference: {mean_difference:.4f}")
        print(f"95% Confidence Interval for the Mean: ({lower_bound:.4f}, {upper_bound:.4f})")
        
        if lower_bound <= 0 <= upper_bound:
            print("The mean is not significantly biased (zero is within the confidence interval).")
            mean_biased[sheet_name] = False
        else:
            print("The mean is significantly biased (zero is outside the confidence interval).")
            mean_biased[sheet_name] = True
    
    return mean_biased

def generate_qq_plots(
    dataframes: Dict[str, pd.DataFrame]
) -> None:
    for sheet_name, df in dataframes.items():
        plt.figure(figsize=(10, 6))
        stats.probplot(df['difference'], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {sheet_name}", fontsize=14)
        plt.xlabel("Theoretical Quantiles", fontsize=12)
        plt.ylabel("Sample Quantiles", fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()
        skewness = df['difference'].skew()
        kurtosis = df['difference'].kurtosis()
        print(f"Skewness for {sheet_name}: {skewness:.4f}")
        print(f"Kurtosis for {sheet_name}: {kurtosis:.4f}")
        
        
def add_rolling_volatility(df: pd.DataFrame, column_name='outturn_price_spread', window=48):
    df['rolling_volatility'] = df[column_name].rolling(window=window, min_periods=1).std()
    return df