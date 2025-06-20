import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from typing import Dict
from scipy.stats import shapiro, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def process_raw_data(
    raw_data_filepath: str
) -> Dict[str, pd.DataFrame]:
    raw_data_dfs = pd.read_excel(raw_data_filepath, sheet_name=None, dtype=None)
    processed_data_dfs = {}
    for sheet_name, raw_data_df in raw_data_dfs.items():
        
        processed_data_df = raw_data_df.copy()
        processed_data_df = processed_data_df.replace('-', 0).infer_objects(copy=False)
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
        
        
def add_rolling_volatility(df: pd.DataFrame, window=48):
    df['rolling_volatility_price_spread'] = df['outturn_price_spread'].rolling(window=window, min_periods=1).std()
    df['rolling_volatility_GB_price_spread'] = df['GB Price'].rolling(window=window, min_periods=1).std()
    price_columns = [col for col in df.columns if 'price' in col.lower() and 'gb' not in col.lower()]
    if price_columns:
        df[f'rolling_volatility_{price_columns[0]}'] = df[price_columns[0]].rolling(window=window).std()
    return df

def add_hourly_dummy_variables(
    df: pd.DataFrame
) -> pd.DataFrame:
    df['hour'] = pd.to_datetime(df['UTC Datetime']).dt.hour
    for hour in range(24):
        df[f'hour_{hour}'] = (df['hour'] == hour).astype(int)
    df = df.drop('hour', axis=1)
    return df

def perform_volatility_regression_analysis(
    raw_data_filepath: str,
    output_data_filepath: str
) -> None:
    raw_data_dfs = process_raw_data(raw_data_filepath)
    for sheet_name, df in raw_data_dfs.items():
        df = add_rolling_volatility(df)
        df = add_hourly_dummy_variables(df)
        df = df[(df['Offered Capacity'] != 0) & (df['Offered Capacity'] != '-')]

        hour_columns = [col for col in df.columns if col.startswith('hour_')]
        volatility_columns = [col for col in df.columns if 'rolling_volatility' in col]
        X_columns = volatility_columns + hour_columns
        X = df[X_columns].dropna()
        y = df.loc[X.index, 'difference']
        scaler = StandardScaler()
        X_scaled = X.copy()
        for vol_col in volatility_columns:
            X_scaled[vol_col] = scaler.fit_transform(X[[vol_col]])
    
        model = LinearRegression()
        model.fit(X_scaled, y)

        print(f"\nRegression Results for {sheet_name}:")
        print(f"R-squared: {r2_score(y, model.predict(X_scaled)):.4f}")
        print(f"Rolling Volatility Coefficient: {model.coef_[0]:.4f}")

        X_sm = sm.add_constant(X_scaled)
        model_sm = sm.OLS(y, X_sm).fit()
        summary_df = pd.DataFrame({
            'Variable': model_sm.params.index,
            'Coefficient': model_sm.params.values,
            'Std_Error': model_sm.bse.values,
            'P_Value': model_sm.pvalues.values,
            'Conf_Int_Lower': model_sm.conf_int()[0].values,
            'Conf_Int_Upper': model_sm.conf_int()[1].values
        })
        stats_df = pd.DataFrame({
            'Statistic': ['R-squared', 'Adj. R-squared', 'F-statistic', 'F-statistic P-value'],
            'Value': [model_sm.rsquared, model_sm.rsquared_adj, model_sm.fvalue, model_sm.f_pvalue]
        })

        with pd.ExcelWriter(output_data_filepath, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name=f"{sheet_name}_coefficients", index=False)
            stats_df.to_excel(writer, sheet_name=f"{sheet_name}_statistics", index=False)
        print(model_sm.summary())