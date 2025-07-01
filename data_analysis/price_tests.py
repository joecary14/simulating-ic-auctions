import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import datetime
from typing import Dict
from scipy.stats import shapiro, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import rv_continuous
from sklearn.preprocessing import StandardScaler
from scipy.stats import t, norm, genpareto
from scipy.optimize import minimize

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
        
        
def add_rolling_volatility(df: pd.DataFrame):
    df['datetime'] = pd.to_datetime(df['UTC Datetime'])
    df = df.sort_values('datetime')
    df['day'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour

    weekly_volatility_data = []
    for day in df['day'].unique():
        week_ago = day - pd.Timedelta(days=7)
        
        # Get data from the past week for this specific hour
        past_week_mask = (
            (df['day'] >= week_ago) & 
            (df['day'] < day)
        )
        past_week_data = df[past_week_mask]
        
        if len(past_week_data) > 1:
            volatility_price_spread = past_week_data['outturn_price_spread'].std()
            volatility_gb_price = past_week_data['GB Price'].std()
            price_columns = [col for col in df.columns if 'price' in col.lower() and 'gb' not in col.lower()]
            volatility_other_price = past_week_data[price_columns[0]].std() if price_columns else np.nan
            volatility_market_expectation_price = past_week_data['market_expectation_price'].std()
        else:
            volatility_price_spread = np.nan
            volatility_gb_price = np.nan
            volatility_other_price = np.nan
            volatility_market_expectation_price = np.nan
            
        for hour in range(24):
            weekly_volatility_data.append({
                'day': day,
                'hour': hour,
                'weekly_volatility_price_spread': volatility_price_spread,
                'weekly_volatility_GB_price_spread': volatility_gb_price,
                'weekly_volatility_other_price': volatility_other_price,
                'weekly_volatility_market_expectation_price': volatility_market_expectation_price
            })

    volatility_df = pd.DataFrame(weekly_volatility_data)
    df = df.merge(volatility_df, on=['day', 'hour'], how='left')
    # df['rolling_volatility_price_spread'] = df['outturn_price_spread'].rolling(window=window, min_periods=1).std()
    # df['rolling_volatility_GB_price_spread'] = df['GB Price'].rolling(window=window, min_periods=1).std()
    # price_columns = [col for col in df.columns if 'price' in col.lower() and 'gb' not in col.lower()]
    # if price_columns:
    #     df[f'rolling_volatility_{price_columns[0]}'] = df[price_columns[0]].rolling(window=window).std()
    return df

def perform_volatility_regression_analysis(
    raw_data_filepath: str,
    output_data_filepath: str
) -> None:
    raw_data_dfs = process_raw_data(raw_data_filepath)
    
    for sheet_name, df in raw_data_dfs.items():
        df = add_rolling_volatility(df)
        # Add monthly dummy variables
        df['datetime'] = pd.to_datetime(df['UTC Datetime'])
        df['month'] = df['datetime'].dt.month

        # Create dummy variables for each month
        for month in range(1, 13):
            month_name = pd.to_datetime(f'2023-{month:02d}-01').strftime('%B')
            df[f'month_{month_name}'] = (df['month'] == month).astype(int)
        df = df[(df['Offered Capacity'] != 0) & (df['Offered Capacity'] != '-')]
        volatility_columns = [col for col in df.columns if 'weekly_volatility' in col]
        month_columns = [col for col in df.columns if col.startswith('month_')]
        regressor_columns = volatility_columns + month_columns
        X = df[regressor_columns].dropna()
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

def fit_distributions_and_compare_aic(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Fit t-distribution and normal distribution to difference data.
    Compare using AIC to test for heavy tails.
    """
    results = {}
    
    for sheet_name, df in dataframes.items():
        data = df['difference'].dropna()
        n = len(data)
        
        print(f"\n{'='*50}")
        print(f"DISTRIBUTION FITTING: {sheet_name}")
        print(f"{'='*50}")
        print(f"Sample size: {n}")
        
        normal_params = norm.fit(data)
        normal_mu, normal_sigma = normal_params
        
        normal_loglik = np.sum(norm.logpdf(data, loc=normal_mu, scale=normal_sigma))
        normal_aic = 2 * 2 - 2 * normal_loglik  # 2 parameters (mu, sigma)

        t_params = t.fit(data)
        t_df, t_loc, t_scale = t_params

        t_loglik = np.sum(t.logpdf(data, df=t_df, loc=t_loc, scale=t_scale))
        t_aic = 2 * 3 - 2 * t_loglik  # 3 parameters (df, loc, scale)

        aic_difference = normal_aic - t_aic
        
        print(f"\nNORMAL DISTRIBUTION:")
        print(f"  Parameters: μ = {normal_mu:.4f}, σ = {normal_sigma:.4f}")
        print(f"  Log-likelihood: {normal_loglik:.4f}")
        print(f"  AIC: {normal_aic:.4f}")
        
        print(f"\nt-DISTRIBUTION:")
        print(f"  Parameters: df = {t_df:.4f}, loc = {t_loc:.4f}, scale = {t_scale:.4f}")
        print(f"  Log-likelihood: {t_loglik:.4f}")
        print(f"  AIC: {t_aic:.4f}")
        
        print(f"\nAIC COMPARISON:")
        print(f"  AIC difference (Normal - t): {aic_difference:.4f}")
        
        if aic_difference > 2:
            print(f"  → t-distribution fits SIGNIFICANTLY better (heavy tails detected)")
            better_fit = "t-distribution"
        elif aic_difference < -2:
            print(f"  → Normal distribution fits significantly better")
            better_fit = "normal"
        else:
            print(f"  → No significant difference in fit")
            better_fit = "similar"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Histogram with fitted distributions (top-left)
        axes[0, 0].hist(data, bins=30, density=True, alpha=0.7, color='lightblue', 
                       edgecolor='black', label='Data')
        
        x_range = np.linspace(data.min(), data.max(), 100)
        axes[0, 0].plot(x_range, norm.pdf(x_range, normal_mu, normal_sigma), 
                       'r-', linewidth=2, label='Normal')
        axes[0, 0].plot(x_range, t.pdf(x_range, t_df, t_loc, t_scale), 
                       'g-', linewidth=2, label='t-distribution')
        
        axes[0, 0].set_xlabel('Difference')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title(f'{sheet_name}: Distribution Fits')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Subplot 2: Q-Q plot against fitted normal (top-right)
        fitted_normal = norm(loc=normal_mu, scale=normal_sigma)
        stats.probplot(data, dist=fitted_normal, plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q vs Fitted Normal')
        axes[0, 1].grid(alpha=0.3)
        
        # Subplot 3: Q-Q plot against fitted t-distribution (bottom-left)
        fitted_t = t(df=t_df, loc=t_loc, scale=t_scale)
        stats.probplot(data, dist=fitted_t, plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q vs Fitted t-distribution')
        axes[1, 0].grid(alpha=0.3)
        
        # Subplot 4: AIC comparison and parameters (bottom-right)
        axes[1, 1].axis('off')
        likelihood_ratio = np.exp(aic_difference / 2)
        
        text_content = f"""AIC COMPARISON & PARAMETERS
        
        BEST FIT: {better_fit.upper()}
        AIC Difference: {aic_difference:.2f}
        Likelihood Ratio: {likelihood_ratio:.2e}

        NORMAL DISTRIBUTION:
        μ = {normal_mu:.4f}
        σ = {normal_sigma:.4f}
        AIC = {normal_aic:.2f}

        t-DISTRIBUTION:
        df = {t_df:.4f}
        loc = {t_loc:.4f}
        scale = {t_scale:.4f}
        AIC = {t_aic:.2f}

        INTERPRETATION:
        """
        if aic_difference > 10:
            interpretation = "DECISIVE evidence for heavy tails"
        elif aic_difference > 4:
            interpretation = "STRONG evidence for heavy tails"
        elif aic_difference > 2:
            interpretation = "MODERATE evidence for heavy tails"
        elif aic_difference < -2:
            interpretation = "Evidence AGAINST heavy tails"
        else:
            interpretation = "Similar model support"
            
        text_content += interpretation
        
        # Add text to subplot
        axes[1, 1].text(0.05, 0.95, text_content, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Store results (unchanged)
        results[sheet_name] = {
            'normal_params': normal_params,
            'normal_aic': normal_aic,
            't_params': t_params,
            't_aic': t_aic,
            'aic_difference': aic_difference,
            'better_fit': better_fit,
            'heavy_tails': aic_difference > 2
        }
    
    return results

from scipy.stats import genpareto

def fit_gpd_and_compare_aic(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Fit Generalized Pareto Distribution and compare with normal and t-distribution.
    GPD is particularly good for modeling extreme values and heavy tails.
    """
    results = {}
    
    for sheet_name, df in dataframes.items():
        data = df['difference'].dropna()
        n = len(data)
        
        print(f"\n{'='*50}")
        print(f"GENERALIZED PARETO DISTRIBUTION FITTING: {sheet_name}")
        print(f"{'='*50}")
        print(f"Sample size: {n}")
        
        # 1. Fit Normal Distribution
        normal_params = norm.fit(data)
        normal_mu, normal_sigma = normal_params
        normal_loglik = np.sum(norm.logpdf(data, loc=normal_mu, scale=normal_sigma))
        normal_aic = 2 * 2 - 2 * normal_loglik  # 2 parameters
        
        # 2. Fit t-Distribution  
        t_params = t.fit(data)
        t_df, t_loc, t_scale = t_params
        t_loglik = np.sum(t.logpdf(data, df=t_df, loc=t_loc, scale=t_scale))
        t_aic = 2 * 3 - 2 * t_loglik  # 3 parameters
        
        # 3. Fit Generalized Pareto Distribution
        try:
            gpd_params = genpareto.fit(data)
            gpd_c, gpd_loc, gpd_scale = gpd_params  # c=shape, loc=location, scale=scale
            gpd_loglik = np.sum(genpareto.logpdf(data, c=gpd_c, loc=gpd_loc, scale=gpd_scale))
            gpd_aic = 2 * 3 - 2 * gpd_loglik  # 3 parameters
            gpd_fit_success = True
        except Exception as e:
            print(f"Warning: GPD fitting failed: {e}")
            gpd_params = None
            gpd_aic = np.inf
            gpd_fit_success = False
        
        # Print results
        print(f"\nNORMAL DISTRIBUTION:")
        print(f"  Parameters: μ = {normal_mu:.4f}, σ = {normal_sigma:.4f}")
        print(f"  Log-likelihood: {normal_loglik:.4f}")
        print(f"  AIC: {normal_aic:.4f}")
        
        print(f"\nt-DISTRIBUTION:")
        print(f"  Parameters: df = {t_df:.4f}, loc = {t_loc:.4f}, scale = {t_scale:.4f}")
        print(f"  Log-likelihood: {t_loglik:.4f}")
        print(f"  AIC: {t_aic:.4f}")
        
        if gpd_fit_success:
            print(f"\nGENERALIZED PARETO DISTRIBUTION:")
            print(f"  Parameters: c = {gpd_c:.4f}, loc = {gpd_loc:.4f}, scale = {gpd_scale:.4f}")
            print(f"  Log-likelihood: {gpd_loglik:.4f}")
            print(f"  AIC: {gpd_aic:.4f}")
            
            # Interpret shape parameter
            if gpd_c > 0:
                print(f"  Shape parameter c > 0: Heavy tails (Pareto-type)")
            elif gpd_c == 0:
                print(f"  Shape parameter c = 0: Exponential-type tails")
            else:
                print(f"  Shape parameter c < 0: Bounded distribution")
        
        # AIC Comparisons
        print(f"\nAIC COMPARISONS:")
        aic_diff_normal_t = normal_aic - t_aic
        print(f"  Normal vs t-distribution: {aic_diff_normal_t:.4f}")
        
        if gpd_fit_success:
            aic_diff_normal_gpd = normal_aic - gpd_aic
            aic_diff_t_gpd = t_aic - gpd_aic
            print(f"  Normal vs GPD: {aic_diff_normal_gpd:.4f}")
            print(f"  t-distribution vs GPD: {aic_diff_t_gpd:.4f}")
            
            # Determine best fit
            aics = {'Normal': normal_aic, 't-distribution': t_aic, 'GPD': gpd_aic}
            best_dist = min(aics, key=aics.get)
            print(f"\n  BEST FIT (lowest AIC): {best_dist} (AIC = {aics[best_dist]:.4f})")
            
            # Significant differences (AIC difference > 2)
            if aic_diff_normal_gpd > 2:
                print(f"  → GPD fits SIGNIFICANTLY better than Normal")
            if aic_diff_t_gpd > 2:
                print(f"  → GPD fits SIGNIFICANTLY better than t-distribution")
            elif aic_diff_t_gpd < -2:
                print(f"  → t-distribution fits SIGNIFICANTLY better than GPD")
            else:
                print(f"  → GPD and t-distribution have similar fit quality")
        
        # Create comparison plots
        plt.figure(figsize=(20, 5))
        
        # Subplot 1: Histogram with all fitted distributions
        plt.subplot(1, 4, 1)
        plt.hist(data, bins=30, density=True, alpha=0.7, color='lightblue', 
                edgecolor='black', label='Data')
        
        x_range = np.linspace(data.min(), data.max(), 200)
        plt.plot(x_range, norm.pdf(x_range, normal_mu, normal_sigma), 
                'r-', linewidth=2, label='Normal')
        plt.plot(x_range, t.pdf(x_range, t_df, t_loc, t_scale), 
                'g-', linewidth=2, label='t-distribution')
        
        if gpd_fit_success:
            plt.plot(x_range, genpareto.pdf(x_range, gpd_c, gpd_loc, gpd_scale), 
                    'orange', linewidth=2, label='GPD')
        
        plt.xlabel('Difference')
        plt.ylabel('Density')
        plt.title(f'{sheet_name}: Distribution Fits')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Subplot 2: Q-Q plot against normal
        plt.subplot(1, 4, 2)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title('Q-Q vs Normal')
        plt.grid(alpha=0.3)
        
        # Subplot 3: Q-Q plot against t-distribution
        plt.subplot(1, 4, 3)
        fitted_t = t(df=t_df, loc=t_loc, scale=t_scale)
        stats.probplot(data, dist=fitted_t, plot=plt)
        plt.title('Q-Q vs t-distribution')
        plt.grid(alpha=0.3)
        
        # Subplot 4: Q-Q plot against GPD (if successful)
        plt.subplot(1, 4, 4)
        if gpd_fit_success:
            fitted_gpd = genpareto(c=gpd_c, loc=gpd_loc, scale=gpd_scale)
            stats.probplot(data, dist=fitted_gpd, plot=plt)
            plt.title('Q-Q vs GPD')
        else:
            plt.text(0.5, 0.5, 'GPD Fit Failed', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('GPD Fit Failed')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Store results
        results[sheet_name] = {
            'normal_params': normal_params,
            'normal_aic': normal_aic,
            't_params': t_params,
            't_aic': t_aic,
            'gpd_params': gpd_params if gpd_fit_success else None,
            'gpd_aic': gpd_aic if gpd_fit_success else np.inf,
            'gpd_fit_success': gpd_fit_success,
            'best_distribution': min(aics, key=aics.get) if gpd_fit_success else ('t-distribution' if t_aic < normal_aic else 'Normal'),
            'heavy_tails_gpd': gpd_c > 0 if gpd_fit_success else False
        }
    
    return results