import math
import polars as pl
import constants as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from datetime import timedelta

def get_naive_forecasts(
    raw_auction_data_dfs: dict[str, pl.DataFrame],
    rolling_window_days: int = 30
) -> dict[str, pl.DataFrame]:
    naive_forecasts = create_naive_forecast_by_date_and_period(raw_auction_data_dfs)
    
    forecast_errors = calculate_forecast_errors(naive_forecasts)
    
    forecast_uncertainties = calculate_forecast_stdevs(forecast_errors, rolling_window_days)
    
    forecast_errors_with_rolling_stdevs_and_correlations = calculate_rolling_correlations(forecast_uncertainties, rolling_window_days)
    
    return forecast_errors_with_rolling_stdevs_and_correlations

def create_naive_forecast_by_date_and_period(
    raw_auction_data_dfs: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    
    forecasts_by_country = {}
    forecasts_by_interconnector = {}
    
    for interconnector, raw_auction_data_df in raw_auction_data_dfs.items():
        source_country = ct.ic_name_to_source_country_dict[interconnector]
        if source_country in forecasts_by_country:
            forecasts_by_interconnector[interconnector] = forecasts_by_country[source_country]
            continue
        
        df = raw_auction_data_df.clone()
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        df = df.with_columns(
            pl.col(ct.ColumnNames.DATE.value).str.strptime(pl.Date),
        )
        
        df = df.sort(
            by=[ct.ColumnNames.DATE.value, ct.ColumnNames.DELIVERY_PERIOD.value]
        )
        
        df = df.with_columns(
            pl.col(ct.ColumnNames.DATE.value).dt.weekday().alias("weekday")
        )
        
        df = df.with_columns([
            pl.when(pl.col("weekday").is_in([2, 3, 4, 5]))
              .then(pl.col(ct.ColumnNames.DATE.value) - timedelta(days=1))
              .otherwise(pl.col(ct.ColumnNames.DATE.value) - timedelta(days=7))
              .alias("reference_date")
        ])
        
        df = df.with_columns([
            pl.struct([
                pl.col("reference_date"),
                pl.col(ct.ColumnNames.DELIVERY_PERIOD.value)
            ]).alias("reference_key"),
            pl.struct([
                pl.col(ct.ColumnNames.DATE.value),
                pl.col(ct.ColumnNames.DELIVERY_PERIOD.value)
            ]).alias("current_key")
        ])
        
        current_price_data = {}
        for row in df.select([
            "current_key", 
            ct.ColumnNames.DOMESTIC_PRICE.value, 
            ct.ColumnNames.FOREIGN_PRICE.value
        ]).iter_rows():
            key = tuple(row[0].values())
            current_price_data[key] = (row[1], row[2])
        
        domestic_forecasts = []
        foreign_forecasts = []
        
        for row in df.select("reference_key").iter_rows():
            key = tuple(row[0].values())
            if key in current_price_data:
                domestic_forecasts.append(current_price_data[key][0])
                foreign_forecasts.append(current_price_data[key][1])
            else:
                domestic_forecasts.append(math.nan)
                foreign_forecasts.append(math.nan)
        

        df_with_forecasts = df.with_columns([
            pl.Series(ct.ColumnNames.FORECAST_DOMESTIC_PRICE.value, domestic_forecasts).cast(pl.Float64),
            pl.Series(ct.ColumnNames.FORECAST_FOREIGN_PRICE.value, foreign_forecasts).cast(pl.Float64),
        ])
        
        df_with_forecasts = df_with_forecasts.drop(["reference_date", "reference_key", "current_key"])
        
        forecasts_by_country[source_country] = df_with_forecasts
        forecasts_by_interconnector[interconnector] = df_with_forecasts
    
    return forecasts_by_interconnector

def calculate_forecast_errors(
    forecast_prices_by_ic: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    forecast_prices_and_errors_by_ic = {}
    for interconnector, forecast_df in forecast_prices_by_ic.items():
        forecast_with_errors = forecast_df.clone()
        forecast_with_errors = forecast_with_errors.drop_nans()
        forecast_with_errors = forecast_with_errors.with_columns([
            (pl.col(ct.ColumnNames.FORECAST_DOMESTIC_PRICE.value) - 
             pl.col(ct.ColumnNames.DOMESTIC_PRICE.value))
            .alias(ct.ColumnNames.DOMESTIC_FORECAST_ERROR.value),
            (pl.col(ct.ColumnNames.FORECAST_FOREIGN_PRICE.value) - 
             pl.col(ct.ColumnNames.FOREIGN_PRICE.value))
            .alias(ct.ColumnNames.FOREIGN_FORECAST_ERROR.value)
        ])
        forecast_prices_and_errors_by_ic[interconnector] = forecast_with_errors
        
    return forecast_prices_and_errors_by_ic

def calculate_forecast_stdevs(
    forecast_error_dfs: dict[str, pl.DataFrame],
    rolling_window_days: int
) -> dict[str, pl.DataFrame]:
    for interconnector, df in forecast_error_dfs.items():
        df_copy = df.clone()
        df_copy = df_copy.with_columns(pl.col(ct.ColumnNames.DATE.value).cast(pl.Date))
        unique_dates = df_copy[ct.ColumnNames.DATE.value].unique().to_numpy()
        all_dates = []
        domestic_stds = []
        foreign_stds = []
        for current_date in unique_dates:
            start_date = current_date - np.timedelta64(rolling_window_days, 'D')
            filtered_df = df_copy.filter(
                (pl.col(ct.ColumnNames.DATE.value) >= start_date) & 
                (pl.col(ct.ColumnNames.DATE.value) < current_date)
            )
            if len(filtered_df) < rolling_window_days*24 - 1:
                domestic_std = np.nan
                foreign_std = np.nan
            else:
                domestic_std = filtered_df[ct.ColumnNames.DOMESTIC_FORECAST_ERROR.value].std()
                foreign_std = filtered_df[ct.ColumnNames.FOREIGN_FORECAST_ERROR.value].std()
            
            python_date = current_date.astype('datetime64[D]').astype(object)
            all_dates.append(python_date)
            domestic_stds.append(domestic_std)
            foreign_stds.append(foreign_std)
        
        temp_df = pl.DataFrame({
            ct.ColumnNames.DATE.value: pl.Series(all_dates, dtype=pl.Date),
            ct.ColumnNames.DOMESTIC_FORECAST_ERROR_STDEV.value: pl.Series(domestic_stds, dtype=pl.Float64),
            ct.ColumnNames.FOREIGN_FORECAST_ERROR_STDEV.value: pl.Series(foreign_stds, dtype=pl.Float64)
        })
        
        df_copy = df_copy.join(
            temp_df,
            on=ct.ColumnNames.DATE.value,
            how="left"
        )
        df_copy = df_copy.drop_nans()
        
        forecast_error_dfs[interconnector] = df_copy
    
    return forecast_error_dfs

#TODO - fix this function, as above
def calculate_rolling_correlations(
    forecast_error_dfs: dict[str, pl.DataFrame],
    rolling_window_days: int
) -> dict[str, pl.DataFrame]:
    for ic, prices_df in forecast_error_dfs.items():
        df_with_correl = prices_df.clone()
        unique_dates = df_with_correl[ct.ColumnNames.DATE.value].unique().to_numpy()
        all_dates = []
        forecast_error_correlations = []
        for current_date in unique_dates:
            start_date = current_date - np.timedelta64(rolling_window_days, 'D')
            filtered_df = df_with_correl.filter(
                (pl.col(ct.ColumnNames.DATE.value) >= start_date) & 
                (pl.col(ct.ColumnNames.DATE.value) < current_date)
            )
            if len(filtered_df) < rolling_window_days*24 - 1:
                correlation = np.nan
            else:
                correlation = filtered_df.select(
                    pl.corr(
                        ct.ColumnNames.DOMESTIC_FORECAST_ERROR.value,
                        ct.ColumnNames.FOREIGN_FORECAST_ERROR.value
                    )
                ).to_numpy()[0][0]
            
            python_date = current_date.astype('datetime64[D]').astype(object)
            all_dates.append(python_date)
            forecast_error_correlations.append(correlation)
        
        temp_df = pl.DataFrame({
            ct.ColumnNames.DATE.value: pl.Series(all_dates, dtype=pl.Date),
            ct.ColumnNames.DOMESTIC_FORECAST_ERROR_STDEV.value: pl.Series(forecast_error_correlations, dtype=pl.Float64)
        })
        
        df_with_correl = df_with_correl.join(
            temp_df,
            on=ct.ColumnNames.DATE.value,
            how="left"
        )
        df_with_correl = df_with_correl.drop_nans()
        
        forecast_error_dfs[ic] = df_with_correl
    
    return forecast_error_dfs

def check_error_normality(
    forecast_error_dfs: dict[str, pl.DataFrame]
) -> dict[str, dict[str, bool]]:
    
    normality_results = {}
    
    for interconnector, df in forecast_error_dfs.items():
        domestic_errors = df.filter(
            pl.col(ct.ColumnNames.DOMESTIC_FORECAST_ERROR.value).is_not_null()
        )[ct.ColumnNames.DOMESTIC_FORECAST_ERROR.value].to_numpy()
        
        foreign_errors = df.filter(
            pl.col(ct.ColumnNames.FOREIGN_FORECAST_ERROR.value).is_not_null()
        )[ct.ColumnNames.FOREIGN_FORECAST_ERROR.value].to_numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Normality Analysis for {interconnector}', fontsize=16)
        
        is_domestic_normal = analyze_errors(
            domestic_errors, 
            f"Domestic Forecast Errors ({interconnector})",
            axes[0, 0],
            axes[0, 1],
        )
        
        is_foreign_normal = analyze_errors(
            foreign_errors, 
            f"Foreign Forecast Errors ({interconnector})",
            axes[1, 0],
            axes[1, 1],
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title
        plt.show()
        
        normality_results[interconnector] = {
            "domestic_errors_normal": is_domestic_normal,
            "foreign_errors_normal": is_foreign_normal
        }
    
    return normality_results

def analyze_errors(
    errors: np.ndarray, 
    title: str,
    hist_ax: plt.Axes,
    qq_ax: plt.Axes,
    alpha: float = 0.05
) -> bool:
    
    errors = errors[~np.isnan(errors)]
    
    if len(errors) == 0:
        hist_ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        qq_ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return False
    
    # Perform Shapiro-Wilk test for normality
    stat, p_value = stats.shapiro(errors)
    is_normal = p_value > alpha
    
    # Create histogram with normal distribution overlay
    hist_ax.hist(errors, bins=30, density=True, alpha=0.6, color='lightblue')
    
    # Fit normal distribution and plot PDF
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    hist_ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
    
    # Add test results to plot
    hist_ax.set_title(f"{title} Distribution\nShapiro-Wilk Test: p={p_value:.4f} ({'Normal' if is_normal else 'Not Normal'})")
    hist_ax.set_xlabel("Error Value")
    hist_ax.set_ylabel("Density")
    
    # Add mean and standard deviation to plot
    hist_ax.axvline(mu, color='k', linestyle='--', alpha=0.7, label=f'Mean: {mu:.2f}')
    hist_ax.axvline(mu + sigma, color='g', linestyle='--', alpha=0.7, label=f'Std Dev: {sigma:.2f}')
    hist_ax.axvline(mu - sigma, color='g', linestyle='--', alpha=0.7)
    hist_ax.legend()
    
    # Create Q-Q plot
    stats.probplot(errors, dist="norm", plot=qq_ax)
    qq_ax.set_title(f"Q-Q Plot for {title}")
    
    return is_normal