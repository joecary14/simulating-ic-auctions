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
    
    calculate_rolling_correlations(forecast_uncertainties, rolling_window_days)
    
    return forecast_uncertainties

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
        df = df.with_columns(
            pl.col(ct.ColumnNames.DATE.value).str.strptime(pl.Date, fmt="%Y-%m-%d"),
        )
        
        df = df.sort(
            by=[ct.ColumnNames.DATE.value, ct.ColumnNames.DELIVERY_PERIOD.value]
        )
        
        df = df.with_columns(
            pl.col(ct.ColumnNames.DATE.value).dt.weekday().alias("weekday")
        )
        
        df = df.with_columns([
            pl.when(pl.col("weekday").is_in([1, 2, 3, 4]))
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
        
        reference_data = {}
        for row in df.select([
            "reference_key", 
            ct.ColumnNames.DOMESTIC_PRICE.value, 
            ct.ColumnNames.FOREIGN_PRICE.value
        ]).iter_rows():
            reference_data[row[0]] = (row[1], row[2])
        
        domestic_forecasts = []
        foreign_forecasts = []
        
        for row in df.select("reference_key").iter_rows():
            key = row[0]
            if key in reference_data:
                domestic_forecasts.append(reference_data[key][0])
                foreign_forecasts.append(reference_data[key][1])
            else:
                domestic_forecasts.append(None)
                foreign_forecasts.append(None)
        

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
    for interconnector, forecast_df in forecast_prices_by_ic.items():
        forecast_df = forecast_df.with_columns([
            (pl.col(ct.ColumnNames.FORECAST_DOMESTIC_PRICE.value) - 
             pl.col(ct.ColumnNames.DOMESTIC_PRICE.value))
            .alias(ct.ColumnNames.DOMESTIC_FORECAST_ERROR.value),
            (pl.col(ct.ColumnNames.FORECAST_FOREIGN_PRICE.value) - 
             pl.col(ct.ColumnNames.FOREIGN_PRICE.value))
            .alias(ct.ColumnNames.FOREIGN_FORECAST_ERROR.value)
        ])
        
    return forecast_prices_by_ic

def calculate_forecast_stdevs(
    forecast_error_dfs: dict[str, pl.DataFrame],
    rolling_window_days: int
) -> dict[str, pl.DataFrame]:
    for interconnector, df in forecast_error_dfs.items():
        df = df.with_columns([
            pl.col(ct.ColumnNames.DOMESTIC_FORECAST_ERROR.value).shift(1).alias("shifted_domestic_error"),
            pl.col(ct.ColumnNames.FOREIGN_FORECAST_ERROR.value).shift(1).alias("shifted_foreign_error"),
        ])
        
        df = df.groupby_rolling(
            index_column=ct.ColumnNames.DATE.value,
            period=f"{rolling_window_days}d"
        ).agg([
            pl.col("shifted_domestic_error").std().alias(ct.ColumnNames.DOMESTIC_FORECAST_ERROR_STDEV.value),
            pl.col("shifted_foreign_error").std().alias(ct.ColumnNames.FOREIGN_FORECAST_ERROR_STDEV.value),
        ])
        
        df = df.drop(["shifted_domestic_error", "shifted_foreign_error"])
        
        forecast_error_dfs[interconnector] = df
    
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

def calculate_rolling_correlations(
    forecast_error_dfs: dict[str, pl.DataFrame],
    rolling_window_days: int
) -> None:
    for ic, prices_df in forecast_error_dfs.items():
        df_with_correl = prices_df.clone()
        df_with_correl = df_with_correl.with_columns(
            pl.col(ct.ColumnNames.DATE.value).str.strptime(pl.Datetime, fmt="%Y-%m-%d").alias(ct.ColumnNames.DATE.value),
            ).with_columns(
             (pl.col(ct.ColumnNames.DATE.value).cast(pl.Datetime) + pl.col(ct.ColumnNames.DELIVERY_PERIOD.value)).cast(pl.Duration).apply(lambda x: x * 3600)).alias("timestamp")
        df = df.sort("timestamp")
        
        df = df.with_columns(
            pl.col(ct.ColumnNames.DOMESTIC_PRICE.value)
            .rolling_corr(pl.col(ct.ColumnNames.FOREIGN_PRICE.value), window_size=rolling_window_days*24, min_periods =(rolling_window_days-1)*24).alias("rolling_correlation")
        )
        
        forecast_error_dfs[ic] = df_with_correl

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