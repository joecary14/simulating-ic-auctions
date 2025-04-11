import polars as pl
import numpy as np
import constants as ct
import auction_simulation.auction_information as auction_information

def simulate_day(
    forecast_prices_with_errors_one_day : pl.DataFrame,
    covariance_matrix : np.ndarray,
    number_of_generators : int,
    alpha_by_generator : dict[str, float],
    beta_by_generator : dict[str, float],
    bid_capacity_by_generator : pl.DataFrame,
    generator_marginal_cost : float,
    generator_capacity : int,
) -> pl.DataFrame:
    
    auction_information_one_day = get_auction_information_one_sim(
        forecast_prices_with_errors_one_day,
        covariance_matrix,
        number_of_generators,
        alpha_by_generator,
        beta_by_generator,
        bid_capacity_by_generator,
        generator_marginal_cost
    )
    profits_by_generator = calculate_profit_by_generator_one_sim(
        auction_information_one_day,
        generator_capacity,
        generator_marginal_cost
    )
    
    return profits_by_generator

def get_auction_information_one_sim(
    forecast_prices_with_errors_one_day : pl.DataFrame,
    covariance_matrix : np.ndarray,
    number_of_generators : int,
    alpha_by_generator : dict[str, float],
    beta_by_generator : dict[str, float],
    bid_capacity_by_generator : pl.DataFrame,
    generator_marginal_cost : float
) -> auction_information.AuctionInformation:
    
    periods = sorted(forecast_prices_with_errors_one_day[ct.ColumnNames.DELIVERY_PERIOD.value].unique().to_list())
    actual_domestic_price = []
    actual_foreign_price = []
    bids_by_generator_by_period = {}
    for period in periods:
        period_data = forecast_prices_with_errors_one_day.filter(
            pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
        )
        domestic_forecast = period_data[ct.ColumnNames.FORECAST_DOMESTIC_PRICE.value][0]
        foreign_forecast = period_data[ct.ColumnNames.FORECAST_FOREIGN_PRICE.value][0]
        
        samples = np.random.multivariate_normal(
            [0, 0], covariance_matrix, size=number_of_generators + 1
        )
        
        domestic_prices = domestic_forecast + samples[:, 0]
        foreign_prices = foreign_forecast + samples[:, 1]
        actual_domestic_price.append(domestic_prices[0])
        actual_foreign_price.append(foreign_prices[0])
        
        bids_by_generator = get_bids_by_generator(
            domestic_prices[1:],
            foreign_prices[1:],
            alpha_by_generator,
            beta_by_generator,
            generator_marginal_cost
        )
        
        bids_by_generator_by_period[period] = bids_by_generator
    
    rows = []
    for period, bids_dict in bids_by_generator_by_period.items():
        row = {ct.ColumnNames.DELIVERY_PERIOD.value: period}
        row.update(bids_dict)
        rows.append(row)
    
    bids_df = pl.DataFrame(rows)
    bids_df = bids_df.sort(
        pl.col(ct.ColumnNames.DELIVERY_PERIOD.value)
    )
    
    auction_information_one_day = auction_information.AuctionInformation(
        actual_domestic_prices = np.array(actual_domestic_price),
        actual_foreign_prices = np.array(actual_foreign_price),
        bids_by_generator_by_period = bids_df,
        capacity_by_generator_by_period = bid_capacity_by_generator,
        capacity_offered = forecast_prices_with_errors_one_day[ct.ColumnNames.AVAILABLE_CAPACITY.value].to_numpy()
    )
    
    return auction_information_one_day
        
def get_bids_by_generator(
    export_market_prices: np.ndarray,
    domestic_market_prices: np.ndarray,
    alpha_by_generator: dict[str, float],
    beta_by_generator: dict[str, float],
    generator_marginal_cost: float
):
    option_values = np.maximum(export_market_prices - domestic_market_prices, 0)
    option_values[export_market_prices <= generator_marginal_cost] = 0
    
    bid_prices = {str(generator_id) : alpha_by_generator[generator_id] + beta_by_generator[generator_id] * option_values[generator_id] for generator_id in alpha_by_generator.keys()}
   
    return bid_prices

def calculate_profit_by_generator_one_sim(
    auction_information_one_sim : auction_information.AuctionInformation,
    generator_capacity : int,
    generator_marginal_cost : float,
) -> pl.DataFrame:
    
    auction_results, clearing_prices = auction_information_one_sim.run_auction()
    capacity_for_domestic_market = generator_capacity - auction_results
    
    domestic_prices = auction_information_one_sim.actual_domestic_prices.copy()
    foreign_prices = auction_information_one_sim.actual_foreign_prices.copy()
    
    domestic_prices[domestic_prices < generator_marginal_cost] = 0
    foreign_prices[foreign_prices < generator_marginal_cost] = 0
    net_foreign_prices = foreign_prices - clearing_prices
    
    profits_by_generator = (
        capacity_for_domestic_market * domestic_prices
        + auction_results * net_foreign_prices
    )
    
    daily_profits = profits_by_generator.sum(axis=0)
    generator_ids = auction_results.columns
    
    profits_df = pl.DataFrame({col : profit for col, profit in zip(generator_ids, daily_profits)})
        
    return profits_df

def get_covariance_matrix(
    forecast_prices_with_errors_one_day : pl.DataFrame
) -> dict[str, np.ndarray]:
    forecast_error_correlation = forecast_prices_with_errors_one_day[ct.ColumnNames.FORECAST_ERROR_CORRELATIONS.value][0]
    domestic_stdev = forecast_prices_with_errors_one_day[ct.ColumnNames.DOMESTIC_FORECAST_ERROR_STDEV.value][0]
    foreign_stdev = forecast_prices_with_errors_one_day[ct.ColumnNames.FOREIGN_FORECAST_ERROR_STDEV.value][0]
    
    corr_matrix = np.array([
    [1.0, forecast_error_correlation],
    [forecast_error_correlation, 1.0]
    ])
    std_vector = np.array([domestic_stdev, foreign_stdev])
    cov_matrix = np.outer(std_vector, std_vector) * corr_matrix
    
    return cov_matrix