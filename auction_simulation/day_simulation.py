import polars as pl
import numpy as np
import constants as ct
import auction_simulation.auction_information as auction_information

def simulate_day(
    forecast_prices_with_errors_one_day : pl.DataFrame,
    covariance_matrix_by_period : dict[str, np.ndarray],
    number_of_generators : int,
    alpha_by_generator : pl.DataFrame,
    beta_by_generator : pl.DataFrame,
    bid_capacity_by_generator : pl.DataFrame,
    generator_marginal_cost : float,
    generator_capacity : int,
) -> pl.DataFrame:
    
    auction_information_one_day = get_auction_information_one_sim(
        forecast_prices_with_errors_one_day,
        covariance_matrix_by_period,
        number_of_generators,
        alpha_by_generator,
        beta_by_generator,
        generator_marginal_cost,
        bid_capacity_by_generator
    )
    profits_by_generator = calculate_profit_by_generator_one_sim(
        auction_information_one_day,
        generator_capacity,
        generator_marginal_cost
    )
    
    return profits_by_generator

def get_auction_information_one_sim(
    forecast_prices_with_errors_one_day : pl.DataFrame,
    covariance_matrix_by_period : dict[str, np.ndarray],
    number_of_generators : int,
    alpha_by_generator : np.ndarray,
    beta_by_generator : np.ndarray,
    bid_capacity_by_generator : dict[str, np.ndarray],
    generator_marginal_cost : float
) -> auction_information.AuctionInformation:
    
    periods = forecast_prices_with_errors_one_day[ct.ColumnNames.DELIVERY_PERIOD.value].unique().to_list()
    actual_domestic_price = []
    actual_foreign_price = []
    bids_by_generator_by_period = {}
    for period in periods:
        period_data = forecast_prices_with_errors_one_day.filter(
            pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
        )
        domestic_forecast = period_data[ct.ColumnNames.FORECAST_DOMESTIC_PRICE.value][0]
        foreign_forecast = period_data[ct.ColumnNames.FORECAST_FOREIGN_PRICE.value][0]
        cov_matrix = covariance_matrix_by_period[period]
        
        samples = np.random.multivariate_normal(
            [0, 0], cov_matrix, size=number_of_generators + 1
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
    
    bids_by_generator_by_period = pl.DataFrame(bids_by_generator_by_period)  
    
    auction_information_one_day = auction_information.AuctionInformation(
        actual_domestic_prices = np.array(actual_domestic_price),
        actual_foreign_prices = np.array(actual_foreign_price),
        bids_by_generator_by_period = bids_by_generator_by_period,
        capacity_by_generator_by_period = bid_capacity_by_generator,
        capacity_offered = forecast_prices_with_errors_one_day[ct.ColumnNames.AVAILABLE_CAPACITY.value].to_numpy()
    )
    
    return auction_information_one_day
        
def get_bids_by_generator(
    forecast_domestic_prices: np.ndarray,
    forecast_foreign_prices: np.ndarray,
    alpha_by_generator: np.ndarray,
    beta_by_generator: np.ndarray,
    generator_marginal_cost: float
):
    option_values = np.maximum(forecast_foreign_prices - forecast_domestic_prices, 0)
    option_values[forecast_foreign_prices <= generator_marginal_cost] = 0
    
    bid_prices = alpha_by_generator + beta_by_generator * option_values
   
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

def get_covariance_matrix_by_period(
    forecast_prices_with_errors_one_day : pl.DataFrame
) -> dict[str, np.ndarray]:
    periods = forecast_prices_with_errors_one_day[ct.ColumnNames.DELIVERY_PERIOD.value].unique().to_list()
    covariance_matrices = {}
    for period in periods:
        period_data = forecast_prices_with_errors_one_day.filter(
            pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
        )
        price_correlation = period_data[ct.ColumnNames.ROLLING_CORRELATION.value][0]
        domestic_stdev = period_data[ct.ColumnNames.DOMESTIC_FORECAST_ERROR_STDEV.value][0]
        foreign_stdev = period_data[ct.ColumnNames.FOREIGN_FORECAST_ERROR_STDEV.value][0]
        
        corr_matrix = np.array([
        [1.0, price_correlation],
        [price_correlation, 1.0]
        ])
        std_vector = np.array([domestic_stdev, foreign_stdev])
        cov_matrix = np.outer(std_vector, std_vector) * corr_matrix
        covariance_matrices[period] = cov_matrix
    
    return covariance_matrices