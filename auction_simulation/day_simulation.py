import polars as pl
import numpy as np
from scipy.linalg import cholesky
import constants as ct

def run_day_simulation(
    forecast_prices_with_errors_by_ic: dict[str, pl.DataFrame],
    number_of_simulations: int,
    number_of_generators: int
) -> dict[str, pl.DataFrame]:
    for interconnector, forecast_prices_with_errors in forecast_prices_with_errors_by_ic.items():
        dates = forecast_prices_with_errors[ct.ColumnNames.DATE.value].unique().to_list()
        
        for date in dates:
            day_data = forecast_prices_with_errors.filter(
                pl.col(ct.ColumnNames.DATE.value) == date
            )

def run_simulations_one_day(
    day_data: pl.DataFrame,
    interconnector: str,
    number_of_simulations: int,
    number_of_generators: int
) -> pl.DataFrame:

    # Get correlation between domestic and foreign prices
    # This could be calculated from historical data or supplied as a parameter
    price_correlation = 0.7  # Example value, replace with actual correlation
    
    # Extract the required data for each period
    all_periods_results = []
    
    for period in day_data[ct.ColumnNames.DELIVERY_PERIOD.value].unique().sort():
        period_data = day_data.filter(
            pl.col(ct.ColumnNames.DELIVERY_PERIOD.value) == period
        )
        
        # Extract the date
        date = period_data[ct.ColumnNames.DATE.value][0]
        
        # Get the forecasts and standard deviations
        domestic_forecast = period_data[ct.ColumnNames.FORECAST_DOMESTIC_PRICE.value][0]
        foreign_forecast = period_data[ct.ColumnNames.FORECAST_FOREIGN_PRICE.value][0]
        
        domestic_stdev = period_data[ct.ColumnNames.DOMESTIC_FORECAST_ERROR_STDEV.value][0]
        foreign_stdev = period_data[ct.ColumnNames.FOREIGN_FORECAST_ERROR_STDEV.value][0]
        
        # Generate correlated random samples for this period across all simulations
        domestic_prices, foreign_prices = generate_correlated_prices(
            domestic_forecast, 
            foreign_forecast,
            domestic_stdev,
            foreign_stdev,
            price_correlation,
            number_of_simulations
        )
        
        # Process each simulation
        period_results = []
        for sim_idx in range(number_of_simulations):
            domestic_price = domestic_prices[sim_idx]
            foreign_price = foreign_prices[sim_idx]
            
            # For each generator, calculate profit
            for gen_idx in range(1, number_of_generators + 1):
                # Calculate profit (replace with your actual profit calculation)
                profit = calculate_generator_profit(
                    gen_idx, 
                    domestic_price, 
                    foreign_price,
                    interconnector
                )
                
                # Append result
                period_results.append({
                    ct.ColumnNames.DATE.value: date,
                    ct.ColumnNames.DELIVERY_PERIOD.value: period,
                    "simulation": sim_idx,
                    "generator": gen_idx,
                    "profit": profit,
                    "domestic_price": domestic_price,
                    "foreign_price": foreign_price
                })
        
        all_periods_results.extend(period_results)
    
    # Convert to DataFrame
    return pl.DataFrame(all_periods_results)

def generate_correlated_prices(
    domestic_forecast: float,
    foreign_forecast: float,
    domestic_stdev: float,
    foreign_stdev: float,
    correlation: float,
    num_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated random price samples based on forecasts and standard deviations.
    
    Args:
        domestic_forecast: Forecasted domestic price
        foreign_forecast: Forecasted foreign price
        domestic_stdev: Standard deviation of domestic forecast errors
        foreign_stdev: Standard deviation of foreign forecast errors
        correlation: Correlation coefficient between domestic and foreign prices
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of arrays (domestic_prices, foreign_prices)
    """
    # Create correlation matrix
    corr_matrix = np.array([
        [1.0, correlation],
        [correlation, 1.0]
    ])
    
    # Create covariance matrix
    std_vector = np.array([domestic_stdev, foreign_stdev])
    cov_matrix = np.outer(std_vector, std_vector) * corr_matrix
    
    # Generate multivariate normal random samples
    try:
        # Use Cholesky decomposition for more efficient sampling
        L = cholesky(cov_matrix, lower=True)
        uncorrelated_samples = np.random.normal(0, 1, (num_samples, 2))
        correlated_samples = uncorrelated_samples @ L.T
    except np.linalg.LinAlgError:
        # Fallback if Cholesky decomposition fails
        correlated_samples = np.random.multivariate_normal(
            [0, 0], cov_matrix, size=num_samples
        )
    
    # Add forecasts to get the final prices
    domestic_prices = domestic_forecast + correlated_samples[:, 0]
    foreign_prices = foreign_forecast + correlated_samples[:, 1]
    
    # Ensure prices don't go negative (optional)
    domestic_prices = np.maximum(domestic_prices, 0)
    foreign_prices = np.maximum(foreign_prices, 0)
    
    return domestic_prices, foreign_prices

def calculate_generator_profit(
    generator_id: int,
    domestic_price: float,
    foreign_price: float,
    interconnector: str
) -> float:
    """
    Calculate the profit for a generator based on simulated prices.
    
    Args:
        generator_id: ID of the generator
        domestic_price: Simulated domestic price
        foreign_price: Simulated foreign price
        interconnector: Name of the interconnector
        
    Returns:
        Calculated profit
    """
    # Replace with your actual profit calculation logic
    # This is just a placeholder example
    price_difference = foreign_price - domestic_price
    
    # Simple model: profit is proportional to price difference
    # With some randomness for generator characteristics
    generator_factor = 0.8 + (generator_id / 10)  # Example factor
    
    # Example profit calculation
    if price_difference > 0:  # Exporting scenario
        profit = price_difference * generator_factor
    else:  # Importing scenario
        profit = abs(price_difference) * (generator_factor * 0.5)  # Less profit in import scenarios
    
    return profit