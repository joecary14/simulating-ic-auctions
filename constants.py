from enum import Enum

ic_name_to_source_country_dict = {
    "IFA 1" : "France",
    "IFA 2" : "France",
    "NemoLink" : "Belgium",
    "BritNed" : "Netherlands"
}

class ColumnNames(Enum):
    AVAILABLE_CAPACITY = "available_capacity"
    CLEARING_PRICE = "clearing_price"
    DATE = "date"
    DELIVERY_PERIOD = "delivery_period"
    DOMESTIC_FORECAST_ERROR = "domestic_forecast_error"
    DOMESTIC_FORECAST_ERROR_STDEV = "domestic_forecast_error_stdev"
    DOMESTIC_PRICE = "domestic_price"
    FORECAST_DOMESTIC_PRICE = "forecast_domestic_price"
    FORECAST_ERROR_CORRELATIONS = "forecast_error_correlations"
    FORECAST_FOREIGN_PRICE = "forecast_foreign_price"
    FOREIGN_FORECAST_ERROR = "foreign_forecast_error"
    FOREIGN_FORECAST_ERROR_STDEV = "foreign_forecast_error_stdev"
    FOREIGN_PRICE = "foreign_price"
    ROLLING_CORRELATION = "rolling_correlation"
    
class NumericalConstants(Enum):
    DEFAULT_UTILITY = -1e10
    