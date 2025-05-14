import pandas as pd
from elexonpy.api_client import ApiClient
from elexonpy.api.demand_forecast_api import DemandForecastApi
from elexonpy.api.generation_forecast_api import GenerationForecastApi
import data_handler.datetime_functions as datetime_functions
import data_handler.elexon_interaction as elexon_interaction

async def get_elexon_lear_data_for_year(
    year: int,
) -> dict[str, pd.DataFrame]:
    start_date, end_date = datetime_functions.get_start_and_end_dates_from_year(year)
    settlement_dates_with_periods_per_day = datetime_functions.get_settlement_dates_and_settlement_periods_per_day(
        start_date_str=start_date,
        end_date_str=end_date,
        convert_datetime_to_string=True,
    )
    api_client = ApiClient()
    demand_forecast_api = DemandForecastApi(api_client)
    generation_forecast_api = GenerationForecastApi(api_client)
    
    demand_forecasts = await elexon_interaction.get_latest_actionable_forecasts_for_date_range(
        settlement_dates_with_periods_per_day=settlement_dates_with_periods_per_day,
        api_function=demand_forecast_api.forecast_demand_daily_evolution_get
    )
    
    wind_forecasts = await elexon_interaction.get_latest_actionable_forecasts_for_date_range(
        settlement_dates_with_periods_per_day=settlement_dates_with_periods_per_day,
        api_function=generation_forecast_api.forecast_generation_wind_evolution_get
    )
    
    return demand_forecasts, wind_forecasts
    
    