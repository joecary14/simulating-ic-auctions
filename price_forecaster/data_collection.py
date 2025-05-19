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
        start_date=start_date,
        end_date=end_date,
        convert_datetime_to_string=True,
    )
    api_client = ApiClient()
    
    demand_forecast_df = await elexon_interaction.get_latest_actionable_demand_forecast_for_date_range(
        settlement_dates_with_periods_per_day,
        api_client
    )

    wind_forecast_df = await elexon_interaction.get_latest_wind_forecast(
        settlement_dates_with_periods_per_day,
        api_client
    )
    
    combined_forecasts = combine_values(
        demand_forecast_df,
        wind_forecast_df
    )
    
    return combined_forecasts
    
def combine_values(
    demand_forecast_df: pd.DataFrame,
    wind_forecast_df: pd.DataFrame,
):
    # Fill NA values with the value from the row above
    demand_forecast_df = demand_forecast_df.fillna(method='ffill')

    # Ensure the index is a datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(demand_forecast_df.index):
        demand_forecast_df.index = pd.to_datetime(demand_forecast_df.index)

    # Resample to hourly by averaging the value at the hour and the next half hour
    hourly_index = demand_forecast_df.index[demand_forecast_df.index.minute == 0]
    hourly_values = []
    for dt in hourly_index:
        try:
            val1 = demand_forecast_df.loc[dt].values
            val2 = demand_forecast_df.loc[dt + pd.Timedelta(minutes=30)].values
            avg = (val1 + val2) / 2
            hourly_values.append(avg)
        except KeyError:
            continue  # skip if half-hour data is missing

    hourly_df = pd.DataFrame(
        data=[v.flatten() for v in hourly_values],
        index=hourly_index[:len(hourly_values)],
        columns=demand_forecast_df.columns
    )

    combined_df = pd.merge(
        hourly_df,
        wind_forecast_df,
        on='start_time',
        how='inner',
        suffixes=('_demand', '_wind')
    )
    
    return combined_df