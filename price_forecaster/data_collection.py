import pandas as pd
from elexonpy.api_client import ApiClient
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
    demand_forecast_df = demand_forecast_df.ffill()
    demand_forecast_df = demand_forecast_df.set_index('start_time')
    demand_forecast_df.index = pd.to_datetime(demand_forecast_df.index)

    hourly_index = demand_forecast_df.index[demand_forecast_df.index.minute == 0]
    hourly_values = []
    for dt in hourly_index:
        try:
            val1 = demand_forecast_df.loc[dt].values
            val2 = demand_forecast_df.loc[dt + pd.Timedelta(minutes=30)].values
            avg = (val1 + val2) / 2
            hourly_values.append(avg)
        except KeyError:
            continue

    hourly_df = pd.DataFrame(
        data=[v.flatten() for v in hourly_values],
        index=hourly_index[:len(hourly_values)],
        columns=demand_forecast_df.columns
    )
    
    hourly_df = hourly_df.reset_index().rename(columns={'index': 'start_time'})
    hourly_df['start_time'] = pd.to_datetime(hourly_df['start_time']).dt.tz_localize('UTC')
    wind_forecast_df['start_time'] = pd.to_datetime(wind_forecast_df['start_time'])

    combined_df = pd.merge(
        hourly_df,
        wind_forecast_df,
        on='start_time',
        how='inner',
        suffixes=('_demand', '_wind')
    )
    combined_df = combined_df.rename(columns={'generation': 'wind_generation_forecast'})
    
    return combined_df