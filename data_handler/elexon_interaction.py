import math
import asyncio
import constants as ct
import pandas as pd
from datetime import datetime, timedelta, timezone

async def get_latest_actionable_forecasts_for_date_range(
    settlement_dates_with_periods_per_day : dict[str, int],
    api_function,
):
    tasks = [get_latest_actionable_forecast_one_day(
        api_function,
        settlement_date,
        settlement_periods_in_day
    ) for settlement_date, settlement_periods_in_day in settlement_dates_with_periods_per_day.items()]
    
    forecast_data = await asyncio.gather(*tasks)
    forecast_df = pd.concat(forecast_data, axis=0)
    
    return forecast_df  

async def get_latest_actionable_forecast_one_day(
    api_function,
    settlement_date: str,
    settlement_periods_in_day : int,
) -> pd.DataFrame:
    forecast_data = await api_function(
        settlement_date,
        [settlement_period for settlement_period in range(1, settlement_periods_in_day + 1)],
        format='dataframe'
    )
    forecasts = []
    cutoff_publish_time = infer_cutoff_time(settlement_date)
    for settlement_period, forecast in forecast_data.groupby(ct.ColumnNames.SETTLEMENT_PERIOD.value):
        if forecast.empty:
            data = (math.nan, math.nan)
        else:
            forecast['publish_time'] = pd.to_datetime(forecast['publish_time'])
            cutoff_time = pd.to_datetime(cutoff_publish_time)
            valid_forecasts = forecast[forecast['publish_time'] <= cutoff_time]
            closest_forecast = valid_forecasts.loc[valid_forecasts['publish_time'].idxmax()]
            start_time = pd.to_datetime(closest_forecast['start_time'])
            data = (start_time, closest_forecast)
        forecasts.append(data)
    
    forecast_df = pd.DataFrame(forecasts, columns=['start_time', 'forecast'])
    forecast_df['start_time'] = forecast_df['start_time'].fillna(
        method='ffill'
    ) + pd.to_timedelta(30, unit='m')
    forecast_df['forecast'] = forecast_df['forecast'].interpolate(method='linear')
    
    return forecast_df


def infer_cutoff_time(settlement_date: str) -> str:
    settlement_date_obj = datetime.strptime(settlement_date, "%Y-%m-%d")
    cutoff_time = settlement_date_obj - timedelta(days=1)
    cutoff_time = cutoff_time.replace(hour=7, minute=45, second=0, microsecond=0, tzinfo=timezone.utc) #Default cutoff time so that forecast is in ahead of BritNed auction
    return cutoff_time.isoformat()
