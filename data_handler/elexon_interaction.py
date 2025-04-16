import constants as ct
import pandas as pd
import elexonpy.api.demand_forecast_api as demand_forecast_api
import elexonpy.api_client as api_client

async def get_latest_actionable_demand_forecast(
    api_client: api_client.ApiClient,
    settlement_date: str,
    settlement_periods_in_day : int,
    cutoff_publish_time: str
) -> int:
    forecast_api = demand_forecast_api.DemandForecastApi(api_client)
    forecast_data = await forecast_api.forecast_demand_day_ahead_evolution_get(
        settlement_date,
        [settlement_period for settlement_period in range(1, settlement_periods_in_day + 1)],
        format='dataframe'
    )
    
    start_time_to_forecast = {}
    for settlement_period, forecast in forecast_data.groupby(ct.ColumnNames.SETTLEMENT_PERIOD.value):
        forecast['publish_time'] = pd.to_datetime(forecast['publish_time'])
        cutoff_time = pd.to_datetime(cutoff_publish_time)
        valid_forecasts = forecast[forecast['publish_time'] <= cutoff_time]
        if valid_forecasts.empty:
            #TODO handle missing data here
            pass
        if not valid_forecasts.empty:
            closest_forecast = valid_forecasts.loc[valid_forecasts['publish_time'].idxmax()]
            start_time = closest_forecast['start_time']
            start_time_to_forecast[start_time] = closest_forecast

