import math
import asyncio
import constants as ct
import pandas as pd
from datetime import datetime, timedelta, timezone
from elexonpy.api_client import ApiClient
from elexonpy.api.demand_forecast_api import DemandForecastApi
from elexonpy.api.demand_api import DemandApi
from elexonpy.api.generation_forecast_api import GenerationForecastApi
from elexonpy.api.generation_api import GenerationApi

async def get_latest_actionable_demand_forecast_for_date_range(
    settlement_dates_with_periods_per_day : dict[str, int],
    api_client : ApiClient,
) -> pd.DataFrame:
    demand_forecast_api = DemandForecastApi(api_client)
    demand_api = DemandApi(api_client)
    tasks = {}
    for settlement_date, settlement_periods_in_day in settlement_dates_with_periods_per_day.items():
        tasks[settlement_date] = demand_forecast_api.forecast_demand_day_ahead_evolution_get(
            settlement_date,
            [settlement_period for settlement_period in range(1, settlement_periods_in_day + 1)],
            format='dataframe',
            async_req=True
        )
            
    gather_tasks = []
    for key, task in tasks.items():
        gather_tasks.append(asyncio.to_thread(task.get))
    results = await asyncio.gather(*gather_tasks)
    processed_results = []
    missing_points = 0
    i = 0
    for key in tasks.keys():
        settlement_date = key
        cutoff_time = infer_cutoff_time(settlement_date)
        result_df = results[i]
        settlement_periods_in_day_list = [settlement_period for settlement_period in range(1, settlement_dates_with_periods_per_day[settlement_date] + 1)]
        for settlement_period in settlement_periods_in_day_list:
            if result_df.empty:
                start_time, tsdf = await get_missing_demand_data_point(
                    demand_api,
                    settlement_date,
                    settlement_period
                )
                missing_points += 1
            else:
                result_one_sp_df = result_df[result_df['settlement_period'] == settlement_period]
                if result_one_sp_df.empty:
                    start_time, tsdf = await get_missing_demand_data_point(
                        demand_api,
                        settlement_date,
                        settlement_period
                    )
                    missing_points += 1
                
                else:
                    start_time = pd.to_datetime(result_one_sp_df['start_time'].values[0])
                    valid_forecasts = result_one_sp_df[result_one_sp_df['publish_time'] <= cutoff_time]
                    if valid_forecasts.empty:
                        start_time, tsdf = await get_missing_demand_data_point(
                            demand_api,
                            settlement_date,
                            settlement_period
                        )
                        missing_points += 1
                        
                    else:
                        closest_forecast = valid_forecasts.loc[valid_forecasts['publish_time'].idxmax()] #Should be a pandas series
                        tsdf = closest_forecast['transmission_system_demand']
            
            
            data = (start_time, tsdf)
            processed_results.append(data)
                
        i += 1
    
    forecast_df = pd.DataFrame(processed_results, columns=['start_time', 'tsdf'])
    
    return forecast_df

async def get_missing_demand_data_point(
    demand_api : DemandApi,
    settlement_date : str,
    settlement_period : int
):
    outturn_df = await asyncio.to_thread(
        demand_api.demand_outturn_get(
            settlement_date_from = settlement_date,
            settlement_date_to = settlement_date,
            settlement_period = [settlement_period],
            format='dataframe',
            async_req=True
        ).get
    )
    
    if outturn_df.empty:
        return math.nan, math.nan
    
    else:
        start_time = pd.to_datetime(outturn_df['start_time'].values[0])
        transmission_system_demand = outturn_df['initial_transmission_system_demand_outturn'].values[0]
        
    return start_time, transmission_system_demand

async def get_latest_wind_forecast(
    settlement_dates_with_periods_per_day : dict[str, int],
    api_client : ApiClient
) -> pd.DataFrame:
    cutoff_times = [infer_cutoff_time(settlement_date) for settlement_date in settlement_dates_with_periods_per_day.keys()]
    generation_forecast_api = GenerationForecastApi(api_client)
    generation_api = GenerationApi(api_client)
    tasks = [
        generation_forecast_api.forecast_generation_wind_history_get(
            cutoff_time,
            format='dataframe',
            async_req=True
        )
        for cutoff_time in cutoff_times
    ]
    
    results = await asyncio.gather(*[asyncio.to_thread(task.get) for task in tasks])
    wind_forecasts = []
    for result_df, settlement_date in zip(results, settlement_dates_with_periods_per_day.keys()):
        start_date_time = settlement_date + 'T00:00Z'
        end_date_time = settlement_date + 'T23:30Z'
        if result_df.empty:
            wind_data_df = await get_missing_wind_data_for_day(
                start_date_time,
                end_date_time,
                generation_api
            )
        
        else:
            result_df['start_time'] = pd.to_datetime(result_df['start_time'])
            forecast_df = result_df[result_df['start_time'] <= end_date_time]
            forecast_df = forecast_df[forecast_df['start_time'] >= start_date_time]
            if forecast_df.empty:
                wind_data_df = await get_missing_wind_data_for_day(
                    start_date_time,
                    end_date_time,
                    generation_api
                )
            else:
                wind_data_df = forecast_df[['start_time', 'generation']]
        
        wind_forecasts.append(wind_data_df)
    
    wind_forecasts_df = pd.concat(wind_forecasts, ignore_index=True)
    
    return wind_forecasts_df    

async def get_missing_wind_data_for_day(
    start_date_time : str,
    end_date_time : str,
    generation_api : GenerationApi
):
    actual_generation_df = await asyncio.to_thread(
        generation_api.generation_actual_per_type_wind_and_solar_get(
            _from = start_date_time,
            to = end_date_time,
            format='dataframe',
            async_req=True
        ).get
    )
    
    if actual_generation_df.empty:
        return pd.DataFrame(columns=['start_time', 'generation'])

    wind_generation_df = actual_generation_df[actual_generation_df['business_type'] == 'Wind generation']
    grouped = wind_generation_df.groupby('start_time')['quantity'].sum().reset_index()
    grouped.rename(columns={'quantity': 'wind_generation'}, inplace=True)
    grouped['start_time'] = pd.to_datetime(grouped['start_time'])
    grouped['hour'] = grouped['start_time'].dt.floor('h')
    hourly = grouped.groupby('hour')['wind_generation'].mean().reset_index()
    hourly.rename(columns={'hour': 'start_time', 'wind_generation' : 'generation'}, inplace=True)

    grouped = hourly
    
    return grouped

def infer_cutoff_time(settlement_date: str) -> str:
    settlement_date_obj = datetime.strptime(settlement_date, "%Y-%m-%d")
    cutoff_time = settlement_date_obj - timedelta(days=1)
    cutoff_time = cutoff_time.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=timezone.utc) #Default cutoff time so that forecast is in ahead of BritNed auction
    cutoff_time_iso_format = cutoff_time.isoformat()
    return cutoff_time_iso_format
