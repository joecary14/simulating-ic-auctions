import asyncio
import pandas as pd
import data_handler.elexon_interaction as elexon_interaction
import data_handler.datetime_functions as datetime_functions
import price_forecaster.data_collection_v1 as data_collection_v1
from elexonpy.api_client import ApiClient

async def get_data_for_fr_lear_forecast(
    fr_data_filepath: str,
    price_data_filepath: str,
    years: list[int],
    country_id: str,
    output_file_directory: str,
    output_filename: str
) -> None:
    elexon_forecast_data = await get_elexon_forecast_data_for_years(
        years
    )
    
    fr_nuclear_generation = read_in_fr_data(
        fr_data_filepath
    )
    
    price_spread_data = data_collection_v1.get_price_spread_data(
        price_data_filepath,
        country_id
    )
    
    merged_df = pd.merge(
        price_spread_data,
        fr_nuclear_generation,
        on='datetime',
        how='left'
    )
    merged_df = pd.merge(
        merged_df,
        elexon_forecast_data,
        left_on='datetime',
        right_on='start_time',
        how='left'
    )
    if 'start_time' in merged_df.columns:
        merged_df = merged_df.drop(columns=['start_time'])
        
    merged_df = merged_df.set_index('datetime').sort_index()
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    merged_df = merged_df[merged_df.index.year.isin(years)]
    
    for col in merged_df.columns:
        data_collection_v1.populate_missing_values_with_day_before_values(
            col,
            merged_df
        )
    merged_df = merged_df.reset_index()
        
    merged_df = data_collection_v1.rename_columns(
        merged_df,
        years
    )

    merged_df.to_csv(output_file_directory + output_filename, index=False)
    print(f"Data for FR LEAR forecast saved to {output_file_directory + output_filename}")
    

async def get_elexon_forecast_data_for_years(
    years: list[int]
) -> pd.DataFrame:
    api_client = ApiClient()
    tasks = [get_elexon_forecast_data_for_year(year, api_client) for year in years]
    results = await asyncio.gather(*tasks)
    combined_forecasts = pd.concat(results, axis=0)
    
    return combined_forecasts

async def get_elexon_forecast_data_for_year(
    year: int,
    api_client: ApiClient
) -> pd.DataFrame:
    start_date, end_date = datetime_functions.get_start_and_end_dates_from_year(year)
    settlement_dates = datetime_functions.generate_settlement_dates(
        start_date,
        end_date,
        format_date_time_as_string=True
    )
    settlement_start_times = datetime_functions.get_settlement_dates_and_times(
        start_date,
        end_date
    )
    
    wind_forecast_df = await elexon_interaction.get_hour_ahead_wind_forecast(
        settlement_start_times,
        api_client
    )
    
    atl_df = await elexon_interaction.get_actual_total_load(
        settlement_dates,
        api_client
    )
    
    combined_df = data_collection_v1.combine_values(
        atl_df,
        wind_forecast_df
    )
    
    return combined_df

def read_in_fr_data(
    fr_data_filepath: str
) -> pd.DataFrame:
    data = pd.read_excel(fr_data_filepath)
    data['UTC Datetime'] = pd.to_datetime(data['UTC Datetime'], utc=True)
    fr_nuclear_generation = data[['UTC Datetime', 'Nuclear Generation']].copy()
    fr_nuclear_generation.rename(columns={'UTC Datetime': 'datetime'}, inplace=True)
    
    return fr_nuclear_generation

def get_data_for_be_lear_forecast(
    be_demand_data_filepath: str,
    elexon_data_filepath: str,
    price_data_filepath: str,
    years: list[int],
    country_id: str,
    output_file_directory: str,
    output_filename: str
) -> None:
    elexon_forecast_data = pd.read_excel(elexon_data_filepath)
    elexon_forecast_data['datetime'] = pd.to_datetime(elexon_forecast_data['datetime'], utc=True)
    
    demand_forecast_data = read_in_be_demand_data(
        be_demand_data_filepath
    )
    
    price_spread_data = data_collection_v1.get_price_spread_data(
        price_data_filepath,
        country_id
    )
    
    merged_df = pd.merge(
        price_spread_data,
        demand_forecast_data,
        on='datetime',
        how='left'
    )
    merged_df = pd.merge(
        merged_df,
        elexon_forecast_data,
        left_on='datetime',
        right_on='datetime',
        how='left'
    )
    if 'start_time' in merged_df.columns:
        merged_df = merged_df.drop(columns=['start_time'])
        
    merged_df = merged_df.set_index('datetime').sort_index()
    merged_df = merged_df[merged_df.index.year.isin(years)]
    for col in merged_df.columns:
        data_collection_v1.populate_missing_values_with_day_before_values(
            col,
            merged_df
        )
    merged_df = merged_df.reset_index()
        
    merged_df = data_collection_v1.rename_columns(
        merged_df,
        years
    )

    merged_df.to_csv(output_file_directory + output_filename, index=False)
    print(f"Data for BE LEAR forecast saved to {output_file_directory + output_filename}")

def read_in_be_demand_data(
    demand_data_filepath: str
) -> pd.DataFrame:
    data = pd.read_excel(demand_data_filepath)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    demand_forecast = data[['datetime', 'Load']]
    demand_forecast = demand_forecast.set_index('datetime')
    demand_forecast = demand_forecast.resample('h').mean()
    demand_forecast = demand_forecast.reset_index()
    
    return demand_forecast

def get_data_for_dk1_lear_forecast(
    dk1_res_forecast_data: str,
    elexon_data_filepath: str,
    price_data_filepath: str,
    years: list[int],
    country_id: str,
    output_file_directory: str,
    output_filename: str
) -> None:
    elexon_forecast_data = pd.read_excel(elexon_data_filepath)
    elexon_forecast_data['datetime'] = pd.to_datetime(elexon_forecast_data['datetime'], utc=True)
    
    res_forecast_data = read_in_dk1_forecast_data(
        dk1_res_forecast_data
    )
    
    price_spread_data = data_collection_v1.get_price_spread_data(
        price_data_filepath,
        country_id
    )
    
    merged_df = pd.merge(
        price_spread_data,
        res_forecast_data,
        on='datetime',
        how='left'
    )
    merged_df = pd.merge(
        merged_df,
        elexon_forecast_data,
        left_on='datetime',
        right_on='datetime',
        how='left'
    )
    if 'start_time' in merged_df.columns:
        merged_df = merged_df.drop(columns=['start_time'])
        
    merged_df = merged_df.set_index('datetime').sort_index()
    merged_df = merged_df[merged_df.index.year.isin(years)]
    for col in merged_df.columns:
        data_collection_v1.populate_missing_values_with_day_before_values(
            col,
            merged_df
        )
    merged_df = merged_df.reset_index()
        
    merged_df = data_collection_v1.rename_columns(
        merged_df,
        years
    )

    merged_df.to_csv(output_file_directory + output_filename, index=False)
    print(f"Data for DK1 LEAR forecast saved to {output_file_directory + output_filename}")

def read_in_dk1_forecast_data(
    dk1_res_forecast_data: str
) -> pd.DataFrame:
    data = pd.read_excel(dk1_res_forecast_data)
    data['datetime'] = pd.to_datetime(data['UTC Datetime'], utc=True)
    data.drop(columns=['UTC Datetime'], inplace=True)
    
    return data