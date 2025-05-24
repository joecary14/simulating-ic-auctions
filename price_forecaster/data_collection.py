import asyncio
import pandas as pd
from elexonpy.api_client import ApiClient
import data_handler.datetime_functions as datetime_functions
import data_handler.elexon_interaction as elexon_interaction

async def get_data_for_lear_forecast(
    demand_data_filepath: str,
    price_data_filepath: str,
    years: list[int],
    country_id: str,
    output_file_directory: str,
    output_filename: str
) -> None:
    elexon_forecast_data = await get_elexon_data_for_years(
        years
    )
    
    demand_forecast_data = read_in_demand_forecast_data(
        demand_data_filepath
    )
    
    price_spread_data = get_price_spread_data(
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
        right_on='start_time',
        how='left'
    )
    if 'start_time' in merged_df.columns:
        merged_df = merged_df.drop(columns=['start_time'])
        
    merged_df = merged_df.set_index('datetime').sort_index()
    merged_df = merged_df[merged_df.index.year.isin(years)]
    for col in merged_df.columns:
        populate_missing_values_with_day_before_values(
            col,
            merged_df
        )
    merged_df = merged_df.reset_index()
        
    merged_df = merged_df[merged_df['datetime'].dt.year.isin(years)]
    spread_cols = [col for col in merged_df.columns if col.startswith('GB-')]
    other_cols = [col for col in merged_df.columns if col not in spread_cols + ['datetime']]
    merged_df = merged_df[['datetime'] + spread_cols + other_cols]
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

    merged_df.to_csv(output_file_directory + output_filename, index=False)
    
def get_price_spread_data(
    read_in_filepath: str,
    country_id: str
) -> pd.DataFrame:
    df = pd.read_excel(read_in_filepath)
    datetime_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > len(df) // 2:
                datetime_col = col
                break
        except Exception:
            continue
    if datetime_col is None:
        raise ValueError("Could not infer datetime column from values.")
    gb_col = [col for col in df.columns if 'GB' in col][0]
    other_cols = [col for col in df.columns if col not in [datetime_col, gb_col]]

    spread_data = {
        'datetime': df[datetime_col]
    }
    for col in other_cols:
        # Extract country code as the two uppercase letters not in brackets
        country_code = ''.join([c for c in col if c.isupper() and c != 'G' and c != 'B'])[:2]

        spread_col_name = f"GB-{country_code}"
        spread_data[spread_col_name] = df[gb_col] - df[col]

    spread_df = pd.DataFrame(spread_data)
    
    spread_cols = [col for col in spread_df.columns if country_id in col]
    spread_df = spread_df[['datetime'] + spread_cols]
    spread_df['datetime'] = pd.to_datetime(spread_df['datetime'], errors='coerce').dt.tz_localize('UTC')
    
    return spread_df

def read_in_demand_forecast_data(
    read_in_filepath: str
) -> pd.DataFrame:
    df = pd.read_excel(read_in_filepath, index_col=0)
    result = []
    for date, row in df.iterrows():
        if not isinstance(date, pd.Timestamp):
            try:
                date = pd.to_datetime(date, dayfirst=True)
            except:
                print(f"Skipping invalid date: {date}")
                continue
            
        for hour, value in row.items():
            if pd.notna(value):
                try:
                    if not isinstance(hour, int):
                        hour = int(hour)
                    
                    dt = date.replace(hour=hour, minute=0, second=0)
                    result.append({
                        'datetime': dt,
                        'fr_demand_forecast': value
                    })
                except:
                    print(f"Skipping invalid hour: {hour} for date {date}")
                    continue
    
    result_df = pd.DataFrame(result)
    
    if not result_df.empty:
        result_df['datetime'] = pd.to_datetime(result_df['datetime'])
        result_df['datetime'] = result_df['datetime'] - pd.Timedelta(hours=1)
        
        result_df['datetime'] = result_df['datetime'].dt.tz_localize('UTC')
        result_df = result_df.sort_values('datetime').reset_index(drop=True)
    
    return result_df

async def get_elexon_data_for_years(
    years: list[int]
) -> None:
    tasks = [get_elexon_lear_data_for_year(year) for year in years]
    results = await asyncio.gather(*tasks)
    combined_forecasts = pd.concat(results, axis=0)
    
    return combined_forecasts

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
    demand_forecast_df['start_time'] = pd.to_datetime(demand_forecast_df['start_time'])
    demand_forecast_df['hour'] = demand_forecast_df['start_time'].dt.floor('h')
    hourly_demand_forecast_data = demand_forecast_df.groupby('hour')['tsdf'].mean().reset_index()
    hourly_demand_forecast_data.rename(columns={'hour': 'start_time'}, inplace=True)
    hourly_demand_forecast_data['start_time'] = pd.to_datetime(hourly_demand_forecast_data['start_time']).dt.tz_localize('UTC')
    wind_forecast_df['start_time'] = pd.to_datetime(wind_forecast_df['start_time'])

    combined_df = pd.merge(
        hourly_demand_forecast_data,
        wind_forecast_df,
        on='start_time',
        how='left',
        suffixes=('_demand', '_wind')
    )
    combined_df = combined_df.rename(columns={'generation': 'wind_generation_forecast'})
    
    return combined_df

def populate_missing_values_with_day_before_values(
    column_name : str,
    df : pd.DataFrame
) -> None:
    if df[column_name].isna().any():
        day_ago_values = df[column_name].shift(24, freq='h')
        df[column_name] = df[column_name].fillna(day_ago_values)
        if df[column_name].isna().any():
            populate_missing_values_with_day_before_values(
                column_name,
                df
            )
        else:
            return