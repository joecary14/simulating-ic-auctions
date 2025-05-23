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
    output_filepath: str
) -> None:
    forecast_data = await get_elexon_data_for_years(
        years
    )
    
    demand_forecast_data = read_in_demand_forecast_data(
        demand_data_filepath
    )
    
    price_spread_data = get_price_spread_data(
        price_data_filepath,
        country_id
    )
    
    # Merge the dataframes on the 'datetime' column
    merged_df = pd.merge(
        price_spread_data,
        demand_forecast_data,
        on='datetime',
        how='inner'
    )
    merged_df = pd.merge(
        merged_df,
        forecast_data,
        left_on='datetime',
        right_on='start_time',
        how='inner'
    )
    # Drop the duplicate 'start_time' column if present
    if 'start_time' in merged_df.columns:
        merged_df = merged_df.drop(columns=['start_time'])

    # Reorder columns so spread is first, then the rest
    spread_cols = [col for col in merged_df.columns if col.startswith('GB-')]
    other_cols = [col for col in merged_df.columns if col not in spread_cols + ['datetime']]
    merged_df = merged_df[['datetime'] + spread_cols + other_cols]

    # Save to output file
    merged_df.to_csv(output_filepath, index=False)
    
    return forecast_data
    
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
                date = pd.to_datetime(date)
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
                        'value': value
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