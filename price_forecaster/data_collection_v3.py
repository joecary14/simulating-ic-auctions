import pandas as pd
import price_forecaster.data_collection_v1 as data_collection_v1
import price_forecaster.data_collection_v2 as data_collection_v2

def get_data_for_be_lear_forecast(
    fr_data_filepath: str,
    elexon_data_filepath: str,
    price_data_filepath: str,
    years: list[int],
    country_id: str,
    output_file_directory: str,
    output_filename: str
) -> None:
    elexon_forecast_data = pd.read_excel(elexon_data_filepath)
    elexon_forecast_data['datetime'] = pd.to_datetime(elexon_forecast_data['datetime'], utc=True)
    
    fr_nuclear_gen_data = data_collection_v2.read_in_fr_data(
        fr_data_filepath
    )
    
    price_spread_data = data_collection_v1.get_price_spread_data(
        price_data_filepath,
        country_id
    )
    
    merged_df = pd.merge(
        price_spread_data,
        fr_nuclear_gen_data,
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