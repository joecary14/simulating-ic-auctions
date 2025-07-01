import pandas as pd
import price_forecaster.data_collection_v1 as data_collection_v1

def add_monthly_variables(
    original_data_filepath: str,
    output_file_directory: str,
    output_filename: str,
    years: list[int]
) -> None:
    """
    Adds monthly dummy variables to the original data and saves the modified DataFrame to a CSV file.
    
    Parameters:
        original_data_filepath (str): Path to the original data file.
        output_file_directory (str): Directory where the modified data will be saved.
        output_filename (str): Name of the output CSV file.
    """
    raw_data = pd.read_csv(original_data_filepath)
    # Ensure 'datetime' is in UTC
    raw_data['datetime'] = pd.to_datetime(raw_data['datetime'], utc=True)
    raw_data['month'] = raw_data['datetime'].dt.month
    for month in range(1, 13):
        month_name = pd.to_datetime(f'2023-{month:02d}-01').strftime('%B')
        raw_data[f'month_{month_name}'] = (raw_data['month'] == month).astype(int)
    raw_data.drop(columns=['month'], inplace=True)
    renamed_data = data_collection_v1.rename_columns(
        raw_data,
        years
    )
    renamed_data.to_csv(f"{output_file_directory}/{output_filename}", index=False)
    print(f"Data with monthly variables saved to {output_file_directory}/{output_filename}")