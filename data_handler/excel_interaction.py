import polars as pl
import pandas as pd

def read_in_excel_data(
    filepath: str
) -> dict[str, pl.DataFrame]:
    excel_data = pd.read_excel(filepath, sheet_name=None)
    
    polars_data = {sheet_name: pl.DataFrame(sheet_data) for sheet_name, sheet_data in excel_data.items()}

    return polars_data

def write_data_to_excel(
    dataframes_dict: dict[str, pl.DataFrame], 
    output_filepath: str
) -> None:
    with pd.ExcelWriter(output_filepath, engine='xlsxwriter') as writer:
        for sheet_name, dataframe in dataframes_dict.items():
            dataframe.to_pandas().to_excel(writer, sheet_name=sheet_name, index=False)