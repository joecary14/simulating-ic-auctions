import polars as pl
import pandas as pd
import openpyxl

def read_in_excel_data(
    filepath: str
) -> dict[str, pl.DataFrame]:
    workbook = openpyxl.load_workbook(filepath, read_only=True)
    sheet_names = workbook.sheetnames
    workbook.close()
    
    polars_data = {}
    for sheet_name in sheet_names:
        df = pl.read_excel(
            filepath, 
            sheet_name=sheet_name
        )
        
        for col in df.columns:
            df = df.with_columns([
                pl.when(pl.col(col).cast(pl.Utf8).is_in(["-", "NA", "N/A"]))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            ])
        
        polars_data[sheet_name] = df
    
    return polars_data

def write_data_to_excel(
    dataframes_dict: dict[str, pl.DataFrame], 
    output_filepath: str
) -> None:
    with pd.ExcelWriter(output_filepath, engine='xlsxwriter') as writer:
        for sheet_name, dataframe in dataframes_dict.items():
            dataframe.to_pandas().to_excel(writer, sheet_name=sheet_name, index=False)