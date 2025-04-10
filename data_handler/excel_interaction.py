import polars as pl
import pandas as pd

def read_in_excel_data(filepath):
    excel_data = pd.read_excel(filepath, sheet_name=None)
    
    polars_data = {sheet_name: pl.DataFrame(sheet_data) for sheet_name, sheet_data in excel_data.items()}

    return polars_data