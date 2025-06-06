import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection
import price_forecaster.lear_forecast as lear_forecast

demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/FR D-2 Demand Forecast.xlsx'
be_demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/BE D-7 Demand Forecast.xlsx'
price_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/All Prices.xlsx'
output_directory = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Output Data/LEAR/v1'
output_filename = '/gb_dk1_2023_2024_data.csv'
input_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/gb_dk1_data.csv'
dk1_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/DK D-1 RES Forecasts.xlsx'
elexon_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Elexon TSDF & Wind Forecast 2021-2024.xlsx'
nl_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/NL D-1 RES Forecasts.xlsx'
country_code = 'BE'

async def main():
    # data_collection.get_data_for_dk1_lear_forecast(
    #     dk1_data_filepath,
    #     elexon_data_filepath,
    #     price_filepath,
    #     [2023, 2024],
    #     country_code,
    #     output_directory,
    #     output_filename
    # )
    
    lear_forecast.run_lear_forecast(
        input_data_filepath,
        364,
        '2024-06-01',
        '2024-07-01',
        country_code,
        output_directory
    )
     
asyncio.run(main())