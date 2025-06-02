import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection
import price_forecaster.lear_forecast as lear_forecast

demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/FR D-2 Demand Forecast.xlsx'
be_demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/BE D-7 Demand Forecast.xlsx'
price_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/All Prices.xlsx'
output_directory = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/'
output_filename = '/gb_nl_data.csv'
input_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/gb_nl_data.csv'
dk1_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/DK D-1 RES Forecasts.xlsx'
elexon_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Elexon TSDF & Wind Forecast 2021-2022.xlsx'
nl_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/NL D-1 RES Forecasts.xlsx'

async def main():
    lear_forecast.run_lear_forecast(
        input_data_filepath,
        output_directory + output_filename,
        364,
        '2022-01-01',
        '2022-06-01'
    )
     
asyncio.run(main())