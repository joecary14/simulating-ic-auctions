import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection
import price_forecaster.lear_forecast as lear_forecast

demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/FR D-2 Demand Forecast.xlsx'
price_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/All Prices.xlsx'
output_directory = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs'
output_filename = '/gb_fr_data.csv'
input_data_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/Forecast Inputs CSVs/gb_fr_data.csv'

async def main():
    lear_forecast.run_lear_forecast(
        input_data_filepath,
        output_data_filepath=output_directory + output_filename,
        calibration_window_days=364,
        start_test_date='2022-01-01',
        end_test_date='2022-01-31'
    )
     
asyncio.run(main())