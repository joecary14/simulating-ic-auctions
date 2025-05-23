import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection

demand_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/FR D-2 Demand Forecast.xlsx'
price_filepath = '/Users/josephcary/Library/CloudStorage/OneDrive-Nexus365/First Year/Papers/Interconnection/Forecasting/Input Data/All Prices.xlsx'

async def main():
    await data_collection.get_data_for_lear_forecast(
        demand_filepath,
        price_filepath,
        years=[2021],
        country_id='FR',
        output_filepath='data/lear_forecast.xlsx'
    )
     
asyncio.run(main())