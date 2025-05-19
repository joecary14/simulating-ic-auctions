import asyncio
import model.engine as engine
import price_forecaster.data_collection as data_collection

async def main():
    await data_collection.get_elexon_lear_data_for_year(2023)
     
asyncio.run(main())