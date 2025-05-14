import pytz

import pandas as pd

from datetime import datetime, timedelta, timezone

gb_timezone = pytz.timezone('Europe/London')

def get_start_and_end_dates_from_year(year, convert_datetime_to_string=False):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    return start_date, end_date

def get_settlement_dates_and_settlement_periods_per_day(start_date_str, end_date_str, convert_datetime_to_string):
    full_date_list = generate_settlement_dates(start_date_str, end_date_str)
    dates_with_settlement_periods_per_day = get_settlement_periods_for_each_day_in_date_range(full_date_list)
    if convert_datetime_to_string:
        dates_with_settlement_periods_per_day = {key.strftime('%Y-%m-%d'): value 
                                                 for key, value in dates_with_settlement_periods_per_day.items()}
    return dates_with_settlement_periods_per_day

def get_list_of_settlement_dates_and_periods(settlement_dates_with_periods_per_day : dict):
    settlement_dates_and_periods = []
    for settlement_date, settlement_periods_in_day in settlement_dates_with_periods_per_day.items():
        for settlement_period in range(1, settlement_periods_in_day + 1):
            settlement_dates_and_periods.append(f"{settlement_date}-{settlement_period}")
            
    return settlement_dates_and_periods

def generate_settlement_dates(start_date_str, end_date_str, format_date_time_as_string = False):
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD") from e

    date_list = [(start_date + timedelta(days=i)) for i in range((end_date - start_date).days + 1)]
    
    if format_date_time_as_string:
        date_list = [date.strftime('%Y-%m-%d') for date in date_list]
    
    return date_list

def add_settlement_date_to_end_of_list(settlement_dates_inclusive):
    last_settlement_date = settlement_dates_inclusive[-1]
    if type(last_settlement_date) == str:
        last_settlement_date = datetime.strptime(last_settlement_date, '%Y-%m-%d')
    additional_date = last_settlement_date + timedelta(days = 1)
    if type(settlement_dates_inclusive[0]) == str:
        additional_date = additional_date.strftime('%Y-%m-%d')
    return settlement_dates_inclusive + [additional_date]

def get_settlement_periods_for_each_day_in_date_range(settlement_dates_inclusive):
    settlement_periods_per_day = {}
    settlement_dates_for_calculation = settlement_dates_inclusive + [
        settlement_dates_inclusive[-1] + timedelta(days = 1)]

    for i in range(len(settlement_dates_for_calculation) - 1):
        current_date = settlement_dates_for_calculation[i]
        next_date = settlement_dates_for_calculation[i+1]
        offset_now = gb_timezone.utcoffset(current_date)
        offset_next = gb_timezone.utcoffset(next_date)
        settlement_periods_in_day = 48

        if offset_now != offset_next:
            settlement_periods_in_day = (46 if offset_next > offset_now 
                else 50)
            
        settlement_periods_per_day[current_date] = settlement_periods_in_day
    
    return settlement_periods_per_day
        
def translate_settlement_dates_and_periods_to_timestamps(settlement_dates_and_periods):
    translations = {}
    for settlement_date, settlement_periods_per_day in settlement_dates_and_periods.items():
        for settlement_period in range(1, settlement_periods_per_day + 1):
            settlement_date_and_period = f'{settlement_date}-{settlement_period}'
            timestamp = get_timestamp_from_settlement_date_and_period(settlement_date_and_period)
            translations[settlement_date_and_period] = timestamp
    
    return translations

def add_settlement_time_to_end_of_list(datetimes):
        if not datetimes:
            return datetimes
        last_dt_str = datetimes[-1]
        last_dt = datetime.fromisoformat(last_dt_str)
        last_dt_plus_half_hour = last_dt + timedelta(minutes=30)
        return datetimes + [last_dt_plus_half_hour.isoformat()]  

def get_timestamp_from_settlement_date_and_period(settlement_date_and_period):
            date_str, period_str = settlement_date_and_period.rsplit('-', 1)
            settlement_date = datetime.strptime(date_str, '%Y-%m-%d')
            settlement_period = int(period_str)
            base_time = gb_timezone.localize(settlement_date)
            offset_minutes = (settlement_period - 1) * 30
            timestamp = base_time + timedelta(minutes=offset_minutes)
            timestamp_utc = timestamp.astimezone(pytz.utc).isoformat()

            return timestamp_utc

def get_time_as_string_from_np_datetime(datetime_obj):
    pd_timestamp = pd.Timestamp(datetime_obj)
    time_str = pd_timestamp.strftime('%H:%M:%S')
    return time_str

def get_time_as_string_from_dt_datetime(dt_datetime):
    time = dt_datetime.time()
    time_str = time.strftime('%H:%M:%S')
    return time_str
        
def get_previous_settlement_date_and_period(settlement_date_and_period, settlement_dates_with_periods_per_day):
    settlement_dates_and_periods_list = get_list_of_settlement_dates_and_periods(settlement_dates_with_periods_per_day)
    settlement_date_index = settlement_dates_and_periods_list.index(settlement_date_and_period)
    if settlement_date_index == 0:
        raise ValueError(f"No previous settlement date and period available for {settlement_date_and_period}")
    previous_settlement_date_and_period = settlement_dates_and_periods_list[settlement_date_index - 1]
    return previous_settlement_date_and_period

def convert_utc_datetime_to_settlement_date_and_period(utc_datetime):
    
    local_datetime = utc_datetime.astimezone(gb_timezone)
    settlement_date = local_datetime.date()
    settlement_time = local_datetime.time()

    settlement_period = (settlement_time.hour * 2) + (settlement_time.minute // 30) + 1
    
    start_day_utc_offset = gb_timezone.utcoffset(local_datetime.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0))
    end_day_utc_offset = gb_timezone.utcoffset(local_datetime.replace(tzinfo=None, hour=23, minute=0, second=0, microsecond=0))
    
    #Handles clocks going forwards an hour
    if start_day_utc_offset < end_day_utc_offset:
        if local_datetime.utcoffset() > timedelta(0):
            settlement_time = utc_datetime.time()
            settlement_period = (settlement_time.hour * 2) + (settlement_time.minute // 30) + 1
    
    #Handles clocks going back an hour
    if start_day_utc_offset > end_day_utc_offset:
        local_datetime = utc_datetime.astimezone(gb_timezone)
        if local_datetime.utcoffset() == timedelta(0):
            settlement_period = (settlement_time.hour * 2) + (settlement_time.minute // 30) + 3
    
    settlement_date_str = settlement_date.strftime('%Y-%m-%d')
    
    return f"{settlement_date_str}-{settlement_period}"