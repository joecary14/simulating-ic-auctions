import pandas as pd
import epftoolbox.models._lear as epf_lear

def run_lear_forecast(
    input_data_filepath: str,
    output_data_filepath: str,
    calibration_window_days: int,
    start_test_date: str | pd.Timestamp,
    end_test_date: str | pd.Timestamp
):
    epf_lear.evaluate_lear_in_test_dataset(
        input_data_filepath,
        output_data_filepath,
        calibration_window=calibration_window_days,
        begin_test_date= start_test_date,
        end_test_date= end_test_date
    )