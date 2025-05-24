import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import epftoolbox.models._lear as epf_lear
import matplotlib.pyplot as plt

def run_lear_forecast(
    input_data_filepath: str,
    output_data_filepath: str,
    calibration_window_days: int,
    start_test_date: str | pd.Timestamp,
    end_test_date: str | pd.Timestamp
):
    training_data = pd.read_csv(
        input_data_filepath,
        parse_dates=['datetime']
    )
    training_data.set_index('datetime', inplace=True)
    lear_model = epf_lear.LEAR(calibration_window_days)
    if isinstance(start_test_date, str):
        start_test_date = pd.to_datetime(start_test_date, utc=True)
    if isinstance(end_test_date, str):
        end_test_date = pd.to_datetime(end_test_date, utc=True)
        
    combined_results_df = run_lear_forecast_by_day(
        lear_model,
        training_data,
        calibration_window_days,
        start_test_date,
        end_test_date
    )

    lower_actual = combined_results_df['actual'].quantile(0.05)
    upper_actual = combined_results_df['actual'].quantile(0.95)
    lower_pred = combined_results_df['prediction'].quantile(0.05)
    upper_pred = combined_results_df['prediction'].quantile(0.95)

    filtered_df = combined_results_df[
        (combined_results_df['actual'] >= lower_actual) &
        (combined_results_df['actual'] <= upper_actual) &
        (combined_results_df['prediction'] >= lower_pred) &
        (combined_results_df['prediction'] <= upper_pred)
    ]
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_df['actual'], filtered_df['prediction'], alpha=0.5)
    plt.xlabel('Filtered Actual')
    plt.ylabel('Filtered Prediction')
    plt.title('5-95th percentile Predicted vs Actual Scatter Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_data_filepath.replace('.csv', '_scatter.png'))
    plt.close()
    
    r2 = r2_score(combined_results_df['actual'], combined_results_df['prediction'])

    sign_agreement = np.mean(
        np.sign(combined_results_df['actual']) == np.sign(combined_results_df['prediction'])
    ) * 100

    print(f"R^2 score: {r2:.4f}")
    print(f"Sign agreement: {sign_agreement:.2f}%")
    
    return combined_results_df

def run_lear_forecast_by_day(
    lear_model : epf_lear.LEAR,
    training_data: pd.DataFrame,
    calibration_window_days: int,
    start_test_date: str | pd.Timestamp,
    end_test_date: str | pd.Timestamp
) -> pd.DataFrame:
    results = []
    date_range = pd.date_range(start=start_test_date, end=end_test_date, freq='D', tz='UTC')
    for test_date in date_range:    
        predictions, actuals = lear_model.recalibrate_and_forecast_next_day(
            training_data,
            calibration_window_days,
            test_date
        )
        datetimes = pd.date_range(start=test_date, periods=24, freq='h', tz='UTC')
        result_df = pd.DataFrame({
            'datetime': datetimes,
            'prediction': predictions[0],
            'actual': actuals[0]
        })
        results.append(result_df)
        print(f"Forecast for {test_date.strftime('%Y-%m-%d')} completed.")
    
    combined_results_df = pd.concat(results, ignore_index=True)
    combined_results_df.set_index('datetime', inplace=True)
    
    return combined_results_df
