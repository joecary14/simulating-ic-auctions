import pandas as pd
import epftoolbox.models._dnn as epf_dnn
import epftoolbox.models._dnn_hyperopt as epf_dnn_hyperopt

def optimise_hyperparameters(
    input_data_filepath: str,
    output_hyperparameters_folder_path: str,
    start_test_date: str | pd.Timestamp,
    end_test_date: str | pd.Timestamp
):
    epf_dnn_hyperopt.hyperparameter_optimizer(
        input_data_filepath,
        output_hyperparameters_folder_path,
        dataset=None,
        new_hyperopt= 1,
        shuffle_train=0,
        begin_test_date= start_test_date,
        end_test_date= end_test_date
    )
