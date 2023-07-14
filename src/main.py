# %%
# BASE PARA WORKFLOW
import logging
import warnings

import gspread
import pandas as pd
import ray
import yaml
from sklearn.experimental import enable_iterative_imputer  # Requiered

warnings.filterwarnings("ignore")
from data_preprocessing import cohort_filter, sheet_to_dataframe
from model_optimization import launch_to_ray

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="workflow.log",
    )

    logging.info("Starting workflow")

    with open("parameters.yml", "r") as f:
        params = yaml.safe_load(f)

    gspread_creds_file : str = params["gspread_creds_file"]

    features          : list[str] = params["features"]
    required_features : list[str] = params["required_features"]

    binary_target : str = params["binary_target"]
    trials_run    : int = params["trials_run"]
    cpus_per_job  : int = params["cpus_per_job"]

    # Conexion a Google Sheets
    # Use the parameters in the code
    gc = gspread.service_account(filename=gspread_creds_file)

    # Reads the dataframes defined in the parameters.yml
    gsheets_keys : list[dict[str, str | list[str]]] = params["gsheets_keys"]
    dataframes : dict[str, pd.DataFrame] = {}
    for idx in gsheets_keys:
        dataframes[ idx['df_name'] ] = sheet_to_dataframe(key=idx['key'], gc=gc).drop( idx['drop'] , axis="columns")
        dataframes[ idx['df_name'] ].to_csv(f"data/{idx['df_name']}.csv")


    OPBG = params["features"]; 
    logging.info(f"Used features: {OPBG}")

    REQUIERE_COMPLETE = params["required_features"]; 
    logging.info(f"Required features: {REQUIERE_COMPLETE}")

    # MANUAL BLOCK --- PARAMETRIZE THIS? 
    chile_cohort = cohort_filter( dataframes['chile_cohort'] , OPBG, REQUIERE_COMPLETE)
    italy_cohort = cohort_filter( dataframes['italy_cohort'] , OPBG, REQUIERE_COMPLETE)
    rome_cohort = italy_cohort.loc[italy_cohort.index.get_level_values('código').str.startswith('R')]
    flor_cohort = italy_cohort.loc[italy_cohort.index.get_level_values('código').str.startswith('F')]

    # Defines the binary target
    for cohort in [chile_cohort, italy_cohort, rome_cohort, flor_cohort]:
        cohort[binary_target] = (cohort['alfa-fetoprotein'] > 5.2).astype('int')
        cohort.drop(columns='alfa-fetoprotein', inplace=True)

    ray.init(ignore_reinit_error=True)

    RUNS = {
        "chile_cohort": (chile_cohort, None),
        # "italy_cohort": (chile_cohort, italy_cohort),
        # "rome_cohort": (chile_cohort, rome_cohort),
        "flor_cohort": (chile_cohort, flor_cohort),
    }

    # Simle 10,000 -> 10k formatter
    K_num = lambda x : str(x) if x < 1000 else f"{round(x/1000 ,1)}k"

    for study_name, datasets in RUNS.items():
        launch_to_ray(
            df_train=datasets[0], df_test=datasets[1],
            save_to=f"results/{study_name}_{K_num(trials_run)}",
            n_trials=trials_run,
            binary_target=binary_target
        )
