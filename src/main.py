# %%
# BASE PARA WORKFLOW
import logging
import warnings

import gspread
import ray
from sklearn.experimental import enable_iterative_imputer  # Requiered

from model_optimization import launch_to_ray
from data_preprocessing import cohort_filter, sheet_to_dataframe

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="workflow.log",
    )

    logging.info("Starting workflow")

    # %%
    # PARTE DE IMPORTAR PARSEAR LOS SHEETS DE CAMI

    # %% ---- PROCESADO EN SI
    # Conexion a Google Sheets
    gc = gspread.service_account(
        # PARAM: service account for google sheets
        filename="credentials/gcp.credentials.json"
    )

    # df1 --- Cohorte de Chile
    df1 = sheet_to_dataframe(key="1aHCPVXG9Lv9Eg_V7bGd9Zj9jGbXcS8m0lDp9s8HDF-c", gc=gc).drop("nut status", axis="columns")
    df1.to_csv("data/tirosinemia_1aHCPVXG9Lv9Lv.csv")

    # df2 --- Cohorte de Italia
    df2 = sheet_to_dataframe(key="1vpFD-SLCcub_QuUFohRyt9W7LI7vAWiGTQRSJxVnAro", gc=gc).drop(
        "data record (not for analysis)", axis="columns"
    )
    df2.to_csv("data/tirosinemia_italia_1vpFD-SLCcub.csv")


    # %%
    # PARTE DE PARSEAR DICHOS SHEETS A PARQUET DE ALEJANDRO
    # pierde el punto en que originalmente esto estaba pensado en HPC

    # fmt: off
    OPBG : list[str] = [
        # 'código',
        'age at diagnosis (months)',            
        'ntbc dosis mg/kg/day',
        'ntbc levels (dbs)',
        "suac", #'sca (urine)',
        'methionine (plasma)',
        'tyrosine (plasma)',
        'phenylalanine (plasma)',
        'pt (sec)',
        'bili total',
        'gpt',
        'got',
        'ggt',
        'alkaline phosphatase',
        'alfa-fetoprotein',
        'glicemia'
    ]
    logging.info(f"Used features: {OPBG}")

    REQUIERE_COMPLETE : list[str] = [
        'suac',
        'alfa-fetoprotein',
    ]
    logging.info(f"Requiered features: {REQUIERE_COMPLETE}")

    # %% 
    # FILTRADO EN SI

    chile_cohort = cohort_filter(df1, OPBG, REQUIERE_COMPLETE )
    italy_cohort = cohort_filter(df2, OPBG, REQUIERE_COMPLETE )
    rome_cohort  = italy_cohort.loc[ italy_cohort.index.get_level_values('código').str.startswith('R') ]
    flor_cohort  = italy_cohort.loc[ italy_cohort.index.get_level_values('código').str.startswith('F') ]

    # Note, the dtypes may be different between cohorts, as some don't have decimals and thus default to int.
    # This shouldn't affect functionality, but it should be corrected next. 

    [ f"Using of {cohort.shape[0]} samples" for cohort in [chile_cohort, rome_cohort, flor_cohort]]

    # %%
    # PARTE DE USAR ESTO PARA CONSTRUIR Y TESTEAR MODELOS DE MANU
    # hpc_utils.py

    warnings.filterwarnings("ignore")

    CPUS_PER_JOB = 5

    # Non canonical Alfa-fet
    binary_target = "Alpha-Fet"
    for cohort in [chile_cohort, italy_cohort, rome_cohort, flor_cohort]:
        cohort[binary_target] = (cohort['alfa-fetoprotein'] > 5.2 ).astype('Int8')
        cohort.drop( columns = 'alfa-fetoprotein',  inplace=True )

    # import sys
    # assert len(sys.argv) > 3, "Not enough arguments provided"

    # Parses the arguments from the command line
    #fmt: off
    #DF_TRAIN     = # sys.argv[1]
    #DF_TEST      = # sys.argv[2]
    TRIALS_RUN   = 10 # int(sys.argv[3])
    CPUS_PER_JOB =  5 # int(sys.argv[4])
    #SAVE_TO      = # f"results/{sys.argv[5]}"

    ray.init(ignore_reinit_error=True)  # Initialize ray cluster

    RUNS = {
        "chile_cohort" : ( chile_cohort , None ),
        "italy_cohort" : ( chile_cohort , italy_cohort ),
        "rome_cohort"  : ( chile_cohort , rome_cohort  ),
        "flor_cohort"  : ( chile_cohort , flor_cohort  ),
    }

    K_num = lambda x : str(x) if x < 1000 else f"{round(x/1000 ,1)}k"
    # Simle 10,000 -> 10k formatter

    for study_name, datasets in RUNS.items():
        launch_to_ray(
            df_train=datasets[0], df_test=datasets[1],
            save_to=f"results/{study_name}_{K_num(TRIALS_RUN)}",
            n_trials=TRIALS_RUN,
            binary_target=binary_target
        )
