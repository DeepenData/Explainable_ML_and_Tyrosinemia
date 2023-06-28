# %% 
# BASE PARA WORKFLOW
import logging

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="workflow.log",
    )

logging.info("Starting workflow")

# %%
# PARTE DE IMPORTAR PARSEAR LOS SHEETS DE CAMI
import gspread
import pandas as pd

def sheet_to_dataframe(
    key : str,          # The Google Sheet name key
    gc : gspread.Client # The client for Google Sheets # TODO: default client in environment?
    ) -> pd.DataFrame:
    """Reads a Google Sheet into a DataFrame"""

    # Now you can open your Google Sheets document
    spreadsheet = gc.open_by_key( key )  # carga todo el google sheet

    # Get the list of all worksheets in the spreadsheet
    worksheets = spreadsheet.worksheets()  # saca todas las worksheets
    logging.info(f"Reading from Google Sheets data\n\tFILE {spreadsheet}\n\tSHEETS {worksheets}")

    # Create a list to hold all dataframes
    df_list : list[pd.DataFrame] = []

    # Loop through all worksheets and convert each into a dataframe
    for worksheet in worksheets[3:]:  # saqué las primeras 3 pq no eran de los tiempos de los pacientes

        # DATA INTO A DATAFRAME
        # Get all the values in the worksheet
        data : list[list] = worksheet.get_all_values()
        # Convert the data to a dataframe
        df = pd.DataFrame(data)
        # Set the column names to the values in the first row
        df.columns = df.iloc[0]
        # eliminar del df la primera columna
        df = df.iloc[1:]

        # CLEANSE OF THE COLUMN NAMES
        # sacar los espacios y que quede todo en minuscula los nombres de las  columnas
        df.columns = df.columns.str.strip().str.lower()
        # eliminar cualquier numero que esté antes de una palabra
        df.columns = df.columns.str.replace(r"^\b(?:13|12|11|10|[1-9])\b", " ", regex=True)
        # en minusculas eliminar los espacios
        df.columns = df.columns.str.strip().str.lower()

        logging.info(
            f"SHEET {worksheet} has {df.shape[0]} entries, {df.shape[1]} features\n\t {df.columns}"
        )

        # Add the dataframe to the list
        df_list.append(df)  # una lista de dataframes

    # PARSE ALL COLUMNS INTO A COMMON LIST
    all_columns = pd.Index([])
    for df in df_list:
        all_columns = all_columns.union(df.columns)

    logging.info(
        f"Final list of columns: {all_columns}"
    )

    final_df = (pd.concat(df_list, axis='index')
        .set_index(['tiempo','código'])
        .apply(pd.to_numeric, errors = 'ignore')
        .convert_dtypes()
    )

    return final_df


def compare_dataframes(df1 : pd.DataFrame, df2 : pd.DataFrame) -> None:
    """Sanity check for comparing dataframes"""
    # Read CSV files into pandas DataFrames
    # Compare number of columns
    if len(df1.columns) != len(df2.columns):
        print(f"Number of columns are different. CSV1 has {len(df1.columns)} columns and CSV2 has {len(df2.columns)} columns.")
    else:
        print(f"Both CSV files have the same number of columns: {len(df1.columns)}")


    # Compare column names
    if set(df1.columns) != set(df2.columns):
        print("Column names are different:")
        print("Columns in first CSV but not in second: ", set(df1.columns) - set(df2.columns))
        print("Columns in second CSV but not in first: ", set(df2.columns) - set(df1.columns))
    else:
        print("Column names are the same.")

    # Compare column order
    if df1.columns.tolist() != df2.columns.tolist():
        print("Column order is different.")
        print("First CSV columns: ", df1.columns.tolist())
        print("Second CSV columns: ", df2.columns.tolist())
    else:
        print("Column order is the same.")

    # Compare column data types
    dtype_diff = False
    for column in df1.columns.intersection(df2.columns):
        if df1[column].dtype != df2[column].dtype:
            dtype_diff = True
            print(f"Column '{column}' has different data types:")
            print(f"Data type in first CSV: {df1[column].dtype}")
            print(f"Data type in second CSV: {df2[column].dtype}")
    if not dtype_diff:
        print("All common columns have the same data types.")

# %% ---- PROCESADO EN SI
# Conexion a Google Sheets
gc = gspread.service_account(
    # PARAM: service account for google sheets
    filename='credentials/gcp.credentials.json'
)

# df1 --- Cohorte de Chile
df1 = (
    sheet_to_dataframe(key = "1aHCPVXG9Lv9Eg_V7bGd9Zj9jGbXcS8m0lDp9s8HDF-c", gc=gc)
    .drop('nut status', axis = 'columns')
)
df1.to_csv('data/tirosinemia_1aHCPVXG9Lv9Lv.csv')

# df2 --- Cohorte de Italia
df2 = (
    sheet_to_dataframe(key = "1vpFD-SLCcub_QuUFohRyt9W7LI7vAWiGTQRSJxVnAro", gc=gc)
    .drop('data record (not for analysis)', axis = 'columns')
)
df2.to_csv('data/tirosinemia_italia_1vpFD-SLCcub.csv')


# %%
# PARTE DE PARSEAR DICHOS SHEETS A PARQUET DE ALEJANDRO
# pierde el punto en que originalmente esto estaba pensado en HPC

#fmt: off
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

# Common filtering of the cohorts, so they pass the same names
def cohort_filter(df : pd.DataFrame) -> pd.DataFrame: 
    """Drops the rows (samples) that have NaNs in any of the REQUIERE_COMPLETE features"""
    return df[ OPBG ].dropna(axis='rows', subset=REQUIERE_COMPLETE)

# %% 
# FILTRADO EN SI

chile_cohort = cohort_filter(df1)
italy_cohort = cohort_filter(df2)
rome_cohort  = italy_cohort.loc[ italy_cohort.index.get_level_values('código').str.startswith('R') ]
flor_cohort  = italy_cohort.loc[ italy_cohort.index.get_level_values('código').str.startswith('F') ]

# Note, the dtypes may be different between cohorts, as some don't have decimals and thus default to int.
# This shouldn't affect functionality, but it should be corrected next. 

[ f"Using of {cohort.shape[0]} samples" for cohort in [chile_cohort, rome_cohort, flor_cohort]]

# %%
# PARTE DE USAR ESTO PARA CONSTRUIR Y TESTEAR MODELOS DE MANU
import ray

# hpc_utils.py
import warnings

warnings.filterwarnings("ignore")

import optuna
import xgboost
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.experimental import enable_iterative_imputer # Requiered
from sklearn.impute import IterativeImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

import numpy as np

class DataImputer:
    """
    A class to impute missing numerical values in a pandas DataFrame.

    Attributes
    ----------
    df : pandas.DataFrame
        The input dataframe with data.
    random_state : int
        The seed used by the random number generator for the imputer.
    max_iter : int
        The maximum number of imputing iterations to perform.
    scaler : sklearn.StandardScaler
        The standard scaler object.
    imputer : sklearn.IterativeImputer
        The iterative imputer object.

    Methods
    -------
    fit_transform():
        Fits the imputer on data and returns the imputed DataFrame.
    insert_random_nans(probability: float = 0.2):
        Inserts random NaNs to numerical columns in the dataframe.
    """

    def __init__(self, df, random_state=None, max_iter=50):
        """
        Constructs all the necessary attributes for the DataImputer object.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe with data.
        random_state : int, optional
            The seed used by the random number generator for the imputer (default is None).
        max_iter : int, optional
            The maximum number of imputing iterations to perform (default is 10).
        """
        self.df = df
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(
            random_state=self.random_state, max_iter=self.max_iter, initial_strategy="median"
        )

    def fit_transform(self):
        """
        Fits the imputer on data and performs imputations, and then inverse transform to retain the original scale.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with imputed values.
        """
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        numerical_df = self.df[numerical_cols]

        # Scale numerical data
        scaled_df = pd.DataFrame(self.scaler.fit_transform(numerical_df.values), columns=numerical_df.columns)

        # Fit imputer and perform imputations on the scaled numerical dataset
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            df_imputed = pd.DataFrame(self.imputer.fit_transform(scaled_df), columns=scaled_df.columns)

        # Inverse transform to retain the original scale
        original_scale_df = pd.DataFrame(self.scaler.inverse_transform(df_imputed), columns=df_imputed.columns)

        # Combine imputed numerical data with the categorical data
        categorical_cols = self.df.select_dtypes(exclude=np.number).columns
        df_imputed_with_categorical = pd.concat([original_scale_df, self.df[categorical_cols]], axis=1)

        self.df = df_imputed_with_categorical

        return self

    def insert_random_nans(self, probability: float = 0.2):
        """
        Inserts random NaNs to numerical columns in the dataframe with a specified probability.

        Parameters
        ----------
        probability : float, optional
            The probability of a value being replaced with NaN (default is 0.2).

        Returns
        -------
        DataImputer
            The instance of the DataImputer.
        """
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        df_with_nans = self.df.copy()

        np.random.seed(self.random_state)

        for col in numerical_cols:
            mask = np.random.choice([False, True], size=len(df_with_nans), p=[1 - probability, probability])
            df_with_nans.loc[mask, col] = np.nan

        self.df = df_with_nans

        return self


class DataSplitter:
    """Una instancia del objeto con la data spliteada"""

    def __init__(self, test_size: float, random_state: int):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, label_col: str):
        """Corre un splitting stratificado"""
        # TODO: porque el Dataframe no es parte de la instancia del objeto ?

        X = df.drop(label_col, axis=1)
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)
        y_train_df = pd.DataFrame(y_train, columns=[label_col])
        y_test_df = pd.DataFrame(y_test, columns=[label_col])

        return X_train_df, X_test_df, y_train_df.iloc[:, 0].astype("category"), y_test_df.iloc[:, 0].astype("category")


class ModelInstance:
    """Instancia de un modelo"""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        xgb_params: dict = None,
        test_size: float = 0.3,
        kfold_splits: int = 3,
        seed: float = None,
        Independent_testset: bool = False,
        Independent_testset_df: pd.DataFrame = None,
        verbose: bool = False,
    ) -> None:
        if xgb_params is None:
            xgb_params = {"seed": seed}

        self.Independent_testset = Independent_testset
        self.verbose = verbose

        df = DataImputer(df, random_state=seed).fit_transform().df

        if not self.Independent_testset:
            self.X_train, self.X_test, self.y_train, self.y_test = DataSplitter(
                test_size=test_size, random_state=seed
            ).split_data(df=df, label_col=target)

        else:
            # Use an independent testset
            Independent_testset_df = DataImputer(Independent_testset_df, random_state=seed).fit_transform().df

            self.X_train, _, self.y_train, _ = DataSplitter(test_size=0.1, random_state=seed).split_data(
                df=df, label_col=target
            )

            self.y_test = Independent_testset_df[target]
            self.X_test = Independent_testset_df.drop(target, axis=1, inplace=False)

        cv = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=seed)
        folds = list(cv.split(self.X_train, self.y_train))

        for train_idx, val_idx in folds:
            # Sub-empaquetado del train-set en formato de XGBoost
            dtrain = xgboost.DMatrix(
                self.X_train.iloc[train_idx, :],
                label=self.y_train.iloc[train_idx],
                enable_categorical=True,
            )
            dval = xgboost.DMatrix(
                self.X_train.iloc[val_idx, :],
                label=self.y_train.iloc[val_idx],
                enable_categorical=True,
            )

            self.model = xgboost.train(
                dtrain=dtrain,
                params=xgb_params,
                evals=[(dtrain, "train"), (dval, "val")],
                num_boost_round=1000,
                verbose_eval=False,
                early_stopping_rounds=10,
            )

    def get_AUC_on_test_data(self) -> float:
        # if not self.Independent_testset:
        testset = xgboost.DMatrix(self.X_test, label=self.y_test, enable_categorical=True)
        y_preds = self.model.predict(testset)
        if self.verbose:
            print(f"samples in test: {self.X_test.shape[0]}")

        return roc_auc_score(testset.get_label(), y_preds)

    def get_feature_explanation(self) -> pd.DataFrame:
        explainer = shap.TreeExplainer(self.model)

        # Extrae la explicacion SHAP en un DF
        # if not self.Independent_testset:
        explanation = explainer(self.X_test).cohorts(self.y_test.replace({0: "Healty", 1: "Abnormal"}).tolist())

        cohort_exps = list(explanation.cohorts.values())

        exp_shap_abnormal = pd.DataFrame(cohort_exps[0].values, columns=cohort_exps[0].feature_names)  # .abs().mean()

        exp_shap_healty = pd.DataFrame(cohort_exps[1].values, columns=cohort_exps[1].feature_names)  # .abs().mean()

        feature_metrics: pd.DataFrame = pd.concat(
            {
                "SHAP_healty": exp_shap_healty.abs().mean(),
                "SHAP_abnormal": exp_shap_abnormal.abs().mean(),
            },
            axis="columns",
        )

        return feature_metrics  # ["SHAP_abnormal"][a_feature]


def objective(
    trial,
    data: pd.DataFrame,
    target: str,
    shap_feature,
    tuned_params=None,
    finetunning: bool = False,
    Independent_testset: bool = False,
    Independent_testset_df: pd.DataFrame = None,
) -> tuple[float, float]:
    """
    The function that runs a single model and evaluates it.
    """

    if finetunning:
        seed = trial.suggest_int("seed", 1, 10_000)  # random.randint(1, 10_000)

        # TOOD: definir fuera?
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int(
                "max_depth",
                2,
                10,
            ),
            "eta": trial.suggest_float("eta", 0.01, 0.5),
            "subsample": trial.suggest_float("subsample", 0.1, 0.5),
            "lambda": trial.suggest_float("lambda", 0, 1),
            "alpha": trial.suggest_float("alpha", 0, 1),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0, 5),
            "seed": seed,
        }

        if data[target].dtype != "category":
            params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    else:
        params = tuned_params
        seed = trial.suggest_int("seed", 1, 10_000)

    model_instance = ModelInstance(
        df=data,
        target=target,
        test_size=0.3,
        xgb_params=params,
        kfold_splits=trial.suggest_int("kfold_splits", 2, 5),
        seed=seed,  # trial.suggest_int("seed", 1, 10_000),
        Independent_testset=Independent_testset,
        Independent_testset_df=Independent_testset_df,
    )

    return (
        model_instance.get_AUC_on_test_data(),
        model_instance.get_feature_explanation()["SHAP_abnormal"][shap_feature],
    )


CPUS_PER_JOB = 5


@ray.remote(num_cpus=CPUS_PER_JOB, max_retries=5)
def make_a_study(
    study_name: str,
    data: pd.DataFrame,
    target: str,
    shap_feature: str,  #
    n_jobs=CPUS_PER_JOB,  # Por defecto todos los cores
    n_trials: int = 50,
    Independent_testset: bool = False,
    Independent_testset_df: pd.DataFrame = None,
) -> optuna.Study:
    # Instancia el estudio
    hyperparameters_fine_tuning = optuna.create_study(
        study_name=study_name,
        load_if_exists=False,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(),
    )

    # Corre el run de optimizacion
    hyperparameters_fine_tuning.optimize(
        lambda trial: objective(
            trial,
            data,
            target=target,
            shap_feature=shap_feature,
            finetunning=True,
            Independent_testset=Independent_testset,
            Independent_testset_df=Independent_testset_df,
        ),
        n_trials=n_trials,
        n_jobs=CPUS_PER_JOB,  # TODO: posiblemente pueda detectarlo ?
        catch=(
            TypeError,
            ValueError,
        ),
    )
    return hyperparameters_fine_tuning


def make_multiple_studies(
    data: ray.ObjectRef | pd.DataFrame,
    features: list[str],
    targets: list[str],
    n_trials: int = 50,
    Independent_testset: bool = False,
    Independent_testset_df: pd.DataFrame = None,
) -> list[optuna.Study | ray.ObjectRef]:
    """Es un wrapper conveniente para multiples optimizadores"""

    return [
        make_a_study.remote(
            f"{f} ({t})",
            data,
            t,
            f,
            n_trials=n_trials,
            Independent_testset=Independent_testset,
            Independent_testset_df=Independent_testset_df,
        )
        for f in features
        for t in targets
        if f != t
    ]


# to_ray_cluster.py
import pickle


def launch_to_ray(
    df_train: pd.DataFrame, binary_target: list[str], save_to: str, n_trials=50, df_test: pd.DataFrame = None
):
    df_train[binary_target] = df_train[binary_target].astype("category")

    df_train.reset_index(drop=True, inplace=True)

    if type(df_test) != type(None):  # CURSED
        df_test[binary_target] = df_test[binary_target].astype("category")
        df_test.reset_index(drop=True, inplace=True)

    ray_df_train = ray.put(df_train)
    ray_df_test = ray.put(df_test)
    features: list[str] = df_train.columns.tolist()
    features.remove(binary_target)
    # return ray_df_train, ray_df_test, features

    studies = make_multiple_studies(
        ray_df_train,
        features=features,
        targets=[binary_target],
        n_trials=n_trials,
        Independent_testset=type(df_test) != type(None),
        Independent_testset_df=ray_df_test,
    )

    S = ray.get(studies)
    with open(f"{save_to}.pickle", "wb") as handle:
        pickle.dump(S, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    
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

    # df_chile = pd.read_parquet("data/df_chile.parquet.gzip")
    # df_rome  = pd.read_parquet("data/df_rome.parquet.gzip")
    # df_flor  = pd.read_parquet("data/df_flor.parquet.gzip")
    # df_italy = pd.read_parquet("data/df_italy.parquet.gzip")
    #
    # launch_to_ray(df_train = df_chile , binary_target= binary_target, save_to = "studies_1k_chile", n_trials = 3, Independent_testset = False)
    # launch_to_ray(df_train = df_chile , binary_target= binary_target, save_to = "studies_1k_rome", n_trials =  3, Independent_testset = True, df_test = df_rome)
    # launch_to_ray(df_train = df_chile , binary_target= binary_target, save_to = "studies_1k_flor" , n_trials =   3,Independent_testset = True, df_test = df_flor)
    # launch_to_ray(df_train = df_chile , binary_target= binary_target, save_to = "studies_1k_italy" , n_trials =  3,Independent_testset = True, df_test = df_italy)

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
