
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, when
from pyspark.sql.types import FloatType

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_replace

def drop_columns_containing_word(df, word, exception=None):
    """
    Drop columns from a PySpark DataFrame if their names contain a specific word, 
    unless the column name matches the provided exception.

    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame from which to drop columns.
    word (str): The word to look for in column names.
    exception (str, optional): The column name that should not be dropped even if it contains the word.

    Returns:
    df (pyspark.sql.DataFrame): The DataFrame after dropping the columns.
    """
    for col in df.columns:
        if word.lower() in col.lower() and col != exception:
            df = df.drop(col)

    return df



class pysparkDataFrameTransformer:
    """
    A utility class that provides methods to perform transformations on a PySpark DataFrame.
    This class supports chaining of methods for a fluent-like API.

    Attributes:
    data_frame (DataFrame): The PySpark DataFrame to be transformed.
    """

    def __init__(self, data_frame: DataFrame):
        """
        Initializes a new instance of the DataFrameTransformer class.

        Parameters:
        data_frame (DataFrame): The PySpark DataFrame to be transformed.
        """
        self.data_frame = data_frame

    def replace_with_regex(self, column_names: list, pattern: str, replacement: str):
        """
        Replaces occurrences of a regular expression pattern in specified columns with a replacement string.

        Parameters:
        column_names (list): The names of the columns in which to replace the pattern.
        pattern (str): The regular expression pattern to replace.
        replacement (str): The string to use as the replacement.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for column_name in column_names:
            self.data_frame = self.data_frame.withColumn(column_name, 
                                                         regexp_replace(col(column_name), pattern, replacement))
        return self

    def replace_non_numeric_with_float(self, column_names: list, replacement: float):
        """
        Replaces non-numeric entries in specified columns with a replacement float.

        Parameters:
        column_names (list): The names of the columns in which to replace non-numeric entries.
        replacement (float): The float to use as the replacement.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for column_name in column_names:
            self.data_frame = self.data_frame.withColumn(
                column_name,
                when(col(column_name).cast("float").isNotNull(), col(column_name).cast("float")).otherwise(replacement)
            )
        return self

    def remove_rows_with_missing_values(self, column_names: list):
        """
        Removes rows from the DataFrame that contain missing values (NA, NAN) in any of the specified columns.

        Parameters:
        column_names (list): The names of the columns in which to look for missing values.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for column_name in column_names:
            self.data_frame = self.data_frame.filter(col(column_name).isNotNull())
        return self

    def binarize_column(self, column_names: list, cut_off: float, new_column_names: list = None, drop_original: bool = False):
        """
        Creates new categorical columns based on a cut-off value.
        The new columns will have entries 'below' if the corresponding entry in the input column is below the cut-off, 
        and 'above' if the entry is equal to or greater than the cut-off.

        Parameters:
        column_names (list): The names of the input columns.
        cut_off (float): The cut-off value.
        new_column_names (list, optional): The names of the new binarized columns. Defaults to None.
        drop_original (bool, optional): Whether to drop the original columns. Defaults to False.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for i, column_name in enumerate(column_names):
            new_column_name = new_column_names[i] if new_column_names else f"binarized_{column_name}"
            self.data_frame = self.data_frame.withColumn(
                new_column_name, 
                when(col(column_name) < cut_off, str(0)).otherwise(str(1))
            )
            if drop_original:
                self.data_frame = self.data_frame.drop(column_name)
        return self


import json
import random
import urllib

import numpy as np
import optuna
import pandas as pd
import ray
import requests
import shap
import xgboost
import yaml

from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

CPUS_PER_JOB : int = 8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



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

    def __init__(self, df, random_state=None, max_iter=10):
        """
        Constructs all the necessary attributes for the DataImputer object.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe with data.
        random_state : int, optional
            The seed used by the random number generator for the imputer (default is 100).
        max_iter : int, optional
            The maximum number of imputing iterations to perform (default is 10).
        """
        self.df = df
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(random_state=self.random_state, max_iter=self.max_iter)

    def fit_transform(self):
        """
        Fits the imputer on data and performs imputations, and then inverse transform to retain the original scale.

        Returns
        -------
        DataImputer
            The instance of the DataImputer.
        """
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        numerical_df = self.df[numerical_cols]

        # Scale numerical data
        scaled_df = pd.DataFrame(self.scaler.fit_transform(numerical_df.values), columns=numerical_df.columns)

        # Fit imputer and perform imputations on the scaled numerical dataset
        df_imputed = pd.DataFrame(self.imputer.fit_transform(scaled_df), columns=scaled_df.columns)

        # Inverse transform to retain the original scale
        original_scale_df = pd.DataFrame(self.scaler.inverse_transform(df_imputed), columns=df_imputed.columns)

        # Combine imputed numerical data with the categorical data
        categorical_cols = self.df.select_dtypes(exclude=np.number).columns
        df_imputed_with_categorical = pd.concat([original_scale_df, self.df[categorical_cols]], axis=1)

        self.df = df_imputed_with_categorical

        return self

    def insert_random_nans(self, probability : float = 0.2):
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


def encode_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical variables and converts all data to float type.

    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.

    Returns:
        df (pd.DataFrame): The processed DataFrame with categorical variables encoded and all data converted to float type.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype('category').cat.codes.astype('category')
        else:
            df[col] = df[col].astype('float')

    return df



class SparkDataProcessor:
    """
    This class is responsible for processing data using Apache Spark. It provides methods for data cleaning, 
    encoding categorical features and converting data types.

    Attributes:
        spark (SparkSession): The SparkSession object.
    """

    def __init__(self):
        """Initialize SparkDataProcessor with a SparkSession object."""
        self.spark = SparkSession.builder.appName("pku").master("local[*]").getOrCreate()

    @staticmethod
    def encode_and_convert(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical variables and converts all data to float type.

        Parameters:
            df (pd.DataFrame): The DataFrame to be processed.

        Returns:
            df (pd.DataFrame): The processed DataFrame with categorical variables encoded and all data converted to float type.
        """
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype('category').cat.codes.astype('category')
            else:
                df[col] = df[col].astype('float')

        return df

    def load_file(self, url=None, path=None):
        """
        Retrieves a file content from either a URL or a local file path.

        Parameters:
            url (str, optional): URL from where to retrieve the file content.
            path (str, optional): Local file path from where to retrieve the file content.

        Returns:
            content (str): The file content.
        """
        if url:
            response = requests.get(url)
            content = response.text
        elif path:
            with open(path, 'r') as file:
                content = file.read()
        else:
            raise ValueError('Either a URL or a local file path needs to be provided.')

        return content

    def process_data(self, data_csv_url=None, data_csv_path=None, drop_col_yml_url=None, drop_col_yml_path=None, rename_col_json_url=None, rename_col_json_path=None):
        """
        Process the data by removing, renaming, and transforming columns. The method handles missing values and 
        converts the PySpark DataFrame to a pandas DataFrame.

        Parameters:
            data_csv_url (str, optional): The URL from where to retrieve the CSV data file.
            data_csv_path (str, optional): The local file path from where to retrieve the CSV data file.
            drop_col_yml_url (str, optional): The URL from where to retrieve the parameters file in YAML format.
            drop_col_yml_path (str, optional): The local file path from where to retrieve the parameters file in YAML format.
            rename_col_json_url (str, optional): The URL from where to retrieve the rename mapping file in JSON format.
            rename_col_json_path (str, optional): The local file path from where to retrieve the rename mapping file in JSON format.

        Returns:
            df (pd.DataFrame): The processed DataFrame.

        Raises:
            ValueError: If neither URL nor local file path is provided for any input file.
            ValueError: If loading content from any input file fails.
        """

        # Ensure that either URL or local path is provided for each input file
        if not data_csv_url and not data_csv_path:
            raise ValueError('Either data_csv_url or data_csv_path must be provided.')
        
        # Load CSV data into Spark DataFrame
        csv_file_path = data_csv_path or urllib.request.urlretrieve(data_csv_url, filename="/tmp/data.csv")[0]
        df = self.spark.read.csv(csv_file_path, inferSchema=True, header=True)

        # Load parameters and rename mapping files
        drop_col_yml_content = self.load_file(drop_col_yml_url, drop_col_yml_path) if drop_col_yml_url or drop_col_yml_path else None
        rename_col_json_content = self.load_file(rename_col_json_url, rename_col_json_path) if rename_col_json_url or rename_col_json_path else None

        # Parse parameters and rename mapping
        params = yaml.safe_load(drop_col_yml_content) if drop_col_yml_content else None
        rename_dict = json.loads(rename_col_json_content) if rename_col_json_content else None

        # Drop columns as specified in the parameters file
        if params and "feature_engineering" in params and "removed_features" in params["feature_engineering"]:
            df = df.drop(*params["feature_engineering"]["removed_features"])

        # Rename columns based on the rename mapping
        if rename_dict:
            for old_name, new_name in rename_dict.items():
                df = df.withColumnRenamed(old_name, new_name)

        return self.encode_and_convert(df.toPandas()), df





class DataSplitter:
    """Una instancia del objeto con la data spliteada"""
    def __init__(self, test_size: float, random_state: int):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df : pd.DataFrame, label_col : str):
        """Corre un splitting stratificado"""
        # TODO: porque el Dataframe no es parte de la instancia del objeto ? 

        X = df.drop(label_col, axis=1)
        y = df[label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        X_train_df = pd.DataFrame(X_train, columns= X.columns )
        X_test_df  = pd.DataFrame(X_test,  columns= X.columns )
        y_train_df = pd.DataFrame(y_train, columns=[label_col])
        y_test_df  = pd.DataFrame(y_test,  columns=[label_col])

        return X_train_df, X_test_df, y_train_df.iloc[:, 0].astype('category'), y_test_df.iloc[:, 0].astype('category')


class ModelInstance:
    """Instancia de un modelo"""

    def __init__(
        self, df : pd.DataFrame, 
        target: str, test_size: float, 
        xgb_params: dict, kfold_splits: int, 
        seed: float = None
    ) -> None:
        if xgb_params is None:
            xgb_params = {"seed": seed}
        
        df        = DataImputer(df).fit_transform().df

        self.X_train, self.X_test, self.y_train, self.y_test = DataSplitter(
            test_size=test_size, random_state=seed
        ).split_data(df=df, label_col=target)

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
        testset = xgboost.DMatrix(self.X_test, label=self.y_test, enable_categorical=True)
        y_preds = self.model.predict(testset)

        return roc_auc_score(testset.get_label(), y_preds)

    def get_feature_explanation(self) -> pd.DataFrame:

        explainer = shap.TreeExplainer(self.model)

        # Extrae la explicacion SHAP en un DF
        explanation = explainer(self.X_test).cohorts(self.y_test.replace({0: "Healty", 1: "Abnormal"}).tolist())

        cohort_exps = list(explanation.cohorts.values())

        exp_shap_abnormal = pd.DataFrame(cohort_exps[0].values, columns=cohort_exps[0].feature_names)  # .abs().mean()

        exp_shap_healty = pd.DataFrame(cohort_exps[1].values, columns=cohort_exps[1].feature_names)  # .abs().mean()

        feature_metrics : pd.DataFrame = pd.concat(
            {
                "SHAP_healty": exp_shap_healty.abs().mean(),
                "SHAP_abnormal": exp_shap_abnormal.abs().mean(),
            },
            axis="columns",
        )

        return feature_metrics  # ["SHAP_abnormal"][a_feature]


def objective(
    trial, data : pd.DataFrame, target : str, shap_feature, 
    tuned_params=None, finetunning: bool = False, 
    ) -> tuple[float, float]:

    """
    The function that runs a single model and evaluates it.
    """

    if finetunning:
        seed = random.randint(1, 10_000)

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
        }

        if data[target].dtype != 'category': 
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
        seed=seed,
    )

    return (
        model_instance.get_AUC_on_test_data(),
        model_instance.get_feature_explanation()["SHAP_abnormal"][shap_feature],
    )


@ray.remote(num_cpus=CPUS_PER_JOB, max_retries=5)
def make_a_study(
        study_name: str, 
        data: pd.DataFrame, 
        target: str, 
        shap_feature: str,  # 
        n_jobs = -1,        # Por defecto todos los cores
        n_trials : int = 50
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
            finetunning=True
        ),
        n_trials=n_trials,
        n_jobs=CPUS_PER_JOB, # TODO: posiblemente pueda detectarlo ?
        catch=(
            TypeError,
            ValueError,
        ),
    )
    return hyperparameters_fine_tuning


def make_multiple_studies(
    data : ray.ObjectRef | pd.DataFrame, 
    features: list[str], 
    targets: list[str],
    n_trials : int = 50) -> list[optuna.Study | ray.ObjectRef ]:
    """Es un wrapper conveniente para multiples optimizadores"""

    return [make_a_study.remote(f"{f}_predicts_{t}", data, t, f, n_trials=n_trials) for f in features for t in targets if f != t]


# %% 
if __name__ == '__main__':
    
    
    ray.init(ignore_reinit_error=True)
        
    _ , spark_df = SparkDataProcessor().process_data(
        data_csv_path= "tirosinemia.csv", 
        # drop_col_yml_url= "https://raw.githubusercontent.com/DeepenData/Phenylketonuria_IR_and_Machine_Learning/alejandro/params.yml", 
        # rename_col_json_url="https://raw.githubusercontent.com/DeepenData/Phenylketonuria_IR_and_Machine_Learning/alejandro/labels_relabel.json"
        )

    OPBG = [ 'age at diagnosis (months)',            
                'ntbc dosis mg/kg/day',
                'ntbc levels (dbs)',
                'sca (urine)',
                'methionine dbs',
                'tyrosine dbs',
                'phenylalanine dbs',
                'methionine (plasma)',
                'tyrosine (plasma)',
                'phenylalanine (plasma)',
                'inr',
                'pt (sec)',
                'bili total',
                'gpt',
                'got',
                'ggt',
                'alkaline phosphatase',
                'alfa-fetoprotein',]

    spark_df = spark_df.select(
        *OPBG
    )


    spark_df = spark_df.drop(
        *['tiempo', 'código', 'pacientes', 'dosis ntbc', 'iq']
    )


    my_target = 'Alpha-Fet'
    spark_df2 = pysparkDataFrameTransformer(
                                            spark_df
                                            ).remove_rows_with_missing_values(
                                                ['alfa-fetoprotein', 'sca (urine)', 'ntbc levels (dbs)', 'ntbc levels (plasma)']
                                                ).binarize_column(column_names=['alfa-fetoprotein'], cut_off=5, new_column_names=[my_target], drop_original=True)

    df        =  encode_and_convert(spark_df2.data_frame.toPandas())
    #df        = DataImputer(df).fit_transform().df

    it_targets  : list[str] = [my_target]

    it_features : list[str] = df.columns.tolist()
    it_features.remove(it_targets[0])
    ray_data = ray.put(df)

    studies = make_multiple_studies(
        ray_data, 
        features = it_features,
        targets = it_targets,
        n_trials=10
    )

    studies = ray.get(studies)
    import pickle
    #saved_dict = { s.study_name : s for s in studies }
    with open('studies_distributed.pickle', 'wb') as handle:
        pickle.dump(studies, handle, protocol=pickle.HIGHEST_PROTOCOL)    
