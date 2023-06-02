import warnings
warnings.filterwarnings("ignore")
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, when
from pyspark.sql.types import FloatType

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_replace

import pandas as pd
from optuna.visualization._pareto_front import _get_pareto_front_info
from typing import Tuple
import optuna
import plotly.graph_objects as go

class TwoObjectiveSolutions:
    """
    A class that provides functionality for handling two-objective solutions in an Optuna study.

    Attributes:
    study (optuna.study.study.Study): The Optuna study object.
    df (pd.DataFrame): DataFrame containing all solutions.
    df_filtered (pd.DataFrame): DataFrame containing solutions that pass the provided cutoffs.
    filtered_trials (List[optuna.trial.FrozenTrial]): Trials that pass the provided cutoffs.
    """

    def __init__(self, study: optuna.study.study.Study, auc_cutoff: float = None, s_cutoff: float = None):
        """
        Initialize the TwoObjectiveSolutions object with an Optuna study.

        Args:
        study (optuna.study.study.Study): The Optuna study object.
        """
        self.study = study
        self.df = None
        self.df_filtered = None
        self.filtered_trials = None
        self.auc_cutoff = auc_cutoff
        self.s_cutoff   = s_cutoff
    def generate_dataframes(self):
        """
        Generate the dataframes for all solutions and the solutions that pass the provided cutoffs.

        Returns:
        self
        """
        all_trials = self.study.get_trials(deepcopy=False)
        pareto_trials = self.study.best_trials

        self.df = self._create_trials_dataframe(all_trials)

        # Identify Pareto optimal trials
        self.df['pareto'] = self.df['Trial'].isin([trial.number for trial in pareto_trials])

        if self.auc_cutoff is not None and self.s_cutoff is not None:
            self.df_filtered = self.df[(self.df["AUC"] > self.auc_cutoff) & (self.df["S"] > self.s_cutoff)]
        else:
            self.df_filtered = None

        return self
    
    
    def get_filtered_trials(self):
        """
        Generate and save trials from the study that surpass both AUC and S cutoffs.

        Args:
        auc_cutoff (float): The AUC cutoff. Only trials with AUC greater than this will be returned.
        s_cutoff (float): The S cutoff. Only trials with S greater than this will be returned.

        Returns:
        self
        """
        # if self.df is None or self.df_filtered is None:
        #     self.generate_dataframes(auc_cutoff, s_cutoff)

        all_trials = self.study.best_trials#(deepcopy=False)
        self.filtered_trials = []
        self.filtered_trials.extend(
            trial
            for trial in all_trials
            if (trial.values[0] > self.auc_cutoff) & (trial.values[1] > self.s_cutoff)
        )
        # for trial in all_trials:
        #     if any(self.df_filtered["AUC"] == trial.values[0] and self.df_filtered["S"] == trial.values[1]):
        #         self.filtered_trials.append(trial)

        for trial in self.filtered_trials:
            assert trial.values[0] > self.auc_cutoff, f"Trial {trial.number} has AUC ({trial.values[0]}) below the cutoff ({self.auc_cutoff})"
            assert trial.values[1] > self.s_cutoff, f"Trial {trial.number} has S ({trial.values[1]}) below the cutoff ({self.s_cutoff})"

        return self


    def plot_solutions(self):
        """
        Generate a scatter plot of the solutions.

        Returns:
        self
        """
        if self.df is None:
            self.generate_dataframes()

        fig = go.Figure()
        
        # Add a trace for the Pareto optimal trials
        pareto_df = self.df[self.df['pareto'] == True]
        fig.add_trace(go.Scatter(x=pareto_df["AUC"], y=pareto_df["S"], mode='markers', marker_symbol='square', name='Pareto Optimal Trials'))

        # Add a trace for the other trials
        other_trials_df = self.df[self.df['pareto'] == False]
        fig.add_trace(go.Scatter(x=other_trials_df["AUC"], y=other_trials_df["S"], mode='markers', name='Other Trials'))

        if self.auc_cutoff is not None and self.s_cutoff is not None:
            fig.add_shape(type="rect",
                        x0=self.auc_cutoff, y0=self.s_cutoff,
                        x1=max(self.df["AUC"]), y1=max(self.df["S"]),
                        line=dict(color="LightSeaGreen", width=2),
                        fillcolor="LightSeaGreen", opacity=0.3)
        
        # Update layout to remove margins
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
        )

        fig.show()

        return self


    def _create_trials_dataframe(self, trials):
        trial_numbers = [trial.number for trial in trials]
        auc = [trial.values[0] for trial in trials]
        s = [trial.values[1] for trial in trials]
        return pd.DataFrame({"Trial": trial_numbers, "AUC": auc, "S": s})
    
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_all_studies(studies, auc_cutoff, s_cutoff):
    # Calculate average "S" for Pareto optimal trials with AUC > auc_cutoff in each study
    study_avg_S = []
    for study in studies:
        solutions = TwoObjectiveSolutions(study, auc_cutoff, s_cutoff)
        solutions.generate_dataframes()
        pareto_trials = solutions.df[(solutions.df['pareto'] == True) & (solutions.df['AUC'] > auc_cutoff)]
        avg_S = pareto_trials['S'].mean() if not pareto_trials.empty else 0
        study_avg_S.append((study, avg_S))
    
    # Sort studies based on average "S"
    study_avg_S.sort(key=lambda x: x[1], reverse=True)

    num_studies = len(studies)
    num_cols = num_studies if num_studies <= 3 else num_studies // 3 + (num_studies % 3 > 0)
    
    fig = make_subplots(rows=3, cols=num_cols, subplot_titles=[study[0].study_name for study in study_avg_S])
    
    max_y = -float('inf')
    min_x = float('inf')
    max_x = -float('inf')

    # Iterate over each study
    for i, (study, _) in enumerate(study_avg_S):
        solutions = TwoObjectiveSolutions(study, auc_cutoff, s_cutoff)
        solutions.generate_dataframes()

        # update the max_y, min_x and max_x for the y-axes and x-axes range 
        max_y = max(max_y, solutions.df["S"].max())
        min_x = min(min_x, solutions.df["AUC"].min())
        max_x = 1.01*max(max_x, solutions.df["AUC"].max())

        # Pareto optimal trials
        pareto_df = solutions.df[solutions.df['pareto'] == True]
        fig.add_trace(go.Scatter(x=pareto_df["AUC"], y=pareto_df["S"], mode='markers', marker_symbol='square', name='Pareto Optimal Trials', legendgroup="group1", marker=dict(color='blue', opacity=0.5), showlegend=(i == 0)), row=(i%3)+1, col=(i//3)+1)

        # Other trials
        other_trials_df = solutions.df[solutions.df['pareto'] == False]
        fig.add_trace(go.Scatter(x=other_trials_df["AUC"], y=other_trials_df["S"], mode='markers', name='Other Trials', legendgroup="group2", marker=dict(color='red', opacity=0.5), showlegend=(i == 0)), row=(i%3)+1, col=(i//3)+1)

        # Add rectangle. Note that y1 = 1.1*max_y to ensure the rectangle reaches the end of y-axis in all subplots.
        fig.add_shape(type="rect",
              x0=solutions.auc_cutoff, y0=-0.02,  # y0 = minimum of y-axis range
              x1=1.0, y1=1.1*max_y,  # x1 = 1.0 (end of normalized x-axis), y1 = maximum of y-axis range
              line=dict(color="LightSeaGreen", width=2),
              fillcolor="LightSeaGreen", opacity=0.3,
              row=(i%3)+1, col=(i//3)+1)


    # Update y-axis range for all subplots
    fig.update_yaxes(range=[-0.02, 1.1*max_y])
    
    # Update x-axis range for all subplots
    fig.update_xaxes(range=[min_x, max_x], tickvals=[0.5,0.8, 1.0])

    # Update layout to remove margins and show legend
    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0), # increased top margin
        showlegend=True
    )
    
    # Update annotations (subplot titles) to decrease font size
    fig.update_annotations(dict(
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10),  # decrease font size to 10
    ))

    fig.show()

    
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# def plot_all_studies(studies, auc_cutoff, s_cutoff):
#     # First calculate the average "S" for the Pareto trials with AUC > auc_cutoff in each study
#     study_avg_S = []
#     for study in studies:
#         solutions = TwoObjectiveSolutions(study, auc_cutoff, s_cutoff)
#         solutions.generate_dataframes()
#         pareto_trials = solutions.df[(solutions.df['pareto'] == True) & (solutions.df['AUC'] > auc_cutoff)]
#         avg_S = pareto_trials['S'].mean() if not pareto_trials.empty else 0
#         study_avg_S.append((study, avg_S))
    
#     # Sort studies based on average "S"
#     study_avg_S.sort(key=lambda x: x[1], reverse=True)

#     num_studies = len(studies)
#     num_cols = num_studies if num_studies <= 3 else num_studies // 3 + (num_studies % 3 > 0)
    
#     fig = make_subplots(rows=3, cols=num_cols, subplot_titles=[study[0].study_name for study in study_avg_S])
    
#     max_y = -float('inf')
#     min_x = float('inf')
#     max_x = -float('inf')

#     for i, (study, _) in enumerate(study_avg_S):
#         solutions = TwoObjectiveSolutions(study, auc_cutoff, s_cutoff)
#         solutions.generate_dataframes()

#         # update the max_y, min_x and max_x for the y-axes and x-axes range 
#         max_y = max(max_y, solutions.df["S"].max())
#         min_x = min(min_x, solutions.df["AUC"].min())
#         max_x = 1.01*max(max_x, solutions.df["AUC"].max())

#         # Pareto optimal trials
#         pareto_df = solutions.df[solutions.df['pareto'] == True]
#         fig.add_trace(go.Scatter(x=pareto_df["AUC"], y=pareto_df["S"], mode='markers', marker_symbol='square', name='Pareto Optimal Trials', legendgroup="group1", marker=dict(color='blue', opacity=0.5), showlegend=(i == 0)), row=(i%3)+1, col=(i//3)+1)

#         # Other trials
#         other_trials_df = solutions.df[solutions.df['pareto'] == False]
#         fig.add_trace(go.Scatter(x=other_trials_df["AUC"], y=other_trials_df["S"], mode='markers', name='Other Trials', legendgroup="group2", marker=dict(color='red', opacity=0.5), showlegend=(i == 0)), row=(i%3)+1, col=(i//3)+1)

#         if solutions.auc_cutoff is not None and solutions.s_cutoff is not None:
#             fig.add_shape(type="rect",
#               x0=solutions.auc_cutoff, y0=-0.02,  # y0 = minimum of y-axis range
#               x1=1.0, y1=1.1*max_y,  # x1 = 1.0 (end of normalized x-axis), y1 = maximum of y-axis range
#               line=dict(color="LightSeaGreen", width=2),
#               fillcolor="LightSeaGreen", opacity=0.3,
#               row=(i%3)+1, col=(i//3)+1)

#     # Update y-axis range for all subplots
#     fig.update_yaxes(range=[-0.02, 1.1*max_y])
    
#     # Update x-axis range for all subplots
#     fig.update_xaxes(range=[min_x, max_x])

#     # Update layout to remove margins and show legend
#     fig.update_layout(
#         autosize=True,
#         margin=dict(l=0, r=0, t=30, b=0), # increased top margin
#         showlegend=True
#     )
    
#     # Update annotations (subplot titles) to decrease font size
#     fig.update_annotations(dict(
#             xref="paper",
#             yref="paper",
#             showarrow=False,
#             font=dict(size=10),  # decrease font size to 10
#     ))

#     fig.show()

def compare_dataframes(df1, df2):
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


def prepare_spark_df(
    spark_df, keep_cols, 
    cols_rows_with_missing_values:list,
    original_continuos_target: list[str]=['alfa-fetoprotein'] ,
    new_binary_target:str='Alpha-Fet'):
    
    
    spark_df = spark_df.select(
        *keep_cols
    )


    spark_df = spark_df.drop(
        *['tiempo', 'código', 'pacientes', 'dosis ntbc', 'iq']
    )

    
    
    spark_df2 = pysparkDataFrameTransformer(
                                            spark_df
                                            ).remove_rows_with_missing_values(
                                                cols_rows_with_missing_values
                                                ).binarize_column(column_names=original_continuos_target, cut_off=5, 
                                                                  new_column_names=[new_binary_target], drop_original=True)

    return encode_and_convert(spark_df2.data_frame.toPandas())


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

CPUS_PER_JOB : int = 5

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.exceptions import ConvergenceWarning

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
            random_state=self.random_state, max_iter=self.max_iter, initial_strategy = 'median')

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
        target: str, 
        xgb_params: dict = None, 
        test_size: float = .3, 
        kfold_splits: int = 3, 
        seed: float = None,
        Independent_testset: bool = False,
        Independent_testset_df: pd.DataFrame = None,
        verbose: bool = False

    ) -> None:
        if xgb_params is None:
            xgb_params = {"seed": seed}
            
        self.Independent_testset = Independent_testset
        self.verbose = verbose

        
        df        = DataImputer(df, random_state = seed).fit_transform().df
        
        
        if not self.Independent_testset:
            
            self.X_train, self.X_test, self.y_train, self.y_test = DataSplitter(
                test_size=test_size, random_state=seed
            ).split_data(df=df, label_col=target)
            
        else:
            #Use an independent testset
            Independent_testset_df  = DataImputer(Independent_testset_df, random_state = seed).fit_transform().df           
            
            self.X_train, _, self.y_train, _ = DataSplitter(
                test_size=test_size, random_state=seed
            ).split_data(df=df, label_col=target)
            
            self.X_test, _,  self.y_test, _  = DataSplitter(
                test_size=0.05, random_state=seed
            ).split_data(df=Independent_testset_df, label_col=target)
            

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
        #if not self.Independent_testset:
        testset = xgboost.DMatrix(self.X_test, label=self.y_test, enable_categorical=True)
        y_preds = self.model.predict(testset)
        if self.verbose:
            print(f"samples in test: {self.X_test.shape[0]}")

        return roc_auc_score(testset.get_label(), y_preds)

    def get_feature_explanation(self) -> pd.DataFrame:

        explainer = shap.TreeExplainer(self.model)

        # Extrae la explicacion SHAP en un DF
        #if not self.Independent_testset:
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
    Independent_testset: bool = False,
    Independent_testset_df: pd.DataFrame = None, 
    ) -> tuple[float, float]:

    """
    The function that runs a single model and evaluates it.
    """

    if finetunning:
        seed = trial.suggest_int("seed", 1, 10_000) #random.randint(1, 10_000)

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
        seed=seed,#trial.suggest_int("seed", 1, 10_000), 
        Independent_testset= Independent_testset,
        Independent_testset_df = Independent_testset_df,
        
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
        n_trials : int = 50,
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
            Independent_testset    = Independent_testset,
            Independent_testset_df = Independent_testset_df
        ),
        n_trials=n_trials,
        n_jobs=-1, # TODO: posiblemente pueda detectarlo ?
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
    n_trials : int = 50, 
    Independent_testset: bool = False,
    Independent_testset_df: pd.DataFrame = None, 
    ) -> list[optuna.Study | ray.ObjectRef ]:
    """Es un wrapper conveniente para multiples optimizadores"""

    return [make_a_study.remote(f"{f}", data, t, f, n_trials=n_trials, 
                                Independent_testset = Independent_testset,
                                Independent_testset_df = Independent_testset_df) for f in features for t in targets if f != t]
