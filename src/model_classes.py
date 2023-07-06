import warnings

import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # Requiered
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


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