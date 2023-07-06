import pickle
import optuna
import ray
from model_classes import ModelInstance

import pandas as pd

def objective(
    trial,
    data: pd.DataFrame,
    target: str,
    shap_feature,
    tuned_params=None,
    finetunning: bool = False,
    Independent_testset: bool = False,
    Independent_testset_df: pd.DataFrame = None,
) -> tuple[float, float]:  # sourcery skip: dict-assign-update-to-union
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


@ray.remote(max_retries=5)
def make_a_study(
    study_name: str,
    data: pd.DataFrame,
    target: str,
    shap_feature: str,  #
    n_jobs=-1,  # Por defecto todos los cores
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