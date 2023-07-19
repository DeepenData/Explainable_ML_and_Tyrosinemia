import pickle
import warnings

import optuna
import pandas as pd
import plotly.graph_objects as go
from optuna import Study
from optuna.trial import TrialState

warnings.filterwarnings("ignore")


def get_studies(a_study: str) -> list[Study]:
    """Simple read pickle method"""
    with open(f"{a_study}.pickle", "rb") as handle:
        studies = pickle.load(handle)

    return studies


def filter_studies(studies: list[Study]) -> list[Study]:
    """Returns only the completed studies"""
    new_studies = []
    for study in studies:
        # Get all completed trials in the study
        completed_trials = study.get_trials(states=[TrialState.COMPLETE])
        # Create a new study with the same study name but meant for multi-objective optimization
        new_study = optuna.create_study(study_name=study.study_name, directions=["maximize", "maximize"])
        # Create and add each trial to the new study
        for trial in completed_trials:
            new_trial = optuna.create_trial(
                params=trial.params,
                distributions=trial.distributions,
                values=trial.values,  # Use values instead of value for multi-objective optimization
                state=TrialState.COMPLETE,
            )
            new_study.add_trial(new_trial)
        new_studies.append(new_study)

    # Asserting that all trials in the new study are completed
    for study in new_studies:
        for trial in study.get_trials():
            assert trial.state == TrialState.COMPLETE

    return new_studies


# %% ---
class TwoObjectiveSolutions:
    """
    A class that provides functionality for handling two-objective solutions in an Optuna study.
    Attributes:
        study (optuna.study.study.Study): The Optuna study object.
        df (pd.DataFrame): DataFrame containing all solutions.
        df_filtered (pd.DataFrame): DataFrame containing solutions that pass the provided cutoffs.
        filtered_trials (List[optuna.trial.FrozenTrial]): Trials that pass the provided cutoffs.
    """

    def __init__(self, study: Study, auc_cutoff: float = None, s_cutoff: float = None):
        """
        Initialize the TwoObjectiveSolutions object with an Optuna study.
        Args:
        study (Study): The Optuna study object.
        """
        self.study = study
        self.df = None
        self.df_filtered = None
        self.filtered_trials = None
        self.auc_cutoff = auc_cutoff
        self.s_cutoff = s_cutoff

    def generate_dataframes(self):
        """Generate the dataframes for all solutions and the solutions that pass the provided cutoffs."""
        all_trials = self.study.get_trials(deepcopy=False)
        pareto_trials = self.study.best_trials

        self.df = self._create_trials_dataframe(all_trials)

        # Identify Pareto optimal trials
        self.df["pareto"] = self.df["Trial"].isin([trial.number for trial in pareto_trials])

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
        """
        # if self.df is None or self.df_filtered is None:
        #     self.generate_dataframes(auc_cutoff, s_cutoff)

        all_trials = self.study.best_trials  # (deepcopy=False)
        self.filtered_trials = []
        self.filtered_trials.extend(
            trial for trial in all_trials if (trial.values[0] > self.auc_cutoff) & (trial.values[1] > self.s_cutoff)
        )
        # for trial in all_trials:
        #     if any(self.df_filtered["AUC"] == trial.values[0] and self.df_filtered["S"] == trial.values[1]):
        #         self.filtered_trials.append(trial)

        for trial in self.filtered_trials:
            assert (
                trial.values[0] > self.auc_cutoff
            ), f"Trial {trial.number} has AUC ({trial.values[0]}) below the cutoff ({self.auc_cutoff})"
            assert (
                trial.values[1] > self.s_cutoff
            ), f"Trial {trial.number} has S ({trial.values[1]}) below the cutoff ({self.s_cutoff})"

        return self

    def plot_solutions(self):
        """Generate a scatter plot of the solutions."""
        if self.df is None:
            self.generate_dataframes()

        fig = go.Figure()

        # Add a trace for the Pareto optimal trials
        pareto_df = self.df[self.df["pareto"] == True]
        fig.add_trace(
            go.Scatter(
                x=pareto_df["AUC"],
                y=pareto_df["S"],
                mode="markers",
                marker_symbol="square",
                name="Pareto Optimal Trials",
            )
        )

        # Add a trace for the other trials
        other_trials_df = self.df[self.df["pareto"] == False]
        fig.add_trace(go.Scatter(x=other_trials_df["AUC"], y=other_trials_df["S"], mode="markers", name="Other Trials"))

        if self.auc_cutoff is not None and self.s_cutoff is not None:
            fig.add_shape(
                type="rect",
                x0=self.auc_cutoff,
                y0=self.s_cutoff,
                x1=max(self.df["AUC"]),
                y1=max(self.df["S"]),
                line=dict(color="LightSeaGreen", width=2),
                fillcolor="LightSeaGreen",
                opacity=0.3,
            )

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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_studies(studies: list[Study], color: str, main_title: str) -> plt.Figure:
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    names = []
    impotances = []

    for study in studies:
        solutions = TwoObjectiveSolutions(study, auc_cutoff=0.7, s_cutoff=0).get_filtered_trials().filtered_trials

        names.append(study.study_name)
        impotances.append([solution.values for solution in solutions])
    data = dict(zip(names, impotances))

    # Remove substring "(Alpha-Fet)" from all the keys
    data = {key.replace(" (Alpha-Fet)", ""): value for key, value in data.items()}

    # Calculate the median of y-values for each key and sort the dictionary based on it
    medians = {key: np.mean([i[1] for i in value]) for key, value in data.items()}
    data = {k: v for k, v in sorted(data.items(), key=lambda item: medians[item[0]], reverse=True)}

    # Get top 5 items
    data = {k: data[k] for k in list(data.keys())[:13]}

    # Create subplots
    fig, axes = plt.subplots(2, 13, figsize=(30, 6))
    fig.suptitle(main_title, fontsize=16)

    # Create scatterplots and boxplots
    for idx, (ax_scatter, ax_boxplot, (key, value)) in enumerate(zip(axes[0], axes[1], data.items())):
        df = pd.DataFrame(value, columns=["x", "y"])

        # Set background color for subplot
        ax_scatter.set_facecolor((0.8, 0.8, 0.8, 0.3))
        ax_boxplot.set_facecolor((0.8, 0.8, 0.8, 0.3))

        # Adjust marker transparency
        sns.scatterplot(x="x", y="y", data=df, ax=ax_scatter, color=color, edgecolor="black", s=50, alpha=0.7)

        ax_scatter.set_title(key)
        ax_scatter.set_xlim([0.5, 1.1])
        ax_scatter.set_ylim([0, 3])

        if idx == 0:
            ax_scatter.set_ylabel("Importance")  # Set y-label for the first scatter plot
        else:
            ax_scatter.set_ylabel("")  # Remove y-label for the other scatter plots

        # Set common x-label for the first row
        if idx == len(data) - 7:
            ax_scatter.set_xlabel("AUC-ROC")
        else:
            ax_scatter.set_xlabel("")  # Remove x-label for the other plots

        # Create boxplots and adjust transparency
        sns.boxplot(y="y", data=df, orient="h", ax=ax_boxplot, color=color, saturation=0.7, linewidth=1.5)
        ax_boxplot.set_yticks(np.arange(0, 3, 0.75))

        # ax_boxplot.set_xticks([])  # Remove x-ticks for vertical boxplots

        if idx == 0:
            ax_boxplot.set_ylabel("Importance")  # Set y-label for the first boxplot
        else:
            ax_boxplot.set_ylabel("")  # Remove y-label for the other boxplots

    plt.tight_layout()

    return fig
