import logging
import mlflow
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from mlflow.tracking import MlflowClient

from src.data_utils.utils import load_params
from src.model.mlflow_utils import (
    fetch_trial_runs_dataframe,
    find_latest_experiment,
    load_model_params_from_experiment,
)
from dotenv import load_dotenv

load_dotenv()

TARGET_METRICS = ('optimisation_score', 'total_profit')

# logging configuration
logger = logging.getLogger('model_analysis')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_analysis_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def _numeric_model_param_columns(
    trial_runs_df: pd.DataFrame,
    model_param_names: list[str],
) -> list[str]:
    numeric_params = []
    for name in model_param_names:
        col = f"params.{name}"
        if col not in trial_runs_df.columns:
            continue
        series = pd.to_numeric(trial_runs_df[col], errors="coerce")
        if series.notna().sum() >= 2:
            numeric_params.append(name)
    return numeric_params


def compute_param_metric_correlations(
    trial_runs_df: pd.DataFrame,
    model_param_names: list[str],
    target_metrics: tuple[str, ...] = TARGET_METRICS,
) -> pd.DataFrame:
    """Correlate numeric model parameters with the requested trial metrics."""
    # numeric_params = _numeric_model_param_columns(trial_runs_df, model_param_names)
    numeric_params = model_param_names
    metric_cols = [f"metrics.{metric}" for metric in target_metrics]
    missing_metrics = [
        metric for metric, col in zip(target_metrics, metric_cols) if col not in trial_runs_df.columns
    ]
    if missing_metrics:
        raise ValueError(f"Missing metric columns in trial runs dataframe: {missing_metrics}")

    param_cols = [f"params.{name}" for name in numeric_params]
    analysis_df = trial_runs_df[param_cols + metric_cols].apply(pd.to_numeric, errors="coerce")
    analysis_df.columns = numeric_params + list(target_metrics)

    correlations_df = pd.DataFrame({"parameter": numeric_params})
    for metric in target_metrics:
        correlations_df[metric] = analysis_df[numeric_params].corrwith(analysis_df[metric]).values

    sort_key = correlations_df[list(target_metrics)].abs().max(axis=1)
    return correlations_df.iloc[sort_key.sort_values(ascending=False).index].reset_index(drop=True)


def plot_param_correlations(
    correlations_df: pd.DataFrame,
    metric: str,
    output_path: str,
) -> None:
    """Save a horizontal bar chart of parameter correlations for one metric."""
    plot_df = correlations_df[["parameter", metric]].dropna().copy()
    plot_df = plot_df.reindex(plot_df[metric].abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(plot_df))))
    colors = plot_df[metric].apply(lambda value: "#2ca02c" if value >= 0 else "#d62728")
    ax.barh(plot_df["parameter"], plot_df[metric], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Model parameter")
    ax.set_title(f"Parameter correlations with {metric}")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_param_histograms(
    trial_runs_df: pd.DataFrame,
    model_param_names: list[str],
    model_params: dict,
    output_dir: str,
) -> list[str]:
    """Save one histogram PNG per model parameter and return the saved file paths."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for name in model_param_names:
        col = f"params.{name}"
        if col not in trial_runs_df.columns:
            logger.warning("Parameter '%s' not found in trial runs dataframe", name)
            continue

        series = trial_runs_df[col].dropna()
        if series.empty:
            continue

        reference_value = model_params.get(name)

        fig, ax = plt.subplots(figsize=(8, 5))
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().all():
            ax.hist(numeric_series, bins="auto", edgecolor="black", alpha=0.7)
            ax.set_xlabel(name)
            reference_numeric = pd.to_numeric(reference_value, errors="coerce")
            if pd.notna(reference_numeric):
                ax.axvline(
                    reference_numeric,
                    color="#d62728",
                    linestyle="--",
                    linewidth=2,
                    label="model_params artifact",
                )
        else:
            counts = series.astype(str).value_counts().sort_index()
            labels = counts.index.astype(str).tolist()
            ax.bar(labels, counts.values, edgecolor="black", alpha=0.7)
            ax.set_xlabel(name)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            if reference_value is not None:
                reference_label = str(reference_value)
                if reference_label in labels:
                    ax.axvline(
                        labels.index(reference_label),
                        color="#d62728",
                        linestyle="--",
                        linewidth=2,
                        label="model_params artifact",
                    )

        if ax.get_legend_handles_labels()[0]:
            ax.legend()

        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {name}")
        fig.tight_layout()

        output_path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def main():
    print("Starting model analysis process...")
    # Get root directory and resolve the path for params.yaml
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

    # Load parameters from the root directory
    params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)

    target_metrics = params['model_analysis']['target_metrics']
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()
    model_params = load_model_params_from_experiment(client, json.loads(os.getenv('BUILDING_EXPERIMENT_TAGS')), logger=logger)
    print(f"Loaded model params: {model_params}")

    experiment = find_latest_experiment(client, json.loads(os.getenv('BUILDING_EXPERIMENT_TAGS')))
    experiment_id = experiment.experiment_id

    print(f"Experiment ID: {experiment_id}")

    trial_runs_df = fetch_trial_runs_dataframe(experiment_id)
    print(f"Fetched {len(trial_runs_df)} Trial_* runs (excluding nested runs)")

    correlations_df = compute_param_metric_correlations(
        trial_runs_df,
        list(model_params.keys()),
        target_metrics,
    )
    print(correlations_df.head())

    analysis_dir = os.path.join(root_dir, "data", "models", "analysis")
    correlations_path = os.path.join(analysis_dir, "param_metric_correlations.csv")
    os.makedirs(analysis_dir, exist_ok=True)
    correlations_df.to_csv(correlations_path, index=False)
    print(f"Saved correlations to {correlations_path}")

    for metric in target_metrics:
        plot_path = os.path.join(analysis_dir, f"param_correlations_{metric}.png")
        plot_param_correlations(correlations_df, metric, plot_path)
        print(f"Saved correlation plot to {plot_path}")


    histograms_dir = os.path.join(analysis_dir, "histograms")
    histogram_paths = save_model_param_histograms(
        trial_runs_df,
        list(model_params.keys()),
        model_params,
        histograms_dir,
    )
    print(f"Saved {len(histogram_paths)} parameter histograms to {histograms_dir}")

    best_run_name = experiment.tags.get("best_run_name")
    if not best_run_name:
        raise ValueError(
            f"Experiment '{experiment.name}' (id={experiment.experiment_id}) "
            "is missing the 'best_run_name' tag"
        )

    run_id = experiment.tags.get("best_run_id")
    if not run_id:
        print(f"Searching for run_id with run name: {best_run_name}")
        run_object = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"run_name = '{best_run_name}'",
        )
        if run_object.empty:
            raise ValueError(
                f"No run named '{best_run_name}' found in experiment "
                f"'{experiment.name}' (id={experiment_id})"
            )
        run_id = run_object.iloc[0]["run_id"]

    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifact(local_path=correlations_path, artifact_path='correlations')
        for metric in target_metrics:
            plot_path = os.path.join(analysis_dir, f"param_correlations_{metric}.png")
            mlflow.log_artifact(local_path=plot_path, artifact_path='correlations')
        mlflow.log_artifacts(local_dir=histograms_dir, artifact_path='histograms')
        print(f"Logged {len(histogram_paths)} histograms to MLflow artifact path 'histograms'")




if __name__ == '__main__':
    main()
