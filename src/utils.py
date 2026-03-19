"""Utility functions for running evaluations with DeepEval."""

import os
import time
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ConversationalGEval,
    ConversationCompletenessMetric,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn

convo_metrics_l = [
    "Focus [Conversational GEval]",
    "Engagement [Conversational GEval]",
    "Helpful [Conversational GEval]",
    "Voice [Conversational GEval]",
    "Conversation Completeness",
]

all_metrics = convo_metrics_l + ["Answer Relevancy", "Safe"]


# Configuration constants
DEFAULT_BATCH_SIZE = 5
EVALUATION_DELAY = 5


def create_evaluation_dataset_from_dataframe(
    dataframe: pd.DataFrame,
    question_column: str = "synthetic_questions",
    response_column: str = "tursio_response",
) -> EvaluationDataset:
    """
    Create an EvaluationDataset from a DataFrame with test cases.

    Args:
        dataframe: DataFrame with columns question_column, response_column, 'persona', and 'kpi'.

    Returns:
        EvaluationDataset: The created evaluation dataset with test cases.
    """
    dataset = EvaluationDataset()

    # Build test cases from dataframe
    for idx, row in dataframe.iterrows():
        # Skip rows with missing responses
        if pd.isna(row[response_column]):
            continue

        # Create golden reference and test case
        golden = Golden(input=row[question_column])
        dataset.add_golden(golden)

        # Add context metadata
        context = [f"Role: {row['persona']}", f"KPI: {row['kpi']}"]

        dataset.add_test_case(
            LLMTestCase(
                input=row[question_column],
                actual_output=row[response_column],
                retrieval_context=context,
            )
        )

    print(
        f"Added {len(dataset.test_cases)} test cases from "
        f"{len(dataset.goldens)} goldens"
    )
    return dataset


def calculate_success_rates(data, threshold=0.5):
    # data_answer_relevacy = data[data["metric_name"] == "Answer Relevancy"]
    # data_bias = data[data["metric_name"] == "Bias"]

    data_metric = {}
    for metric in all_metrics:
        data_metric[metric] = data[data["metric_name"] == metric]

    convo_success_rates = {}
    for metric in all_metrics:
        if data_metric[metric].empty:
            convo_success_rates[metric] = 0
            continue

        criteria = data_metric[metric]["metric_score"] >= threshold
        convo_success_rates[metric] = criteria.sum() / len(data_metric[metric])

    for metric, rate in convo_success_rates.items():
        print(f"{metric} Success Rate: {rate * 100:.2f}%")
    return convo_success_rates


def plot_success_rates(convo_success_rates, difficulty_level="Medium", experiment="Tursio", path="eval_reports"):
    metrics = list(convo_success_rates.keys())
    rates = list(convo_success_rates.values())

    # sort by rates
    sorted_metrics_rates = sorted(zip(metrics, rates), key=lambda x: x[1])
    metrics, rates = zip(*sorted_metrics_rates)
    plt.figure(figsize=(10, 6))
    plt.barh(metrics, [rate * 100 for rate in rates], color="skyblue")
    plt.xlabel("Success Rate (%)", fontsize=18)
    plt.ylabel("", fontsize=18)
    plt.title(f"Conversational Metric Success Rates - {experiment} - {difficulty_level} Difficulty", fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 100)
    for index, value in enumerate(rates):
        plt.text(value * 100 + 1, index, f"{value * 100:.2f}%", fontsize=15, va="center")
   
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(
        os.path.join(path, f"conversational_metric_success_rates_{experiment.lower()}_{difficulty_level.lower()}.png"), bbox_inches="tight"
    )
    plt.show()

    return metrics, rates


# for all merics plot histogram plot of metric scores for each metric as subplots
def plot_metric_score_histograms(data, difficulty_level="Simple", experiment="Tursio", path="eval_reports", drop_safe=True):
    tmp_metrics = all_metrics.copy()
    if drop_safe:
        data = data[data["metric_name"] != "Safe"]
        tmp_metrics.remove("Safe")

    num_metrics = len(tmp_metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(tmp_metrics):
        metric_data = data[data["metric_name"] == metric]["metric_score"]
        axes[i].hist(metric_data, bins=10, color="skyblue", edgecolor="black")
        axes[i].set_title(f"{metric}", fontsize=18, pad=10)
        axes[i].set_xlabel("Metric Score", fontsize=15)
        axes[i].set_ylabel("Frequency", fontsize=15)
        axes[i].tick_params(axis='both', labelsize=15)
        axes[i].set_xlim(left=-0.1)
        axes[i].set_xlim(right=1.1)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Metric Score Histograms - {experiment} - {difficulty_level} Difficulty", fontsize=20, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(
        os.path.join(path, f"metric_score_histograms_{experiment.lower()}_{difficulty_level.lower()}.png"), bbox_inches="tight"
    )
    plt.show()
    return


# bar plot with hue as difficulty level
def plot_all_difficulty_success_rates(data_all, experiment="Tursio", path="eval_reports"):
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=data_all,
        x="Success Rate",
        y="Metric",
        hue="difficulty_level",
        palette="pastel",
    )
    plt.title(f"Conversational Metric Success Rates by Difficulty Level - {experiment}", fontsize=16)
    plt.xlabel("Success Rate (%)", fontsize=15)
    plt.ylabel("Metric", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(
        fontsize=15,
        title="difficulty_level",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )

    # Get current xlim and set it to leave space for labels
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] * 1.15)  # Add 15% space for labels

    # add each bar value as text (only for bars with width > 1)
    for patch in ax.patches:
        width = patch.get_width()
        if width > 1:  # Only show labels for meaningful bar widths
            ax.text(
                width + 1.5,
                patch.get_y() + patch.get_height() / 2,
                f"{width:.1f}%",
                ha="left",
                va="center",
                fontsize=15,
                # fontweight="bold",
            )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(
        os.path.join(path, f"all_difficulty_success_rates_{experiment.lower()}.png"), bbox_inches="tight"
    )
    plt.show()
    return


def pre_process_data(path_data, path_data_convo):
    # Load data
    data = pd.read_csv(path_data)
    data_convo = pd.read_csv(path_data_convo)

    # Replace "Bias" with "Safe" in metric_name
    data.loc[data["metric_name"] == "Bias", "metric_name"] = "Safe"

    # Invert metric_score for "Safe" metric
    data.loc[data["metric_name"] == "Safe", "metric_score"] = (
        1 - data.loc[data["metric_name"] == "Safe", "metric_score"]
    )

    data_all = pd.concat([data, data_convo], ignore_index=True)
    return data_all
