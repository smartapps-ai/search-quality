"""Response evaluation module with Pydantic models and class-based architecture."""

import json
import logging
import pandas as pd

from config import get_difficulty_path, BENCHMARK_PPL_DIR, BENCHMARK_GPT_DIR
from response_metrics import (
    EvaluationConfig,
    ResponseMetrics,
    ConversationalResponseMetrics,
)

# Configuration constants
DEFAULT_BATCH_SIZE = 5
EVALUATION_DELAY = 5

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Processor
# ============================================================================


class DataProcessor:
    """Handles loading and preparing evaluation data from CSV and JSON files."""

    @staticmethod
    def load_data_from_csv(csv_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        return pd.read_csv(csv_path)

    @staticmethod
    def load_data_from_json(json_path: str) -> dict:
        """Load data from a JSON file."""
        with open(json_path, "r") as f:
            return json.load(f)

    @staticmethod
    def merge_questions_and_responses(
        questions_df: pd.DataFrame,
        responses: dict,
        question_col: str = "answerable_question",
    ) -> pd.DataFrame:
        """Merge questions DataFrame with responses from JSON."""
        responses_df = pd.DataFrame.from_records(responses["answers"])
        merged_df = pd.merge(
            questions_df, responses_df, left_on=question_col, right_on="question"
        ).drop(columns=["question"])
        return merged_df


# ============================================================================
# Response Evaluator
# ============================================================================


class ResponseEvaluator:
    """Orchestrates response evaluation with support for standard and conversational test cases."""

    def __init__(self, data_dir: str = "benchmark_ppl", output_dir: str = "out_ppl"):
        """
        Initialize the evaluator.

        Args:
            data_dir: Directory containing benchmark data files.
            output_dir: Directory for saving evaluation results.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processor = DataProcessor()

    def load_benchmark_data(self, difficulty: str) -> pd.DataFrame:
        """
        Load benchmark data for a specific difficulty level.

        Args:
            difficulty: One of 'simple', 'medium', or 'hard'.

        Returns:
            Merged DataFrame with questions and responses.
        """
        # Map data_dir to benchmark type
        if "ppl" in self.data_dir:
            self.benchmark_app = "ppl"
        elif "gpt" in self.data_dir:
            self.benchmark_app = "gpt"
        else:
            raise ValueError(f"Unknown data_dir: {self.data_dir}")
        
        q_key = f"benchmark_{self.benchmark_app}_q_mapped"
        response_key = f"benchmark_{self.benchmark_app}_response"
    
        questions_df = self.processor.load_data_from_csv(
            str(get_difficulty_path(difficulty, q_key))
        )
        responses = self.processor.load_data_from_json(
            str(get_difficulty_path(difficulty, response_key))
        )
        merged_df = self.processor.merge_questions_and_responses(
            questions_df, responses, question_col="answerable_question"
        )
        logger.info(
            f"Loaded {len(merged_df)} benchmark records for {difficulty} difficulty"
        )
        return merged_df

    def run_evaluation(
        self,
        dataframe: pd.DataFrame,
        batch_size: int = DEFAULT_BATCH_SIZE,
        question_column: str = "answerable_question",
        response_column: str = "answer",
        eval_type: str = "ppl_simple",
    ) -> pd.DataFrame:
        """
        Run standard evaluation on test cases.

        Args:
            dataframe: DataFrame with test cases to evaluate.
            batch_size: Number of test cases per batch.
            question_column: Name of the question column.
            response_column: Name of the response column.
            eval_type: Type of evaluation for output directory.

        Returns:
            DataFrame with evaluation results.
        """
        config = EvaluationConfig(
            batch_size=batch_size,
            output_dir=self.output_dir,
            eval_type=eval_type,
            question_column=question_column,
            response_column=response_column,
            evaluation_delay=EVALUATION_DELAY,
        )
        evaluator = ResponseMetrics(config)
        return evaluator.evaluate(dataframe)

    def run_evaluation_convo(
        self,
        dataframe: pd.DataFrame,
        batch_size: int = DEFAULT_BATCH_SIZE,
        question_column: str = "answerable_question",
        response_column: str = "answer",
        eval_type: str = "ppl_simple_convo",
    ) -> pd.DataFrame:
        """
        Run conversational evaluation on test cases.

        Args:
            dataframe: DataFrame with conversational test cases.
            batch_size: Number of test cases per batch.
            question_column: Name of the question column.
            response_column: Name of the response column.
            eval_type: Type of evaluation for output directory.

        Returns:
            DataFrame with evaluation results.
        """
        config = EvaluationConfig(
            batch_size=batch_size,
            output_dir=self.output_dir,
            eval_type=eval_type,
            question_column=question_column,
            response_column=response_column,
            evaluation_delay=EVALUATION_DELAY,
        )
        evaluator = ConversationalResponseMetrics(config)
        return evaluator.evaluate(dataframe)

    def run_simple_evaluation(self, batch_size: int = 2) -> None:
        """Run evaluations on simple difficulty benchmark data."""
        logger.info("Running evaluations for simple difficulty")
        data = self.load_benchmark_data("simple")

        self.run_evaluation(
            dataframe=data,
            batch_size=batch_size,
            question_column="answerable_question",
            response_column="answer",
            eval_type=f"{self.benchmark_app}_simple",
        )

        self.run_evaluation_convo(
            dataframe=data,
            batch_size=batch_size,
            question_column="answerable_question",
            response_column="answer",
            eval_type=f"{self.benchmark_app}_simple_convo",
        )

    def run_medium_evaluation(self, batch_size: int = 2) -> None:
        """Run evaluations on medium difficulty benchmark data."""
        logger.info("Running evaluations for medium difficulty")
        data = self.load_benchmark_data("medium")

        self.run_evaluation(
            dataframe=data,
            batch_size=batch_size,
            question_column="answerable_question",
            response_column="answer",
            eval_type=f"{self.benchmark_app}_medium",
        )

        self.run_evaluation_convo(
            dataframe=data,
            batch_size=batch_size,
            question_column="answerable_question",
            response_column="answer",
            eval_type=f"{self.benchmark_app}_medium_convo",
        )

    def run_hard_evaluation(self, batch_size: int = 2) -> None:
        """Run evaluations on hard difficulty benchmark data."""
        logger.info("Running evaluations for hard difficulty")
        data = self.load_benchmark_data("hard")

        self.run_evaluation(
            dataframe=data,
            batch_size=batch_size,
            question_column="answerable_question",
            response_column="answer",
            eval_type=f"{self.benchmark_app}_hard",
        )

        self.run_evaluation_convo(
            dataframe=data,
            batch_size=batch_size,
            question_column="answerable_question",
            response_column="answer",
            eval_type=f"{self.benchmark_app}_hard_convo",
        )

    def run_all_evaluations(self, batch_size: int = 2) -> None:
        """Run all evaluations across all difficulty levels."""
        self.run_simple_evaluation(batch_size)
        self.run_medium_evaluation(batch_size)
        self.run_hard_evaluation(batch_size)
