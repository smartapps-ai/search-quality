"""Response evaluation module with Pydantic models and class-based architecture."""

import logging
import os
import time
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ConversationalGEval,
    ConversationCompletenessMetric,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn

# Configuration constants
DEFAULT_BATCH_SIZE = 5
EVALUATION_DELAY = 5

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================


class MetricData(BaseModel):
    """Pydantic model for metric evaluation data."""

    name: str
    score: float
    threshold: float
    success: bool
    reason: str


class TestResultData(BaseModel):
    """Pydantic model for test result data."""

    name: str
    input: Optional[str] = None
    actual_output: str
    expected_output: Optional[str] = None
    test_success: bool
    metrics: List[MetricData] = Field(default_factory=list)


class ConversationalTestResultData(BaseModel):
    """Pydantic model for conversational test result data."""

    name: str
    turns: List = Field(default_factory=list)
    test_success: bool
    metrics: List[MetricData] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs."""

    batch_size: int = DEFAULT_BATCH_SIZE
    output_dir: str
    eval_type: str
    question_column: str = "synthetic_questions"
    response_column: str = "tursio_response"
    run_async: bool = False
    evaluation_delay: float = EVALUATION_DELAY


# ============================================================================
# Response Evaluator Classes
# ============================================================================


class ResponseMetrics:
    """Handles evaluation of test cases using DeepEval metrics."""

    def __init__(self, config: EvaluationConfig):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.async_config = AsyncConfig(run_async=config.run_async)
        self.metrics = [AnswerRelevancyMetric(), BiasMetric()]

    def create_dataset_from_dataframe(
        self, dataframe: pd.DataFrame
    ) -> EvaluationDataset:
        """Create an evaluation dataset from a pandas DataFrame."""
        dataset = EvaluationDataset()
        for idx, row in dataframe.iterrows():
            test_case = LLMTestCase(
                input=row[self.config.question_column],
                actual_output=row[self.config.response_column],
            )
            dataset.add_test_case(test_case)

        logger.info(f"Created dataset with {len(dataset.test_cases)} test cases")
        return dataset

    def evaluate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Run evaluation on the given dataset and return results as DataFrame.

        Args:
            dataframe: DataFrame with test cases to evaluate.

        Returns:
            DataFrame with evaluation results.
        """
        dataset = self.create_dataset_from_dataframe(dataframe)
        start_time = time.time()
        all_results = []

        # Evaluate in batches
        for step in range(0, len(dataset.test_cases), self.config.batch_size):
            batch_end = min(step + self.config.batch_size, len(dataset.test_cases))
            logger.info(f"Evaluating test cases {step} to {batch_end}")

            test_batch = dataset.test_cases[step:batch_end]

            try:
                results = evaluate(
                    test_cases=test_batch,
                    metrics=self.metrics,
                    async_config=self.async_config,
                )
                all_results.extend(results.test_results)
            except Exception as e:
                logger.error(f"Error evaluating batch at {step}: {e}")
                continue

            # Avoid rate limiting
            time.sleep(self.config.evaluation_delay)

        duration = time.time() - start_time
        logger.info(f"Evaluation completed in {duration:.2f} seconds")

        # Convert results and save
        results_df = self._convert_to_dataframe(all_results)
        self._save_results(results_df)

        return results_df

    def _convert_to_dataframe(self, all_results: List) -> pd.DataFrame:
        """Convert evaluation results to a pandas DataFrame."""
        data_for_df = []

        for test_result in all_results:
            for metric_data in test_result.metrics_data:
                row = {
                    "name": test_result.name,
                    "input": test_result.input,
                    "actual_output": test_result.actual_output,
                    "expected_output": test_result.expected_output,
                    "test_success": test_result.success,
                    "metric_name": metric_data.name,
                    "metric_score": metric_data.score,
                    "metric_threshold": metric_data.threshold,
                    "metric_success": metric_data.success,
                    "metric_reason": metric_data.reason,
                }
                data_for_df.append(row)

        return pd.DataFrame(data_for_df)

    def _save_results(self, results_df: pd.DataFrame) -> None:
        """Save evaluation results to CSV."""
        output_dir = os.path.join(self.config.output_dir, self.config.eval_type)
        os.makedirs(output_dir, exist_ok=True)

        results_csv_path = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Results saved to {results_csv_path}")


class ConversationalResponseMetrics:
    """Handles evaluation of conversational test cases."""

    def __init__(self, config: EvaluationConfig):
        """Initialize the conversational evaluator with configuration."""
        self.config = config
        self.async_config = AsyncConfig(run_async=config.run_async)
        self.metrics = self._setup_metrics()

    @staticmethod
    def _setup_metrics() -> List:
        """Setup conversational evaluation metrics."""
        return [
            ConversationalGEval(
                name="Focus",
                criteria="Does the response directly address the specific question or task?",
            ),
            ConversationalGEval(
                name="Helpful",
                criteria="Does the response meaningfully help the user?",
            ),
            ConversationalGEval(
                name="Voice",
                criteria="Does the response use clear, active voice?",
            ),
            ConversationalGEval(
                name="Engagement",
                criteria="Does the response use appropriate and engaging language?",
            ),
            ConversationCompletenessMetric(),
        ]

    def create_test_cases(
        self, dataframe: pd.DataFrame
    ) -> List[ConversationalTestCase]:
        """Create conversational test cases from a DataFrame."""
        test_cases = []

        for idx, row in dataframe.iterrows():
            turn_user = Turn(
                role="user",
                retrieval_context=[
                    f"user persona: {row.get('persona', 'N/A')}",
                    f"KPI: {row.get('kpi', 'N/A')}",
                ],
                content=row[self.config.question_column],
            )
            turn_assistant = Turn(
                role="assistant", content=row[self.config.response_column]
            )
            convo_test_case = ConversationalTestCase(turns=[turn_user, turn_assistant])
            test_cases.append(convo_test_case)

        logger.info(f"Created {len(test_cases)} conversational test cases")
        return test_cases

    def evaluate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Run evaluation on conversational test cases.

        Args:
            dataframe: DataFrame with conversational test data.

        Returns:
            DataFrame with evaluation results.
        """
        test_cases = self.create_test_cases(dataframe)
        start_time = time.time()
        all_results = []

        # Evaluate in batches
        for step in range(0, len(test_cases), self.config.batch_size):
            batch_end = min(step + self.config.batch_size, len(test_cases))
            logger.info(f"Evaluating conversational test cases {step} to {batch_end}")

            test_batch = test_cases[step:batch_end]

            try:
                results = evaluate(
                    test_cases=test_batch,
                    metrics=self.metrics,
                    async_config=self.async_config,
                )
                all_results.extend(results.test_results)
            except Exception as e:
                logger.error(f"Error evaluating batch at {step}: {e}")
                continue

            # Avoid rate limiting
            time.sleep(self.config.evaluation_delay)

        duration = time.time() - start_time
        logger.info(f"Evaluation completed in {duration:.2f} seconds")

        # Convert results and save
        results_df = self._convert_to_dataframe(all_results)
        self._save_results(results_df)

        return results_df

    def _convert_to_dataframe(self, all_results: List) -> pd.DataFrame:
        """Convert conversational evaluation results to a pandas DataFrame."""
        data_for_df = []

        for test_result in all_results:
            for metric_data in test_result.metrics_data:
                row = {
                    "name": test_result.name,
                    "turns": test_result.turns,
                    "test_success": test_result.success,
                    "metric_name": metric_data.name,
                    "metric_score": metric_data.score,
                    "metric_threshold": metric_data.threshold,
                    "metric_success": metric_data.success,
                    "metric_reason": metric_data.reason,
                }
                data_for_df.append(row)

        return pd.DataFrame(data_for_df)

    def _save_results(self, results_df: pd.DataFrame) -> None:
        """Save evaluation results to CSV."""
        output_dir = os.path.join(self.config.output_dir, self.config.eval_type)
        os.makedirs(output_dir, exist_ok=True)

        results_csv_path = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Results saved to {results_csv_path}")
