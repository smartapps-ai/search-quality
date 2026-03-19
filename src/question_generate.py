"""
Generate synthetic banking questions using OpenAI API.

This module generates synthetic questions based on personas, KPIs, and difficulty levels
by querying the OpenAI API and processing the responses. Supports both synchronous and
asynchronous processing for improved performance.
"""

import asyncio
import json
import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from config import BANKING_QUESTIONS_PROMPTS, BANKING_SYNTHETIC_QUESTIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class PromptInput(BaseModel):
    """Input data for question generation."""

    id: int
    persona: str
    kpi: str
    difficulty: str
    prompt: str


class GeneratedResponse(BaseModel):
    """Parsed API response containing generated questions."""

    questions: list[str] = Field(default_factory=list)


class SyntheticQuestion(BaseModel):
    """Output data: original metadata with generated question."""

    id: int
    persona: str
    kpi: str
    difficulty: str
    synthetic_questions: str


class QuestionGenerator:
    """Generate synthetic questions using OpenAI API (synchronous)."""

    def __init__(
        self,
        model: str = os.getenv("OPENAI_4O_MODEL"),
        max_tokens: int = 10000,
        system_message: str = "You are a helpful assistant.",
        input_csv: str = None,
        output_csv: str = None,
    ):
        """
        Initialize the question generator.

        Args:
            model: OpenAI model to use
            max_tokens: Maximum tokens in API response
            system_message: System message for the API
            input_csv: Path to input CSV file
            output_csv: Path to save output CSV file
        """
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.input_csv = input_csv or str(BANKING_QUESTIONS_PROMPTS)
        self.output_csv = output_csv or str(BANKING_SYNTHETIC_QUESTIONS)
        # self.input_csv = input_csv
        self.output_csv = output_csv
        self.input_data: Optional[pd.DataFrame] = None
        self.responses: list[str] = []
        self.parsed_responses: list[Optional[GeneratedResponse]] = []

    def load_data(self) -> pd.DataFrame:
        """Load input data from CSV."""
        logger.info(f"Loading data from {self.input_csv}")
        self.input_data = pd.read_csv(self.input_csv)
        logger.info(f"Loaded {len(self.input_data)} rows")
        return self.input_data

    def _prepare_messages(self, row: pd.Series) -> list[dict[str, str]]:
        """
        Prepare message list for a single row.

        Args:
            row: DataFrame row with prompt

        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": row["prompt"]},
        ]
        return messages

    def generate_responses(self) -> list[str]:
        """
        Generate responses for all input rows using OpenAI API.

        Returns:
            List of API responses
        """
        if self.input_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info(f"Generating responses for {len(self.input_data)} rows")

        for idx, row in self.input_data.iterrows():
            messages = self._prepare_messages(row)

            try:
                logger.info(f"Processing row {idx}/{len(self.input_data)}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )

                content = response.choices[0].message.content
                self.responses.append(content)
                logger.info(f"Response received for row {idx}")

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error at row {idx}: {e}")
                self.responses.append("")
            except Exception as e:
                logger.error(f"API error at row {idx}: {e}")
                self.responses.append("")

        return self.responses

    def parse_responses(self) -> list[Optional[GeneratedResponse]]:
        """
        Parse JSON responses from API.

        Returns:
            List of parsed response objects
        """
        logger.info(f"Parsing {len(self.responses)} responses")

        for idx, resp in enumerate(self.responses):
            try:
                parsed = json.loads(resp)
                self.parsed_responses.append(GeneratedResponse(**parsed))
                logger.info(f"Successfully parsed response {idx}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse response {idx}: {str(e)[:100]}")
                self.parsed_responses.append(None)
            except Exception as e:
                logger.warning(f"Validation error for response {idx}: {e}")
                self.parsed_responses.append(None)

        return self.parsed_responses

    def combine_results(self) -> pd.DataFrame:
        """
        Combine original data with parsed responses.

        Returns:
            DataFrame with original metadata and synthetic questions
        """
        logger.info("Combining original data with responses")

        # Extract questions from parsed responses
        questions = [resp.questions if resp else [] for resp in self.parsed_responses]

        # Create combined dataframe
        result_data = self.input_data.copy()
        result_data.drop(columns=["prompt"], inplace=True)
        result_data["synthetic_questions"] = questions

        return result_data

    def explode_questions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Explode list of questions into separate rows.

        Args:
            data: DataFrame with list columns

        Returns:
            Exploded DataFrame
        """
        logger.info("Exploding questions into separate rows")
        exploded = data.explode("synthetic_questions").reset_index(drop=True)
        logger.info(f"Result has {len(exploded)} rows")
        return exploded

    def save_results(self, data: pd.DataFrame) -> None:
        """
        Save results to CSV file.

        Args:
            data: DataFrame to save
        """
        logger.info(f"Saving results to {self.output_csv}")
        data.to_csv(self.output_csv, index=False)
        logger.info("Results saved successfully")

    def run(self) -> pd.DataFrame:
        """
        Execute the complete pipeline.

        Returns:
            Final DataFrame with synthetic questions
        """
        logger.info("Starting question generation pipeline")

        self.load_data()
        self.generate_responses()
        self.parse_responses()
        combined = self.combine_results()
        exploded = self.explode_questions(combined)
        self.save_results(exploded)

        logger.info("Pipeline completed successfully")
        return exploded


class AsyncQuestionGenerator:
    """Generate synthetic questions using OpenAI API (asynchronous for concurrent requests)."""

    def __init__(
        self,
        model: str = os.getenv("OPENAI_4O_MINI_MODEL"),
        max_tokens: int = 10000,
        system_message: str = "You are a helpful assistant.",
        input_csv: str = "data_bank/banking_questions_prompts.csv",
        output_csv: str = "data_bank/banking_synthetic_questions.csv",
        max_concurrent: int = 5,
    ):
        """
        Initialize the async question generator.

        Args:
            model: OpenAI model to use
            max_tokens: Maximum tokens in API response
            system_message: System message for the API
            input_csv: Path to input CSV file
            output_csv: Path to save output CSV file
            max_concurrent: Maximum concurrent API requests
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.max_concurrent = max_concurrent
        self.input_data: Optional[pd.DataFrame] = None
        self.responses: list[str] = []
        self.parsed_responses: list[Optional[GeneratedResponse]] = []

    def load_data(self) -> pd.DataFrame:
        """Load input data from CSV."""
        logger.info(f"Loading data from {self.input_csv}")
        self.input_data = pd.read_csv(self.input_csv)
        logger.info(f"Loaded {len(self.input_data)} rows")
        return self.input_data

    def _prepare_messages(self, row: pd.Series) -> list[dict[str, str]]:
        """Prepare message list for a single row."""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": row["prompt"]},
        ]
        return messages

    async def _generate_single_response(
        self, idx: int, row: pd.Series
    ) -> tuple[int, str]:
        """
        Generate a single response asynchronously.

        Args:
            idx: Row index
            row: DataFrame row with prompt

        Returns:
            Tuple of (index, response_content)
        """
        messages = self._prepare_messages(row)

        try:
            logger.info(f"Processing row {idx}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            logger.info(f"Response received for row {idx}")
            return idx, content

        except Exception as e:
            logger.error(f"API error at row {idx}: {e}")
            return idx, ""

    async def generate_responses_async(self) -> list[str]:
        """
        Generate responses for all input rows concurrently using OpenAI API.

        Returns:
            List of API responses
        """
        if self.input_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info(
            f"Generating responses for {len(self.input_data)} rows "
            f"with max {self.max_concurrent} concurrent requests"
        )

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_generate(idx: int, row: pd.Series) -> tuple[int, str]:
            async with semaphore:
                return await self._generate_single_response(idx, row)

        # Create tasks for all rows
        tasks = [bounded_generate(idx, row) for idx, row in self.input_data.iterrows()]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort results by index and extract responses
        results_dict = {}
        for result in results:
            if isinstance(result, tuple):
                idx, content = result
                results_dict[idx] = content
            else:
                logger.error(f"Task failed with exception: {result}")

        # Build responses list in order
        self.responses = [results_dict.get(i, "") for i in range(len(self.input_data))]

        return self.responses

    def parse_responses(self) -> list[Optional[GeneratedResponse]]:
        """Parse JSON responses from API."""
        logger.info(f"Parsing {len(self.responses)} responses")

        for idx, resp in enumerate(self.responses):
            try:
                parsed = json.loads(resp)
                self.parsed_responses.append(GeneratedResponse(**parsed))
                logger.info(f"Successfully parsed response {idx}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse response {idx}: {str(e)[:100]}")
                self.parsed_responses.append(None)
            except Exception as e:
                logger.warning(f"Validation error for response {idx}: {e}")
                self.parsed_responses.append(None)

        return self.parsed_responses

    def combine_results(self) -> pd.DataFrame:
        """Combine original data with parsed responses."""
        logger.info("Combining original data with responses")

        questions = [resp.questions if resp else [] for resp in self.parsed_responses]

        result_data = self.input_data.copy()
        result_data.drop(columns=["prompt"], inplace=True)
        result_data["synthetic_questions"] = questions

        return result_data

    def explode_questions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Explode list of questions into separate rows."""
        logger.info("Exploding questions into separate rows")
        exploded = data.explode("synthetic_questions").reset_index(drop=True)
        logger.info(f"Result has {len(exploded)} rows")
        return exploded

    def save_results(self, data: pd.DataFrame) -> None:
        """Save results to CSV file."""
        logger.info(f"Saving results to {self.output_csv}")
        data.to_csv(self.output_csv, index=False)
        logger.info("Results saved successfully")

    async def run_async(self) -> pd.DataFrame:
        """
        Execute the complete async pipeline.

        Returns:
            Final DataFrame with synthetic questions
        """
        logger.info("Starting async question generation pipeline")

        self.load_data()
        await self.generate_responses_async()
        self.parse_responses()
        combined = self.combine_results()
        exploded = self.explode_questions(combined)
        self.save_results(exploded)

        logger.info("Async pipeline completed successfully")
        return exploded


async def run_async_example():
    """Example of using the async generator."""
    generator = AsyncQuestionGenerator(max_concurrent=10)
    results = await generator.run_async()
    print(results.head())


if __name__ == "__main__":
    # Synchronous example
    print("=== Running synchronous pipeline ===")
    generator = QuestionGenerator()
    results = generator.run()
    print(results.head())

    # Uncomment to run async pipeline instead:
    # print("\n=== Running asynchronous pipeline ===")
    # asyncio.run(run_async_example())
