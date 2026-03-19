import asyncio
import json
import logging
import time
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from anthropic import Anthropic, RateLimitError

from config import BANKING_SYNTHETIC_QUESTIONS, BANKING_MAPPED_QUESTIONS
from prompt.question_mapping_prompt import DOMAIN, QUESTION_MAPPING_PROMPT

# Configure logger
logger = logging.getLogger(__name__)


class SyntheticQuestion(BaseModel):
    """Model for a synthetic question loaded from CSV."""

    original_question: str = Field(..., alias="original_question")
    persona: str
    kpi: str
    difficulty: str
    candidate_questions: Optional[str] = None

    @field_validator("original_question", mode="before")
    @classmethod
    def validate_original_question(cls, v):
        """Fallback for synthetic_questions column if original_question missing."""
        if not v or (isinstance(v, float)):
            return ""
        return str(v)

    @field_validator("persona", "kpi", "difficulty", mode="before")
    @classmethod
    def validate_strings(cls, v):
        """Validate string fields."""
        return str(v) if v else ""

    class Config:
        """Pydantic config."""

        populate_by_name = True


class MappedQuestion(BaseModel):
    """Model for a mapped question result."""

    original_question: str
    persona: str
    kpi: str
    difficulty: str
    mapped_question: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return self.model_dump(exclude_none=False)


class QuestionMapperConfig(BaseModel):
    """Configuration for QuestionMapper."""

    model: str = "claude-3-5-sonnet-20241022"
    batch_size: int = Field(default=10, ge=1, le=100)
    api_key: Optional[str] = None

    class Config:
        """Pydantic config."""

        pass


class QuestionMapper:
    """Map synthetic questions to relevant KPIs using Claude Sonnet."""

    def __init__(
        self,
        synthetic_questions_path: str = None,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        batch_size: int = 10,
        output_path: str = None,
    ):
        """
        Initialize the QuestionMapper.

        Args:
            synthetic_questions_path: Path to CSV with synthetic questions
            api_key: Anthropic API key (uses env var if None)
            model: Claude model to use
            batch_size: Number of questions to process in parallel
        """
        self.synthetic_questions_path = synthetic_questions_path or str(
            BANKING_SYNTHETIC_QUESTIONS
        )
        self.output_path = output_path or str(BANKING_MAPPED_QUESTIONS)

        # Validate config using Pydantic
        self.config = QuestionMapperConfig(
            api_key=api_key,
            model=model,
            batch_size=batch_size,
        )

        self.df = pd.read_csv(self.synthetic_questions_path)
        logger.info(f"Loaded {len(self.df)} questions")

        self.client = Anthropic(api_key=api_key)
        logger.info(f"Initialized QuestionMapper with model: {self.config.model}")

    def _format_prompt(
        self,
        question: SyntheticQuestion,
    ) -> str:
        """Format the mapping prompt with provided values."""
        prompt = QUESTION_MAPPING_PROMPT.format(
            DOMAIN=DOMAIN,
            original_question=question.original_question,
            persona=question.persona,
            kpi=question.kpi,
            difficulty=question.difficulty,
        )
        return prompt

    async def _call_claude_async(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call Claude API asynchronously with exponential backoff for rate limits.

        Args:
            prompt: The prompt to send to Claude
            max_retries: Maximum number of retries on rate limit errors

        Returns:
            The response text from Claude

        Raises:
            anthropic.RateLimitError: If max retries exceeded
        """
        loop = asyncio.get_event_loop()
        retry_count = 0
        base_wait = 1  # Start with 1 second wait

        while retry_count <= max_retries:
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=self.config.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                return response.content[0].text

            except RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(
                        f"Max retries ({max_retries}) exceeded for rate limit. Giving up."
                    )
                    raise

                # Exponential backoff: 1s, 2s, 4s, 8s
                wait_time = base_wait * (2 ** (retry_count - 1))
                logger.warning(
                    f"Rate limited (429). Retry {retry_count}/{max_retries}. "
                    f"Waiting {wait_time}s before retry..."
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Unexpected error in Claude API call: {e}")
                raise

    def _parse_mapped_question(self, response: str) -> str:
        """
        Parse the mapped question from Claude's response.
        Handles:
        - JSON format (with or without markdown code fences)
        - Empty/null mapped_question values
        - Malformed/truncated responses
        """
        # Clean up markdown code fences if present
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]  # Remove ```json
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]  # Remove ```
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]  # Remove trailing ```
        clean_response = clean_response.strip()

        try:
            # Try to parse as JSON
            data = json.loads(clean_response)

            # Extract mapped_question from the JSON response
            if isinstance(data, dict) and "mapped_question" in data:
                mapped_q = data["mapped_question"]
                # If mapped_question is empty or None, return empty string (question not mappable)
                return mapped_q if mapped_q else ""

            # Fallback: try other field names
            elif isinstance(data, dict) and "question" in data:
                return data["question"] or ""

            # If it's a dict but no expected fields, return empty
            logger.debug(
                f"JSON parsed but no mapped_question field found: {list(data.keys())}"
            )
            return ""

        except json.JSONDecodeError as e:
            # If not valid JSON, try text extraction as fallback
            logger.debug(
                f"Failed to parse JSON: {str(e)[:100]}. Response: {clean_response[:200]}"
            )
            pass

        # If not JSON, try text extraction
        if "mapped_question" in clean_response.lower():
            # Extract content after "mapped_question"
            mapped = clean_response.split("mapped_question")[-1]
            # Clean up and extract just the question text
            mapped = mapped.split("\n")[0].strip()
            # Remove quotes and colons
            mapped = mapped.strip("\"':-").strip()
            return mapped if mapped else ""

        # If we can't parse it, return empty string (unmappable question)
        logger.debug(
            f"Could not extract mapped_question from response: {clean_response[:200]}"
        )
        return ""

        return response.strip()

    async def _map_single_question(self, row: dict) -> MappedQuestion:
        """Map a single question asynchronously."""
        try:
            # Validate input using Pydantic
            question = SyntheticQuestion(
                original_question=row.get(
                    "original_question", row.get("synthetic_questions", "")
                ),
                persona=row.get("persona", ""),
                kpi=row.get("kpi", ""),
                difficulty=row.get("difficulty", ""),
                candidate_questions=row.get("candidate_questions"),
            )

            # Format and call Claude
            prompt = self._format_prompt(question)
            response = await self._call_claude_async(prompt)
            mapped_question = self._parse_mapped_question(response)

            return MappedQuestion(
                original_question=question.original_question,
                persona=question.persona,
                kpi=question.kpi,
                difficulty=question.difficulty,
                mapped_question=mapped_question,
            )
        except Exception as e:
            logger.error(f"Error mapping question: {str(e)}", exc_info=True)
            return MappedQuestion(
                original_question=row.get("original_question", ""),
                persona=row.get("persona", ""),
                kpi=row.get("kpi", ""),
                difficulty=row.get("difficulty", ""),
                mapped_question=None,
                error=str(e),
            )

    async def map_questions(self) -> pd.DataFrame:
        """
        Map synthetic questions to relevant KPIs using Claude Sonnet.
        Includes rate limit handling and delays between batches.

        Returns:
            DataFrame with original and mapped questions
        """
        logger.info(f"Loading synthetic questions from {self.synthetic_questions_path}")

        # Process questions in batches
        mappings: list[MappedQuestion] = []
        total_batches = (
            len(self.df) + self.config.batch_size - 1
        ) // self.config.batch_size

        for batch_idx in range(total_batches):
            # Add delay between batches to avoid rate limiting
            if batch_idx > 0:
                batch_delay = 2  # 2 seconds between batches
                logger.info(f"Waiting {batch_delay}s before next batch...")
                await asyncio.sleep(batch_delay)

            start_idx = batch_idx * self.config.batch_size
            end_idx = min((batch_idx + 1) * self.config.batch_size, len(self.df))
            batch_df = self.df.iloc[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} "
                f"({start_idx + 1}-{end_idx} of {len(self.df)})"
            )

            # Process batch concurrently
            batch_tasks = [
                self._map_single_question(row.to_dict())
                for _, row in batch_df.iterrows()
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle any exceptions from gather
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch task: {result}")
                    # Create error mapping for failed questions
                    mappings.append(
                        MappedQuestion(
                            original_question="",
                            persona="",
                            kpi="",
                            difficulty="",
                            error=str(result),
                        )
                    )
                else:
                    mappings.append(result)

        # Convert mappings to DataFrame
        df_mappings = pd.DataFrame([m.to_dict() for m in mappings])
        logger.info(f"Completed mapping {len(df_mappings)} questions")

        return df_mappings

    async def map_and_save(self) -> pd.DataFrame:
        """
        Map questions and save to CSV.

        Args:
            output_path: Path to save the mapped questions CSV

        Returns:
            DataFrame with mapped questions
        """
        df_mappings = await self.map_questions()
        df_mappings.to_csv(self.output_path, index=False)
        logger.info(f"Saved mapped questions to {self.output_path}")
        return df_mappings
