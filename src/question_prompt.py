"""
Generate prompts for question creation with difficulty levels and personas.

This module creates prompts for generating banking questions with different
complexity levels and personas by combining data mappings with golden examples.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BANKING_QUESTIONS_PERSONA_KPI_MAPPING,
    BANKING_PERSONA_KPI,
    BANKING_QUESTIONS_PROMPTS,
)
from data.difficulty_definition import DIFFICULTY_DEFINITION
from data.table_schema import TABLE_JOINS
from prompt.question_gen_prompt import QUESTION_GEN_PROMPT, DOMAIN

# Configure logger
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PromptGenerator:
    """Generate prompts for all persona-kpi-difficulty combinations."""

    def __init__(
        self,
        persona_kpi_path: str = None,
        golden_data_path: str = None,
        output_path: str = None,
    ):
        """Initialize the PromptGenerator by loading data mappings and golden examples."""
        self.persona_kpi_path = persona_kpi_path or str(BANKING_PERSONA_KPI)
        self.golden_data_path = golden_data_path or str(
            BANKING_QUESTIONS_PERSONA_KPI_MAPPING
        )
        self.output_path = output_path or str(BANKING_QUESTIONS_PROMPTS)
        self.data_mapping, self.data_golden = self._load_data()

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data mapping and golden examples."""
        logger.info(
            f"Loading data mapping and golden examples from: {self.persona_kpi_path} and {self.golden_data_path}"
        )
        data_mapping = (
            pd.read_csv(self.persona_kpi_path)[["persona", "kpi"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        data_golden = pd.read_csv(self.golden_data_path)

        logger.info(
            f"Data mapping shape: {data_mapping.shape}, Data golden shape: {data_golden.shape}"
        )

        return data_mapping, data_golden

    @staticmethod
    def _get_examples_text(examples: list[str], max_examples: int = 5) -> str:
        """Format examples as text for prompt injection."""
        return "\n- ".join(examples[:max_examples])

    @staticmethod
    def _format_prompt(
        template: str,
        persona: str,
        kpi: str,
        difficulty: str,
        examples_text: str,
        n_questions: int,
    ) -> str:
        """Format the prompt template with provided values."""
        replacements = {
            "{row['persona']}": persona,
            "{domain}": DOMAIN,
            "{row['kpi']}": kpi,
            "{row['difficulty']}": difficulty,
            "{difficulty_meaning}": DIFFICULTY_DEFINITION.get(difficulty, ""),
            "{TABLE_JOINS}": TABLE_JOINS,
            "{examples_text}": examples_text,
            "{n_questions}": str(n_questions),
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _make_prompt(self, row: pd.Series, n_questions: int = 10) -> str:
        """
        Create a prompt for a given row with difficulty and persona.

        Args:
            row: DataFrame row containing 'difficulty' and 'persona'
            n_questions: Number of questions to generate

        Returns:
            Formatted prompt string
        """
        # Fetch examples matching difficulty and persona
        examples = self.data_golden[
            (self.data_golden["complexity"] == row["difficulty"])
            & (self.data_golden["persona"] == row["persona"])
        ]["question"].tolist()

        # print(
        #     f"Persona: {row['persona']}, Difficulty: {row['difficulty']}, "
        #     f"Examples found: {len(examples)}"
        # )

        examples_text = self._get_examples_text(examples)

        return self._format_prompt(
            QUESTION_GEN_PROMPT,
            persona=row["persona"],
            kpi=row["kpi"],
            difficulty=row["difficulty"],
            examples_text=examples_text,
            n_questions=n_questions,
        )

    def generate_prompts(
        self,
    ) -> pd.DataFrame:
        """
        Generate prompts for all persona-kpi-difficulty combinations.

        Args:
            output_path: Path to save the generated prompts CSV

        Returns:
            DataFrame with generated prompts
        """
        # Generate prompts for each difficulty level
        df_prompt_list = []
        for difficulty in DIFFICULTY_DEFINITION.keys():
            df_tmp = self.data_mapping.copy()
            df_tmp["difficulty"] = difficulty
            df_tmp["prompt"] = df_tmp.apply(lambda row: self._make_prompt(row), axis=1)
            df_prompt_list.append(df_tmp)

        # Combine all prompts
        df_prompt_all = pd.concat(df_prompt_list, ignore_index=True)
        df_prompt_all = df_prompt_all.reset_index().rename(columns={"index": "id"})

        # Save to CSV
        df_prompt_all.to_csv(self.output_path, index=False)
        logger.info(
            f"Prompts saved to {self.output_path} of size {df_prompt_all.shape}"
        )

        return df_prompt_all


if __name__ == "__main__":
    generator = PromptGenerator()
    df_prompts = generator.generate_prompts()
    print(df_prompts.head())
