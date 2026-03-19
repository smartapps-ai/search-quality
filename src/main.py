import asyncio
import logging
from question_prompt import PromptGenerator
from question_generate import AsyncQuestionGenerator
from question_quality_checker import QuestionQualityChecker
from question_mapper import QuestionMapper
from config import (
    BANKING_PERSONA_KPI,
    BANKING_QUESTIONS_PERSONA_KPI_MAPPING,
    BANKING_QUESTIONS_PROMPTS,
    BANKING_SYNTHETIC_QUESTIONS,
    BANKING_MAPPED_QUESTIONS,
    ensure_directories_exist,
    QUALITY_CHECK_REPORT_SYNTHETIC_QUESTIONS,
    QUALITY_CHECK_REPORT_MAPPED_QUESTIONS,
)
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    """Main pipeline for generating and quality checking questions."""
    ensure_directories_exist()
    try:
        # Step 1: Generate prompts
        logger.info("Starting prompt generation...")
        generator = PromptGenerator(
            persona_kpi_path=str(BANKING_PERSONA_KPI),
            golden_data_path=str(BANKING_QUESTIONS_PERSONA_KPI_MAPPING),
            output_path=str(BANKING_QUESTIONS_PROMPTS),
        )
        df_prompts = generator.generate_prompts()
        logger.info(f"Generated {len(df_prompts)} prompts")

        # Step 2: Generate questions asynchronously
        logger.info("Starting question generation...")
        question_generator = AsyncQuestionGenerator(
            max_concurrent=10,
            output_csv=str(BANKING_SYNTHETIC_QUESTIONS),
            model=os.getenv("OPENAI_4_MODEL"),
            input_csv=str(BANKING_QUESTIONS_PROMPTS),
        )
        results = await question_generator.run_async()
        logger.info(f"Generated questions. Shape: {results.shape}")
        logger.info(f"Sample results:\n{results.head()}")

        # Step 3: Run quality checks
        logger.info("Starting quality checks...")
        qqc = QuestionQualityChecker(
            input_csv=str(BANKING_SYNTHETIC_QUESTIONS),
            question_column="synthetic_questions",
            output_path=str(QUALITY_CHECK_REPORT_SYNTHETIC_QUESTIONS),
        )
        report = qqc.run_all_checks()
        logger.info(f" Quality Check Report: {len(report)} checks completed.")

        # Step 4: Map questions
        logger.info("Starting question mapping...")
        mapper = QuestionMapper(
            synthetic_questions_path=str(BANKING_SYNTHETIC_QUESTIONS),
            output_path=str(BANKING_MAPPED_QUESTIONS),
            model=os.getenv("ANTHROPIC_CLAUDE_4_5_SONNET_MODEL"),
            batch_size=5,
        )
        await mapper.map_and_save()
        logger.info("Question mapping completed successfully!")

        # Step 5: Run quality checks on mapped questions
        logger.info("Starting quality checks...")
        qqc = QuestionQualityChecker(
            input_csv=str(BANKING_MAPPED_QUESTIONS),
            question_column="mapped_question",
            output_path=str(QUALITY_CHECK_REPORT_MAPPED_QUESTIONS),
        )
        report = qqc.run_all_checks()
        logger.info(f" Quality Check Report: {len(report)} checks completed.")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
