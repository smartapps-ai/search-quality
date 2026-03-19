"""
Configuration file for all project paths.
This module centralizes path management for the entire project.
"""

from pathlib import Path
from typing import Dict

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent

# ==================== DATA BANK PATHS ====================
DATA_BANK_DIR = PROJECT_ROOT / "data_bank"

# Input data files
BANKING_PERSONA_KPI = DATA_BANK_DIR / "banking_persona_kpi.csv"
BANKING_QUESTIONS_PERSONA_KPI_MAPPING = (
    DATA_BANK_DIR / "banking_questions_persona_kpi_mapping.csv"
)
BANKING_QUESTIONS_PROMPTS = DATA_BANK_DIR / "banking_questions_prompts.csv"

# Generated data files
BANKING_SYNTHETIC_QUESTIONS = DATA_BANK_DIR / "banking_synthetic_questions.csv"
BANKING_SYNTHETIC_QUESTIONS_RES = DATA_BANK_DIR / "banking_synthetic_questions_res.csv"

# Processed data files
BANKING_MAPPED_QUESTIONS = DATA_BANK_DIR / "banking_mapped_questions.csv"

# ==================== DATA PATHS ====================
DATA_DIR = PROJECT_ROOT / "data"
DIFFICULTY_DEFINITION = DATA_DIR / "difficulty_definition.py"
TABLE_SCHEMA = DATA_DIR / "table_schema.py"

# =================== QUALITY CHECK PATHS ====================
QUALITY_CHECK_DIR = PROJECT_ROOT / "quality_checks"


# ==================== BENCHMARK PATHS ====================
BENCHMARK_GPT_DIR = PROJECT_ROOT / "benchmark_gpt"
BENCHMARK_PPL_DIR = PROJECT_ROOT / "benchmark_ppl"

# Benchmark GPT files
BENCHMARK_GPT_SIMPLE_Q_MAPPED = BENCHMARK_GPT_DIR / "simple_q_mapped.csv"
BENCHMARK_GPT_SIMPLE_RESPONSE = BENCHMARK_GPT_DIR / "simple_response.json"
BENCHMARK_GPT_MEDIUM_Q_MAPPED = BENCHMARK_GPT_DIR / "medium_q_mapped.csv"
BENCHMARK_GPT_MEDIUM_RESPONSE = BENCHMARK_GPT_DIR / "medium_response.json"
BENCHMARK_GPT_HARD_Q_MAPPED = BENCHMARK_GPT_DIR / "hard_q_mapped.csv"
BENCHMARK_GPT_HARD_RESPONSE = BENCHMARK_GPT_DIR / "hard_response.json"

# Benchmark PPL files
BENCHMARK_PPL_SIMPLE_Q_MAPPED = BENCHMARK_PPL_DIR / "simple_q_mapped.csv"
BENCHMARK_PPL_SIMPLE_RESPONSE = BENCHMARK_PPL_DIR / "simple_response.json"
BENCHMARK_PPL_MEDIUM_Q_MAPPED = BENCHMARK_PPL_DIR / "medium_q_mapped.csv"
BENCHMARK_PPL_MEDIUM_RESPONSE = BENCHMARK_PPL_DIR / "medium_response.json"
BENCHMARK_PPL_HARD_Q_MAPPED = BENCHMARK_PPL_DIR / "hard_q_mapped.csv"
BENCHMARK_PPL_HARD_RESPONSE = BENCHMARK_PPL_DIR / "hard_response.json"

# ==================== OUTPUT PATHS ====================
OUT_TURSIO_DIR = PROJECT_ROOT / "out_tursio"
OUT_PPL_DIR = PROJECT_ROOT / "out_ppl"
OUT_GPT_DIR = PROJECT_ROOT / "out_gpt"
QUALITY_CHECK_DIR = PROJECT_ROOT / "quality_checks"

QUALITY_CHECK_REPORT_SYNTHETIC_QUESTIONS = (
    QUALITY_CHECK_DIR / "synthetic_questions_quality_report"
)
QUALITY_CHECK_REPORT_MAPPED_QUESTIONS = (
    QUALITY_CHECK_DIR / "mapped_questions_quality_report"
)

# Tursio output files
DEEPEVAL_TURSIO_RESULTS_SIMPLE = OUT_TURSIO_DIR / "deepeval_tursio_results_simple.csv"
DEEPEVAL_TURSIO_RESULTS_SIMPLE_CONVO = (
    OUT_TURSIO_DIR / "deepeval_tursio_results_simple_convo.csv"
)
DEEPEVAL_TURSIO_RESULTS_MEDIUM = OUT_TURSIO_DIR / "deepeval_tursio_results_medium.csv"
DEEPEVAL_TURSIO_RESULTS_MEDIUM_CONVO = (
    OUT_TURSIO_DIR / "deepeval_tursio_results_medium_convo.csv"
)
DEEPEVAL_TURSIO_RESULTS_HARD = OUT_TURSIO_DIR / "deepeval_tursio_results_hard.csv"
DEEPEVAL_TURSIO_RESULTS_HARD_CONVO = (
    OUT_TURSIO_DIR / "deepeval_tursio_results_hard_convo.csv"
)

# PPL output subdirectories
OUT_PPL_SIMPLE = OUT_PPL_DIR / "ppl_simple"
OUT_PPL_SIMPLE_CONVO = OUT_PPL_DIR / "ppl_simple_convo"
OUT_PPL_MEDIUM = OUT_PPL_DIR / "ppl_medium"
OUT_PPL_MEDIUM_CONVO = OUT_PPL_DIR / "ppl_medium_convo"
OUT_PPL_HARD = OUT_PPL_DIR / "ppl_hard"
OUT_PPL_HARD_CONVO = OUT_PPL_DIR / "ppl_hard_convo"

# GPT output subdirectories
OUT_GPT_SIMPLE = OUT_GPT_DIR / "gpt_simple"
OUT_GPT_SIMPLE_CONVO = OUT_GPT_DIR / "gpt_simple_convo"
OUT_GPT_MEDIUM = OUT_GPT_DIR / "gpt_medium"
OUT_GPT_MEDIUM_CONVO = OUT_GPT_DIR / "gpt_medium_convo"
OUT_GPT_HARD = OUT_GPT_DIR / "gpt_hard"
OUT_GPT_HARD_CONVO = OUT_GPT_DIR / "gpt_hard_convo"

# ==================== PROMPT PATHS ====================
PROMPT_DIR = PROJECT_ROOT / "prompt"
BENCHMARK_RESPONSE_PROMPT = PROMPT_DIR / "benchmark_response_prompt.py"
QUESTION_GEN_PROMPT = PROMPT_DIR / "question_gen_prompt.py"
QUESTION_MAPPING_PROMPT = PROMPT_DIR / "question_mapping_prompt.py"

# ==================== IMAGES PATHS ====================
IMAGES_DIR = PROJECT_ROOT / "images"
TURSIO_EVAL_EXCALIDRAW = IMAGES_DIR / "tursio_eval.excalidraw"

# ==================== DIFFICULTY LEVELS ====================
DIFFICULTY_LEVELS = {
    "simple": {
        "data": DATA_BANK_DIR / "simple.csv",
        "tursio_results": DEEPEVAL_TURSIO_RESULTS_SIMPLE,
        "tursio_results_convo": DEEPEVAL_TURSIO_RESULTS_SIMPLE_CONVO,
        "benchmark_ppl_q_mapped": BENCHMARK_PPL_SIMPLE_Q_MAPPED,
        "benchmark_ppl_response": BENCHMARK_PPL_SIMPLE_RESPONSE,
        "benchmark_gpt_q_mapped": BENCHMARK_GPT_SIMPLE_Q_MAPPED,
        "benchmark_gpt_response": BENCHMARK_GPT_SIMPLE_RESPONSE,
        "out_ppl": OUT_PPL_SIMPLE,
        "out_ppl_convo": OUT_PPL_SIMPLE_CONVO,
        "out_gpt": OUT_GPT_SIMPLE,
        "out_gpt_convo": OUT_GPT_SIMPLE_CONVO,
    },
    "medium": {
        "data": DATA_BANK_DIR / "medium.csv",
        "tursio_results": DEEPEVAL_TURSIO_RESULTS_MEDIUM,
        "tursio_results_convo": DEEPEVAL_TURSIO_RESULTS_MEDIUM_CONVO,
        "benchmark_ppl_q_mapped": BENCHMARK_PPL_MEDIUM_Q_MAPPED,
        "benchmark_ppl_response": BENCHMARK_PPL_MEDIUM_RESPONSE,
        "benchmark_gpt_q_mapped": BENCHMARK_GPT_MEDIUM_Q_MAPPED,
        "benchmark_gpt_response": BENCHMARK_GPT_MEDIUM_RESPONSE,
        "out_ppl": OUT_PPL_MEDIUM,
        "out_ppl_convo": OUT_PPL_MEDIUM_CONVO,
        "out_gpt": OUT_GPT_MEDIUM,
        "out_gpt_convo": OUT_GPT_MEDIUM_CONVO,
    },
    "hard": {
        "data": DATA_BANK_DIR / "hard.csv",
        "tursio_results": DEEPEVAL_TURSIO_RESULTS_HARD,
        "tursio_results_convo": DEEPEVAL_TURSIO_RESULTS_HARD_CONVO,
        "benchmark_ppl_q_mapped": BENCHMARK_PPL_HARD_Q_MAPPED,
        "benchmark_ppl_response": BENCHMARK_PPL_HARD_RESPONSE,
        "benchmark_gpt_q_mapped": BENCHMARK_GPT_HARD_Q_MAPPED,
        "benchmark_gpt_response": BENCHMARK_GPT_HARD_RESPONSE,
        "out_ppl": OUT_PPL_HARD,
        "out_ppl_convo": OUT_PPL_HARD_CONVO,
        "out_gpt": OUT_GPT_HARD,
        "out_gpt_convo": OUT_GPT_HARD_CONVO,
    },
}


def get_difficulty_path(difficulty: str, path_type: str) -> Path:
    """
    Get the path for a specific difficulty level and path type.

    Args:
        difficulty: One of 'simple', 'medium', or 'hard'
        path_type: The type of path to retrieve

    Returns:
        The requested Path object

    Raises:
        ValueError: If difficulty or path_type is invalid
    """
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"Invalid difficulty: {difficulty}. Must be one of {list(DIFFICULTY_LEVELS.keys())}"
        )

    if path_type not in DIFFICULTY_LEVELS[difficulty]:
        raise ValueError(
            f"Invalid path_type: {path_type}. Available types: {list(DIFFICULTY_LEVELS[difficulty].keys())}"
        )

    return DIFFICULTY_LEVELS[difficulty][path_type]


def ensure_directories_exist():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_BANK_DIR,
        DATA_DIR,
        BENCHMARK_GPT_DIR,
        BENCHMARK_PPL_DIR,
        OUT_TURSIO_DIR,
        OUT_PPL_DIR,
        OUT_GPT_DIR,
        PROMPT_DIR,
        IMAGES_DIR,
        OUT_PPL_SIMPLE,
        OUT_PPL_SIMPLE_CONVO,
        OUT_PPL_MEDIUM,
        OUT_PPL_MEDIUM_CONVO,
        OUT_PPL_HARD,
        OUT_PPL_HARD_CONVO,
        OUT_GPT_SIMPLE,
        OUT_GPT_SIMPLE_CONVO,
        OUT_GPT_MEDIUM,
        OUT_GPT_MEDIUM_CONVO,
        OUT_GPT_HARD,
        OUT_GPT_HARD_CONVO,
        QUALITY_CHECK_DIR,
        QUALITY_CHECK_REPORT_SYNTHETIC_QUESTIONS,
        QUALITY_CHECK_REPORT_MAPPED_QUESTIONS,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ==================== UTILITY FUNCTIONS ====================
def get_config_dict() -> Dict:
    """
    Return a dictionary of all configuration paths.

    Returns:
        Dictionary containing all path configurations
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "data_bank": str(DATA_BANK_DIR),
        "data": str(DATA_DIR),
        "benchmark_gpt": str(BENCHMARK_GPT_DIR),
        "benchmark_ppl": str(BENCHMARK_PPL_DIR),
        "out_tursio": str(OUT_TURSIO_DIR),
        "out_ppl": str(OUT_PPL_DIR),
        "out_gpt": str(OUT_GPT_DIR),
        "prompt": str(PROMPT_DIR),
        "images": str(IMAGES_DIR),
    }


if __name__ == "__main__":
    # Test the configuration
    print("Project Root:", PROJECT_ROOT)
    print("\nKey Directories:")
    for key, path in get_config_dict().items():
        print(f"  {key}: {path}")

    print("\nDifficulty Levels:")
    for difficulty in DIFFICULTY_LEVELS.keys():
        print(f"  {difficulty}: {list(DIFFICULTY_LEVELS[difficulty].keys())}")

    print("\nCreating directories...")
    ensure_directories_exist()
    print("✓ All directories verified/created")
