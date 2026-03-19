# Evaluating Structured Data Search

Business users need to search enterprise databases using natural language, just as they now search the web using ChatGPT or Perplexity. However, existing benchmarks are designed for open-domain QA or text-to-SQL and they do not evaluate the end-to-end quality of such a search experience. This project focuses on an evaluation framework for structured database search that generates realistic banking queries across varying difficulty levels and assesses answer quality using relevance, safety, and conversational metrics via an LLM-as-judge approach.

Below, we describe the components for generating, mapping, and evaluating synthetic banking questions, including quality metrics and response evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Data Flow](#data-flow)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)

---

## Overview

This is an end-to-end pipeline for:
1. **Generating synthetic banking questions** based on personas, KPIs, and difficulty levels
2. **Mapping questions** to relevant KPIs using Claude AI
3. **Validating question quality** with duplicate detection, similarity analysis, and lexical diversity metrics
4. **Evaluating responses** using DeepEval metrics (relevancy, bias, conversational quality)

The system uses:
- **OpenAI API** for question generation
- **Anthropic Claude API** for semantic question mapping
- **DeepEval** for response quality metrics
- **Scikit-learn** for similarity analysis

---

## Project Structure

```
tursio/
├── src/                          # Main source code
│   ├── config.py                 # Centralized path & configuration management
│   ├── main.py                   # Main pipeline orchestrator
│   ├── question_prompt.py         # Prompt generation for questions
│   ├── question_generate.py       # OpenAI-based question generation
│   ├── question_mapper.py         # Claude-based KPI mapping
│   ├── question_quality_checker.py # Quality validation metrics
│   ├── response_eval.py           # Response evaluation orchestrator
│   ├── response_metrics.py        # DeepEval metrics computation
│   ├── utils.py                   # Utility functions for evaluations
│   └── test_format.ipynb          # Interactive testing notebook
├── data/                          # Data definitions
│   ├── difficulty_definition.py   # Question difficulty levels
│   └── table_schema.py            # Database schema definitions
├── data_bank/                     # Input & processed data
│   ├── banking_persona_kpi.csv
│   ├── banking_questions_persona_kpi_mapping.csv
│   ├── banking_questions_prompts.csv
│   ├── banking_synthetic_questions.csv
│   └── banking_mapped_questions.csv
├── prompt/                        # Prompt templates
│   ├── question_gen_prompt.py
│   ├── question_mapping_prompt.py
│   └── question_prompt.py
├── benchmark_gpt/                 # GPT benchmark responses
├── benchmark_ppl/                 # PPL benchmark responses
├── out_gpt/                        # GPT evaluation results
├── out_ppl/                        # PPL evaluation results
├── out_tursio/                     # Tursio evaluation results
├── quality_checks/                # Quality check reports
├── requirements.txt               # Python dependencies
└── create_env.sh                  # Environment setup script
```

---

## Module Documentation

### 🔧 **config.py**
**Purpose:** Centralized configuration and path management

**Key Features:**
- Dynamic path resolution using `pathlib.Path`
- Automatic directory creation with `ensure_directories_exist()`
- Configuration dictionaries for easy path lookups
- Supports difficulty-based path organization (simple/medium/hard)

**Key Functions:**
```python
ensure_directories_exist()      # Create all required directories
get_difficulty_path(difficulty, path_type)  # Get path by difficulty level
get_config_dict()               # Get config as dictionary
```

**Key Constants:**
- `PROJECT_ROOT` - Project base directory
- `DATA_BANK_DIR` - Input data directory
- `QUALITY_CHECK_DIR` - Quality reports output
- `BENCHMARK_GPT_DIR`, `BENCHMARK_PPL_DIR` - Benchmark data
- `OUT_TURSIO_DIR`, `OUT_GPT_DIR`, `OUT_PPL_DIR` - Evaluation results

---

### 📝 **question_prompt.py**
**Purpose:** Generate prompts for question creation with personas, KPIs, and difficulty levels

**Key Classes:**
- `PromptGenerator` - Creates prompts from persona-KPI combinations

**Key Methods:**
```python
generate_prompts()              # Generate all prompts for combinations
get_prompt_for_combination()    # Get specific prompt by persona/kpi/difficulty
```

**Inputs:** 
- `banking_persona_kpi.csv` (personas and KPIs)
- `banking_questions_persona_kpi_mapping.csv` (mapping data)
- Golden examples from question templates

**Outputs:**
- `banking_questions_prompts.csv` (prompts ready for generation)

---

### 🤖 **question_generate.py**
**Purpose:** Generate synthetic banking questions using OpenAI API

**Key Classes:**
- `PromptInput` (Pydantic) - Input data model
- `GeneratedResponse` (Pydantic) - API response model
- `SyntheticQuestion` (Pydantic) - Generated question data
- `AsyncQuestionGenerator` - Main generation orchestrator

**Key Methods:**
```python
run_async()                     # Async generation with concurrency control
_call_openai_async()           # Single API call with error handling
_parse_response()              # Parse and validate API responses
```

**Features:**
- Asynchronous processing with configurable concurrency
- Batch processing with inter-request delays
- Error recovery and graceful degradation
- Support for multiple OpenAI models

**Inputs:**
- `banking_questions_prompts.csv` (prompts from question_prompt.py)

**Outputs:**
- `banking_synthetic_questions.csv` (generated questions)

---

### 🗺️ **question_mapper.py**
**Purpose:** Map synthetic questions to relevant KPIs using Claude AI

**Key Classes:**
- `SyntheticQuestion` (Pydantic) - Input question model
- `MappedQuestion` (Pydantic) - Mapped output model
- `QuestionMapper` - Main mapping orchestrator

**Key Methods:**
```python
map_and_save()                  # Map questions and save results
_call_claude_async()            # Claude API call with exponential backoff
_parse_mapped_question()        # Parse Claude response with markdown handling
```

**Features:**
- **Exponential Backoff Retry Logic**: Handles 429 rate limit errors
  - Retries: 1s, 2s, 4s, 8s delays
  - Max 3 attempts before failure
- Batch processing with 2-second inter-batch delays
- Markdown code fence parsing for JSON responses
- Graceful handling of unmappable questions (empty strings)

**Inputs:**
- `banking_synthetic_questions.csv` (questions to map)

**Outputs:**
- `banking_mapped_questions.csv` (mapped results with original_question, mapped_question, persona, kpi, difficulty)

---

### ✅ **question_quality_checker.py**
**Purpose:** Validate question quality with multiple metrics

**Key Classes:**
- `QuestionQualityChecker` - Quality validation orchestrator

**Key Methods:**
```python
run_all_checks()                # Execute all quality checks
find_duplicates()               # Identify duplicate questions
find_similar_questions()        # Find semantically similar questions (TF-IDF)
check_length_outliers()         # Identify unusually short/long questions
check_lexical_diversity()       # Analyze vocabulary variety
```

**Metrics:**
- **Duplicates**: Exact text matches
- **Similarity**: Cosine similarity with TfidfVectorizer (threshold: 0.8)
- **Length Outliers**: Questions outside IQR bounds
- **Lexical Diversity**: Unique word ratio analysis

**Features:**
- Automatic NaN/empty value filtering before analysis
- Graceful handling of small datasets
- Comprehensive reporting with visualizations

**Inputs:**
- CSV files with questions (any column name)

**Outputs:**
- Quality report dictionary with metrics summary

---

### 📊 **response_eval.py**
**Purpose:** Orchestrate evaluation of LLM responses against benchmark data

**Key Classes:**
- `DataProcessor` - Load and merge data from CSV/JSON
- `ResponseEvaluator` - Main evaluation orchestrator
- `EvaluationResult` (Pydantic) - Result data model

**Key Methods:**
```python
run_all_evaluations()           # Execute complete evaluation pipeline
evaluate_simple_responses()     # Simple QA evaluation
evaluate_conversational_responses()  # Multi-turn conversation evaluation
```

**Metrics Evaluated:**
- Answer Relevancy
- Bias Detection
- Conversation Completeness
- Conversational GEval (Focus, Engagement, Helpfulness, Voice)

**Features:**
- Supports both simple QA and conversational evaluation
- Auto-detection of benchmark type (GPT/PPL)
- Batch processing with configurable delays
- Results saved to structured formats (CSV, JSON)

**Inputs:**
- Benchmark data: `*_q_mapped.csv`, `*_response.json`

**Outputs:**
- Evaluation results by difficulty level and metric

---

### 📈 **response_metrics.py**
**Purpose:** Define evaluation metrics and result data models using Pydantic and DeepEval

**Key Classes:**
- `MetricData` (Pydantic) - Individual metric result
- `TestResultData` (Pydantic) - Single test case result
- `EvaluationConfig` (Pydantic) - Evaluation configuration
- `ResponseMetrics` - Simple QA metrics computation
- `ConversationalResponseMetrics` - Multi-turn metrics computation

**Key Methods:**
```python
compute_metrics()               # Calculate all metrics for responses
aggregate_results()             # Aggregate metrics across test cases
generate_report()               # Create evaluation report
```

**Features:**
- Pydantic validation for all data models
- Integration with DeepEval library
- Support for custom metric thresholds
- Comprehensive error handling

---

### 🛠️ **utils.py**
**Purpose:** Utility functions for evaluations and visualizations

**Key Functions:**
```python
create_evaluation_dataset_from_dataframe()  # Create DeepEval dataset from DataFrame
run_evaluations()               # Execute DeepEval evaluation pipeline
create_visualizations()         # Generate metric visualizations
```

**Features:**
- DeepEval dataset creation
- Async evaluation execution
- Visualization generation with Seaborn/Matplotlib
- Metrics aggregation and reporting

---

### 🚀 **main.py**
**Purpose:** Orchestrate the complete pipeline from prompts to final evaluation

**Pipeline Steps:**
1. **Prompt Generation** - Create persona-KPI-difficulty prompts
2. **Question Generation** - Generate synthetic questions from prompts
3. **Quality Checking** - Validate generated questions
4. **Question Mapping** - Map questions to relevant KPIs
5. **Quality Validation** - Validate mapped questions
6. **Response Evaluation** - Evaluate responses against benchmarks

**Key Features:**
- Async/await for concurrent processing
- Comprehensive error handling and logging
- Automatic directory creation
- Progress tracking and reporting

---

## Data Flow

```
Input Data
    ↓
[Question Prompt Generation]
    ↓ banking_questions_prompts.csv
[Question Generation (OpenAI)]
    ↓ banking_synthetic_questions.csv
[Quality Check - Synthetic Questions]
    ↓ (validation report)
[Question Mapping (Claude AI)]
    ↓ banking_mapped_questions.csv
[Quality Check - Mapped Questions]
    ↓ (validation report)
[Response Evaluation (DeepEval)]
    ↓
Evaluation Results
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- OpenAI API key
- Anthropic Claude API key

### Setup Steps

```bash
# 1. Clone the repository
cd tursio

# 2. Create virtual environment
python -m venv tursio_env
source tursio_env/bin/activate  # On Windows: tursio_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
OPENAI_41_MODEL=gpt-4-turbo
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_CLAUDE_4_5_HAIKU_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_CLAUDE_SONNET_MODEL=claude-sonnet-4-5-20250929
EOF

# 5. Run the pipeline
cd src
python main.py
```

---

## Usage

### Interactive Testing (Notebook)

```python
from test_format.ipynb:

# 1. Generate prompts
generator = PromptGenerator(
    persona_kpi_path="data_bank/banking_persona_kpi.csv",
    golden_data_path="data_bank/banking_questions_persona_kpi_mapping.csv",
    output_path="data_bank/banking_questions_prompts.csv"
)
df_prompts = generator.generate_prompts()

# 2. Generate questions
question_generator = AsyncQuestionGenerator(
    max_concurrent=10,
    output_csv="banking_synthetic_questions.csv",
    input_csv="banking_questions_prompts.csv"
)
await question_generator.run_async()

# 3. Check quality
qqc = QuestionQualityChecker(
    input_csv="banking_synthetic_questions.csv",
    question_column="synthetic_questions"
)
report = qqc.run_all_checks()

# 4. Map questions
mapper = QuestionMapper(
    synthetic_questions_path="banking_synthetic_questions.csv",
    output_path="banking_mapped_questions.csv",
    batch_size=5
)
await mapper.map_and_save()

# 5. Evaluate responses
evaluator = ResponseEvaluator(
    data_dir="benchmark_gpt",
    output_dir="evaluation_results"
)
results = evaluator.run_all_evaluations()
```

### Command Line

```bash
cd src
python main.py
```

---

## Pipeline Architecture

### Flowchart: End-to-End Processing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TURSIO PIPELINE FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────────┐
                        │   INPUT DATA             │
                        ├──────────────────────────┤
                        │ • Personas & KPIs        │
                        │ • Question Mappings      │
                        │ • Golden Examples        │
                        │ • Benchmark Responses    │
                        └──────────┬───────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │  PROMPT GENERATION            │
                    │  (question_prompt.py)         │
                    ├───────────────────────────────┤
                    │ • Combine personas + KPIs     │
                    │ • Apply difficulty levels     │
                    │ • Create prompt templates     │
                    └──────────┬────────────────────┘
                               │
                    ┌──────────▼─────────────────────┐
                    │ QUESTION GENERATION            │
                    │ (question_generate.py)         │
                    ├────────────────────────────────┤
                    │ • OpenAI API calls (async)     │
                    │ • Batch processing             │
                    │ • Rate limit handling          │
                    │ • Response parsing             │
                    └──────────┬─────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  QUALITY CHECK - SYNTHETIC      │
              │  (question_quality_checker.py)  │
              ├─────────────────────────────────┤
              │ ✓ Duplicate detection           │
              │ ✓ Similarity analysis (TF-IDF)  │
              │ ✓ Length outliers               │
              │ ✓ Lexical diversity             │
              └──────────┬──────────────────────┘
                         │
             ┌───────────▼──────────────────┐
             │  QUESTION MAPPING            │
             │  (question_mapper.py)        │
             ├──────────────────────────────┤
             │ • Claude Sonnet API calls    │
             │ • Semantic mapping to KPIs   │
             │ • Exponential backoff retry  │
             │ • Markdown JSON parsing      │
             └────────┬─────────────────────┘
                      │
         ┌────────────▼──────────────┐
         │ QUALITY CHECK - MAPPED     │
         │ (question_quality_checker) │
         ├───────────────────────────┤
         │ ✓ Validate mapped quality  │
         │ ✓ Skip unmappable (NaN)    │
         │ ✓ Report metrics           │
         └────────┬──────────────────┘
                  │
        ┌─────────▼──────────────────┐
        │  RESPONSE EVALUATION       │
        │  (response_eval.py)        │
        ├────────────────────────────┤
        │ • Load benchmark responses │
        │ • Merge with questions     │
        │ • DeepEval metrics:        │
        │   - Answer Relevancy       │
        │   - Bias Detection         │
        │   - Conversation Quality   │
        └────────┬───────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │   RESULTS & REPORTS             │
    ├─────────────────────────────────┤
    │ • out_tursio/ - Tursio results  │
    │ • out_gpt/ - GPT results        │
    │ • out_ppl/ - PPL results        │
    │ • Metrics CSV/JSON              │
    │ • Visualizations (PNG/SVG)      │
    └─────────────────────────────────┘
```

### Module Dependency Graph

```
config.py (Central)
    ↑
    ├── question_prompt.py
    │   ├── Imports: difficulty_definition, table_schema
    │   └── Outputs: banking_questions_prompts.csv
    │
    ├── question_generate.py
    │   ├── Inputs: banking_questions_prompts.csv
    │   ├── Uses: OpenAI API
    │   └── Outputs: banking_synthetic_questions.csv
    │
    ├── question_quality_checker.py
    │   └── Input: Any question CSV
    │
    ├── question_mapper.py
    │   ├── Inputs: banking_synthetic_questions.csv
    │   ├── Uses: Anthropic Claude API
    │   └── Outputs: banking_mapped_questions.csv
    │
    ├── response_metrics.py
    │   ├── Uses: DeepEval library
    │   └── Outputs: Metric results
    │
    ├── response_eval.py
    │   ├── Uses: response_metrics.py
    │   ├── Inputs: benchmark_gpt/, benchmark_ppl/
    │   └── Outputs: out_gpt/, out_ppl/
    │
    ├── utils.py
    │   ├── Uses: DeepEval, Seaborn/Matplotlib
    │   └── Outputs: Visualizations
    │
    └── main.py (Orchestrator)
        └── Calls: All modules above sequentially
```

---

## Configuration Management

All paths are centralized in `config.py`. To modify paths:

```python
# config.py
PROJECT_ROOT = Path(__file__).parent.parent
DATA_BANK_DIR = PROJECT_ROOT / "data_bank"
QUALITY_CHECK_DIR = PROJECT_ROOT / "quality_checks"
# ... etc

# Auto-create all directories
ensure_directories_exist()
```

---

## Key Features

✅ **Async Processing** - Concurrent API calls for improved performance
✅ **Rate Limiting** - Exponential backoff retry logic for API errors
✅ **Quality Validation** - Multi-metric quality checking at each stage
✅ **Flexible Configuration** - Centralized path management
✅ **Error Handling** - Graceful degradation and detailed logging
✅ **Multiple APIs** - OpenAI + Anthropic Claude support
✅ **Comprehensive Evaluation** - DeepEval metrics integration

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 404 Model Not Found | Update model names in .env with valid 2025 Claude/GPT models |
| 429 Rate Limit | Reduce batch_size or increase inter-batch delays in code |
| NaN in Quality Check | Code now automatically filters NaN/empty values |
| Import Errors | Ensure you're running from src/ directory with proper PYTHONPATH |
| API Key Errors | Check .env file has correct OPENAI_API_KEY and ANTHROPIC_API_KEY |

---

## Contact

For questions or issues, please contact the Tursio team.
