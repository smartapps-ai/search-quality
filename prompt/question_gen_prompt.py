DOMAIN = "Banking domain"
QUESTION_GEN_PROMPT = """You are a {row['persona']} as {domain} expert.
###
Your goal is to improve the following KPI:
{row['kpi']}.

### Task
Provide actionable {row['difficulty']} difficulty questions to achieve this goal.

#### {row['difficulty']} difficulty questions must :
{difficulty_meaning}

### Table join
{TABLE_JOINS}

Here are some example questions:
{examples_text}.

### Output Rules (MANDATORY)
- Output JSON only
- Do NOT include explanations, markdown, or extra text
- Each question must be concise, specific, and clearly tied to the bank domain and the KPI

### Output Schema
{{
  "questions": [
    "string",
    "string",
    ...
  ]
}}

Generate {n_questions} questions directly in JSON format:"""
