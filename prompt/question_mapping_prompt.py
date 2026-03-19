DOMAIN = "Banking domain"
QUESTION_MAPPING_PROMPT = """
You are a subject-matter expert in {DOMAIN} with deep experience in KPIs, personas, and data-driven analysis. 
Your responsibility is to map each original user question to the single best-matching question that is directly answerable using open web–searchable data.

Task:
For each row in the input CSV:
1. Analyze the original user question, including its analytical intent and scope.
2. Evaluate the candidate questions derived from web search results.
3. Select the ONE candidate question that most accurately aligns with:
   - The intent and analytical depth of the original question
   - The referenced KPI (or a clearly defined, industry-accepted proxy KPI)
   - The persona’s typical decision-making needs
   - The specified difficulty level (simple / medium / hard)

Selection rules:
- Do NOT rewrite or paraphrase.
- Do NOT introduce new KPIs.
- The mapped question must be answerable using publicly available, open web data.
- Preserve the analytical level implied by the difficulty:
  - simple: descriptive, single-metric, snapshot questions
  - medium: comparative, segmented, or time-based questions
  - hard: multi-step, trend-based, segmented, or contextualized questions
- If multiple candidates are plausible:
  - Prefer the most specific, unambiguous, and commonly used industry question.
  - Prefer exact KPI alignment over loosely related metrics.
- If no candidate adequately matches all criteria, return an empty string ("") for "mapped_question".

Input:

Row context:
- original_question: {original_question}
- persona: {persona}
- kpi: {kpi}
- difficulty: {difficulty}

Output:
Return results STRICTLY in the following JSON format:

{{
  "original_question": "{original_question}",
  "mapped_question": "string"
}}

Output ONLY valid JSON.
Do NOT include explanations, comments, or any additional text.
"""
