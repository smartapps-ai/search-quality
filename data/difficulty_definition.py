DIFFICULTY_DEFINITION = {
    "simple": 
        "- Single metric or straightforward count, percentage, or ranking\n"
        "- Minimal filtering and no segmentation\n"
        "- No time-series analysis or trend comparison\n"
        "- Answerable with one aggregation or query",

    "medium": 
        "- Involves filtering, grouping, or basic segmentation\n"
        "- May include a single time window or comparison\n"
        "- Requires joins across a small number of datasets\n"
        "- Produces ratios, breakdowns, or ranked results but not deep causal analysis",

    "hard": 
        "- Require multi-step analysis, segmentation, or time-based trends\n"
        "- Focus on early-warning signals, concentration risk, behavioral risk, or product-level risk\n"
        "- Go beyond simple counts or percentages\n"
        "- Be suitable for executive risk reviews or regulatory discussions"
}