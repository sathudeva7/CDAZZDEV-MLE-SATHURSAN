"""Every prompt Task 3 sends through StructuredLLM, kept apart from the code that sends them.

System messages carry the role, the rules and what each answer field means.
User messages carry only the data, as {placeholders}. Bump `version` whenever
the text changes, so the call log ties each answer to the prompt behind it.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

from common.llm import Prompt

# The llm_sentiment tool labels a whole batch in one call: an agent may call it
# several times a run, and one call per headline would not fit the free tier.
# The label rules match Task 1's HEADLINE_SENTIMENT prompt, so both tasks judge alike.
BATCH_SENTIMENT = Prompt(
    name="batch_headline_sentiment",
    version="1",
    system=(
        "You are an equity research assistant. Judge each numbered news headline's likely effect "
        "on the share price of the research subject named below.\n\n"
        "For every headline, give one entry in `labels`, in the order given:\n"
        "- index: the headline's number.\n"
        "- sentiment: positive if it is likely to lift the share price, negative if it is likely "
        "to weigh on it, neutral if the effect is unclear or the company is only mentioned in passing. "
        "Judge the effect on the share price, not the tone of the wording: a calmly worded report "
        "of insider selling can still be negative.\n"
        "- confidence: from 0 to 1, how sure you are of the label; use below 0.6 when reasonable "
        "analysts could disagree.\n"
        "- reason: at most 15 words naming the mechanism (earnings, demand, costs, regulation, "
        "insider activity, analyst views)."
    ),
    user="Research subject: {subject}\n\nHeadlines:\n{headlines}",
)
