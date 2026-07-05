"""
Node 7: Evaluation — critiques the draft answer for correctness, syllabus alignment, completeness.
Conditional edge: on fail → loops back to answer_node (bounded ≤3 retries).
"""
import json
import logging
from app.agents.state import PipelineState
from app.agents.llm_setup import get_llm
from app.core.config import get_settings

logger = logging.getLogger(__name__)

EVAL_PROMPT = """You are a strict academic quality evaluator for ExamGPT. Evaluate the following AI-generated answer.

## Original Question
{query}

## Draft Answer
{draft_answer}

## Available Context Used
Notes: {has_notes}
PYQs: {has_pyqs}
Knowledge Graph: {has_kg}
Syllabus: {has_syllabus}

## Evaluation Criteria
1. **Correctness**: Is the answer factually accurate based on the provided context?
2. **Completeness**: Does it fully address the question?
3. **Syllabus Alignment**: Is the answer within the course scope?
4. **Citations**: Does it reference source material?
5. **Clarity**: Is it well-structured and easy to understand?
6. **Relevance**: Is the answer relevant to the question?


Respond with a JSON object:
{{
  "passed": true/false,
  "score": 1-10,
  "feedback": "specific feedback if failed, or 'Approved' if passed"
}}

Respond ONLY with the JSON."""


def evaluation_node(state: PipelineState) -> dict:
    logger.info("Evaluation node: critiquing answer")
    llm = get_llm()

    query = state.get("user_query", "")
    draft_answer = state.get("draft_answer", "")
    retry_count = state.get("retry_count", 0)

    prompt = EVAL_PROMPT.format(
        query=query,
        draft_answer=draft_answer,
        has_notes="Yes" if state.get("retrieved_notes") else "No",
        has_pyqs="Yes" if state.get("related_pyqs") else "No",
        has_kg="Yes" if state.get("kg_context") else "No",
        has_syllabus="Yes" if state.get("syllabus_scope", {}).get("syllabus_available") else "No",
    )

    response = llm.invoke(prompt)

    try:
        eval_data = json.loads(response.strip())
        passed = eval_data.get("passed", True)
        feedback = eval_data.get("feedback", "No feedback")
        score = eval_data.get("score", 5)
    except (json.JSONDecodeError, AttributeError):
        # If LLM can't produce valid JSON, pass the answer through
        passed = True
        feedback = "Evaluation parsing failed — passing through."
        score = 5

    evaluation = {
        "passed": passed,
        "score": score,
        "feedback": feedback,
    }

    logger.info(f"Evaluation: passed={passed}, score={score}, retry={retry_count}")
    return {
        "evaluation": evaluation,
        "retry_count": retry_count + (0 if passed else 1),
    }


def should_retry(state: PipelineState) -> str:
    """Conditional edge: retry answer if evaluation failed and under retry limit."""
    settings = get_settings()
    evaluation = state.get("evaluation", {})
    retry_count = state.get("retry_count", 0)

    if not evaluation.get("passed", True) and retry_count < settings.MAX_EVAL_RETRIES:
        logger.info(f"Retrying answer (attempt {retry_count + 1}/{settings.MAX_EVAL_RETRIES})")
        return "retry"

    return "end"
