"""
Node 6: Answer — LLM synthesizes the final answer from all accumulated context with citations.
"""
import logging
from app.agents.state import PipelineState
from app.agents.llm_setup import get_llm

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are ExamGPT, an expert AI study assistant. Answer the student's question using ONLY the provided context. Be thorough, accurate, and cite your sources.

## Student Question
{query}

## Knowledge Graph Context
{kg_context}

## Syllabus Scope
{syllabus_scope}

## Relevant Notes
{notes_context}

## Related Previous Year Questions
{pyq_context}

## Evaluation Feedback (if retrying)
{feedback}

## Instructions
1. Answer the question thoroughly based on the provided context
2. Cite specific sources using [Source: filename] format
3. If the question appears out of syllabus scope, mention this
4. If related PYQs exist, mention the exam relevance
5. Structure your answer with clear headings and bullet points

## Answer:"""


def answer_node(state: PipelineState) -> dict:
    logger.info("Answer node: synthesizing response")
    llm = get_llm()

    query = state.get("user_query", "")
    kg_context = state.get("kg_context", [])
    syllabus_scope = state.get("syllabus_scope", {})
    retrieved_notes = state.get("retrieved_notes", [])
    related_pyqs = state.get("related_pyqs", [])
    evaluation = state.get("evaluation", {})

    # Format contexts for the prompt
    kg_str = "\n".join([
        f"- {r.get('topic', '?')} --[{r.get('relationship', '?')}]--> {r.get('related_name', '?')}"
        for r in kg_context
    ]) if kg_context else "No knowledge graph data available."

    syllabus_str = f"In scope: {syllabus_scope.get('in_scope', [])}" if syllabus_scope else "No syllabus data."

    notes_str = "\n\n".join([
        f"[Source: {n.get('source', 'unknown')}]\n{n.get('content', '')}"
        for n in retrieved_notes
    ]) if retrieved_notes else "No notes available."

    pyq_str = "\n\n".join([
        f"[Year: {p.get('year', '?')}, Source: {p.get('source', 'unknown')}]\n{p.get('content', '')}"
        for p in related_pyqs
    ]) if related_pyqs else "No PYQ data available."

    feedback = evaluation.get("feedback", "None — first attempt.") if evaluation else "None — first attempt."

    prompt = ANSWER_PROMPT.format(
        query=query,
        kg_context=kg_str,
        syllabus_scope=syllabus_str,
        notes_context=notes_str,
        pyq_context=pyq_str,
        feedback=feedback,
    )

    response = llm.invoke(prompt)
    # ChatHuggingFace returns an AIMessage object; extract string content
    draft_answer = response.content if hasattr(response, "content") else str(response)

    # Build citations list
    citations = []
    for n in retrieved_notes:
        citations.append({"source": n.get("source"), "type": "notes"})
    for p in related_pyqs:
        citations.append({"source": p.get("source"), "type": "pyq", "year": p.get("year")})

    logger.info(f"Answer generated: {len(draft_answer)} chars, {len(citations)} citations")
    return {"draft_answer": draft_answer, "citations": citations}
