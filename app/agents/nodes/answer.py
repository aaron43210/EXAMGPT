"""
Node 6: Answer — LLM synthesizes the final answer from all accumulated context with citations.
"""
import logging
from app.agents.state import PipelineState
from app.agents.llm_setup import get_llm

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are ExamGPT, a friendly and knowledgeable AI study assistant. Your personality is warm, encouraging, and conversational — like a brilliant senior student helping a friend.

## Behavior Rules
- If the user is greeting you (e.g., "hi", "hello", "hey") or asking something casual, respond warmly and naturally in 1-2 sentences. Do NOT use headings or bullet points for these.
- If the user asks a genuine study or exam question, answer thoroughly using the provided context, with clear structure and citations.
- If context is not available but the question is simple, answer from your general knowledge concisely.
- Always match the tone and complexity of the question. Short question = short answer. Complex question = detailed answer.

## Recent Conversation
{chat_history}

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
1. Read the Behavior Rules above first.
2. If this is a casual message, respond naturally without structure.
3. If it's a study question, answer using only the provided context.
4. Cite specific sources using [Source: filename] format.
5. If related PYQs exist, mention the exam relevance.

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
    history = state.get("chat_history", [])

    # Format chat history as a readable thread
    if history:
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history
        )
    else:
        history_str = "This is the start of the conversation."

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
        chat_history=history_str,
        kg_context=kg_str,
        syllabus_scope=syllabus_str,
        notes_context=notes_str,
        pyq_context=pyq_str,
        feedback=feedback,
    )

    response = llm.invoke(prompt)
    draft_answer = response.content if hasattr(response, "content") else str(response)

    # Build citations list
    citations = []
    for n in retrieved_notes:
        citations.append({"source": n.get("source"), "type": "notes"})
    for p in related_pyqs:
        citations.append({"source": p.get("source"), "type": "pyq", "year": p.get("year")})

    logger.info(f"Answer generated: {len(draft_answer)} chars, {len(citations)} citations")
    return {"draft_answer": draft_answer, "citations": citations}
