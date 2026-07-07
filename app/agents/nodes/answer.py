"""
Node 6: Answer — LLM synthesizes the final answer from all accumulated context with citations.
"""
import logging
from app.agents.state import PipelineState
from app.agents.llm_setup import get_llm

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are ExamGPT, a helpful AI study assistant. You must behave like a real conversational assistant.

## Critical Rules
1. NEVER introduce yourself again if there is already a conversation history — you are already mid-conversation.
2. NEVER start your reply with "Hi!", "Hello!", or any greeting if the user is continuing a conversation.
3. If the user's message is a casual greeting (first message only), respond warmly in 1-2 sentences max.
4. If the user says "what is the relevance of this?", "explain more", "what about that?" — they are referring to the PREVIOUS TOPIC in the chat history. Use the history to understand what "this" or "that" refers to.
5. Match your response length to the question. Short casual question = short answer. Complex study question = detailed answer.
6. Only use headings/bullet points for complex study topics. NOT for casual chat.
7. BE DECISIVE. If the user asks you to do a task (e.g., "make a quiz", "summarize this"), DO IT IMMEDIATELY using the provided context. Do NOT ask for preferences or clarifications. Assume reasonable defaults (e.g., 5-10 multiple choice questions) and execute the task.
8. AVOID conversational filler like "I'd be happy to help", "Sure!", "Let me know". Just give the direct answer or output.

## Conversation History (use this to understand follow-up questions)
{chat_history}

## Current Question
{query}

## Retrieved Context from Course Documents
### Knowledge Graph
{kg_context}

### Syllabus Scope
{syllabus_scope}

### Relevant Notes
{notes_context}

### Related PYQs
{pyq_context}

### Evaluation Feedback (if retrying)
{feedback}

## Your Response:"""


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

    # Format chat history as a numbered conversation thread
    if history:
        lines = []
        for i, m in enumerate(history, 1):
            role = "Student" if m["role"] == "user" else "ExamGPT"
            lines.append(f"[{i}] {role}: {m['content']}")
        history_str = "\n".join(lines)
    else:
        history_str = "[No previous messages — this is the start of the conversation]"

    kg_str = "\n".join([
        f"- {r.get('topic', '?')} --[{r.get('relationship', '?')}]--> {r.get('related_name', '?')}"
        for r in kg_context
    ]) if kg_context else "None"

    syllabus_str = f"In scope: {syllabus_scope.get('in_scope', [])}" if syllabus_scope else "None"

    notes_str = "\n\n".join([
        f"[Source: {n.get('source', 'unknown')}]\n{n.get('content', '')}"
        for n in retrieved_notes
    ]) if retrieved_notes else "None"

    pyq_str = "\n\n".join([
        f"[Year: {p.get('year', '?')}, Source: {p.get('source', 'unknown')}]\n{p.get('content', '')}"
        for p in related_pyqs
    ]) if related_pyqs else "None"

    feedback = evaluation.get("feedback", "None") if evaluation else "None"

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

    citations = []
    for n in retrieved_notes:
        citations.append({"source": n.get("source"), "type": "notes"})
    for p in related_pyqs:
        citations.append({"source": p.get("source"), "type": "pyq", "year": p.get("year")})

    logger.info(f"Answer generated: {len(draft_answer)} chars, {len(citations)} citations")
    return {"draft_answer": draft_answer, "citations": citations}
