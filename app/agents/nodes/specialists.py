"""
Agents 3-6: specialists that write the response for each intent.
  tutor      — explanations and revision summaries
  quiz       — practice questions in chat
  analyst    — important / most-asked exam questions (uses PYQ stats from the graph)
  chitchat   — greetings, no retrieval
"""
import logging
from app.agents import prompts
from app.agents.llm_setup import chat
from app.agents.nodes.retriever import format_context
from app.agents.state import PipelineState
from app.agents.tracing import traced

logger = logging.getLogger(__name__)

MAX_HISTORY_CHARS = 600


def _history(state: PipelineState) -> list[dict]:
    return [{"role": m["role"], "content": m["content"][:MAX_HISTORY_CHARS]} for m in state.get("chat_history", [])]


def _run(state: PipelineState, prompt: prompts.Prompt, user: str, max_tokens: int, purpose: str) -> dict:
    messages = [{"role": "system", "content": prompt.system}, *_history(state), {"role": "user", "content": user}]
    answer = chat(messages, purpose=purpose, max_tokens=max_tokens, prompt_version=prompt.id)
    return {"draft_answer": answer.strip(), "prompt_version": prompt.id}


@traced("tutor")
def tutor_node(state: PipelineState) -> dict:
    task = prompts.SUMMARY_TASK if state.get("intent") == "summary" else ""
    user = prompts.TUTOR.user.format(context=format_context(state), task=task, input=state["user_query"])
    feedback = (state.get("evaluation") or {}).get("feedback")
    if state.get("retry_count") and feedback:
        user += f"\n\nA reviewer found problems with your previous draft: {feedback}\nWrite an improved answer."
    return _run(state, prompts.TUTOR, user, 1024, "tutor")


@traced("quiz_master")
def quiz_node(state: PipelineState) -> dict:
    user = prompts.QUIZ_CHAT.user.format(context=format_context(state), input=state["user_query"])
    return _run(state, prompts.QUIZ_CHAT, user, 1400, "quiz_master")


@traced("exam_analyst")
def analyst_node(state: PipelineState) -> dict:
    user = prompts.IMPORTANT.user.format(context=format_context(state), input=state["user_query"])
    return _run(state, prompts.IMPORTANT, user, 1200, "exam_analyst")


@traced("chitchat")
def chitchat_node(state: PipelineState) -> dict:
    return _run(state, prompts.CHITCHAT, state["user_query"], 120, "chitchat")
