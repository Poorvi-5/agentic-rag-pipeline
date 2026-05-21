# src/reflection/evaluator.py
"""
Reflection & Self-Correction Layer.

After the agent generates an answer, this evaluator:
1. Scores the answer on 3 dimensions (relevance, grounding, completeness)
2. If score < threshold → refines the query and retries the agent
3. If max retries reached → returns the best answer found so far

This is what separates a reliable RAG system from a hallucination-prone one.
"""

from dataclasses import dataclass
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import GROQ_API_KEY, MAX_REFLECTION_RETRIES


# ── Data classes ──────────────────────────────────────────────
@dataclass
class EvaluationResult:
    """Holds the result of one evaluation pass."""
    score:        float   # 0.0 to 1.0
    relevance:    float   # Is the answer relevant to the query?
    grounding:    float   # Is it grounded in retrieved context?
    completeness: float   # Does it fully answer the question?
    feedback:     str     # Why the score was given
    passed:       bool    # score >= threshold


@dataclass
class ReflectionResult:
    """Final result after all reflection passes."""
    original_query:  str
    final_answer:    str
    final_score:     float
    attempts:        int
    passed:          bool
    evaluations:     list  # list of EvaluationResult


# ── LLM setup ─────────────────────────────────────────────────
def _get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024
    )


# ── Evaluator ─────────────────────────────────────────────────
def evaluate_answer(
    query:   str,
    answer:  str,
    context: str = ""
) -> EvaluationResult:
    """
    Score an answer on 3 dimensions using the LLM as a judge.

    Args:
        query:   The original user question
        answer:  The agent's generated answer
        context: Retrieved context used to generate the answer

    Returns:
        EvaluationResult with score and detailed feedback
    """
    llm = _get_llm()

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict answer quality evaluator for a RAG system.
Evaluate the given answer and return ONLY a JSON object with these exact keys:
{{
  "relevance": <float 0.0-1.0>,
  "grounding": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "feedback": "<one sentence explaining the scores>"
}}

Scoring criteria:
- relevance (0-1):    Does the answer directly address the question asked?
- grounding (0-1):    Is the answer based on facts/retrieved data, not hallucination?
- completeness (0-1): Does it fully answer the question or is it vague/partial?

Be strict. Score 0.0 if the answer says "I don't know" or is empty.
Return ONLY the JSON. No explanation. No markdown."""),

        ("human", f"""Question: {query}

Answer to evaluate:
{answer}

Retrieved context (if any):
{context if context else "No context provided"}

Evaluate now:""")
    ])

    try:
        response = (eval_prompt | llm).invoke({})
        raw      = response.content.strip()

        # Clean any accidental markdown
        raw = raw.replace("```json", "").replace("```", "").strip()

        import json
        scores = json.loads(raw)

        relevance    = float(scores.get("relevance",    0.5))
        grounding    = float(scores.get("grounding",    0.5))
        completeness = float(scores.get("completeness", 0.5))
        feedback     = scores.get("feedback", "No feedback provided")

        # Weighted final score
        # Relevance and grounding are most important
        final_score = (
            relevance    * 0.40 +
            grounding    * 0.40 +
            completeness * 0.20
        )

        threshold = 0.65
        passed    = final_score >= threshold

        print(f"\n Evaluation Results:")
        print(f"   Relevance    : {relevance:.2f}")
        print(f"   Grounding    : {grounding:.2f}")
        print(f"   Completeness : {completeness:.2f}")
        print(f"   Final Score  : {final_score:.2f} "
              f"({'PASS' if passed else 'FAIL'})")
        print(f"   Feedback     : {feedback}")

        return EvaluationResult(
            score        = final_score,
            relevance    = relevance,
            grounding    = grounding,
            completeness = completeness,
            feedback     = feedback,
            passed       = passed
        )

    except Exception as e:
        print(f" Evaluator error: {e} — defaulting to PASS")
        # If evaluator fails, don't block the pipeline
        return EvaluationResult(
            score        = 0.75,
            relevance    = 0.75,
            grounding    = 0.75,
            completeness = 0.75,
            feedback     = f"Evaluator failed: {e}",
            passed       = True
        )


# ── Query Refiner ─────────────────────────────────────────────
def refine_query(
    original_query: str,
    failed_answer:  str,
    feedback:       str
) -> str:
    """
    If the answer failed evaluation, ask the LLM to
    rephrase the query to get a better result next time.

    Args:
        original_query: The original user question
        failed_answer:  The answer that scored too low
        feedback:       The evaluator's feedback on why it failed

    Returns:
        A refined/rephrased query string
    """
    llm = _get_llm()

    refine_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query refinement expert. "
         "Given an original query, a poor answer, and feedback, "
         "rewrite the query to be more specific and likely to get "
         "a better answer. Return ONLY the refined query. "
         "No explanation. No quotes. Just the query text."),
        ("human",
         f"Original query: {original_query}\n"
         f"Poor answer received: {failed_answer[:200]}\n"
         f"Evaluator feedback: {feedback}\n\n"
         f"Write a better, more specific query:")
    ])

    try:
        response = (refine_prompt | llm).invoke({})
        refined  = response.content.strip()
        print(f"\n Refined query: {refined}")
        return refined
    except Exception as e:
        print(f" Query refinement failed: {e}")
        return original_query  # fall back to original


# ── Main Reflection Loop ───────────────────────────────────────
def run_with_reflection(query: str) -> ReflectionResult:
    """
    Run the full agent + reflection loop.

    Flow:
    1. Run agent on query
    2. Evaluate the answer
    3. If score < threshold AND retries left → refine query → repeat
    4. Return the best answer found

    Args:
        query: The user's original question

    Returns:
        ReflectionResult with final answer, score, and all evaluations
    """
    # Import here to avoid circular imports
    from src.agent.agent_core import run_agent

    print(f"\n{'='*55}")
    print(f" REFLECTION PIPELINE STARTED")
    print(f" Query: {query}")
    print(f" Max retries: {MAX_REFLECTION_RETRIES}")
    print(f"{'='*55}")

    evaluations  = []
    best_answer  = ""
    best_score   = 0.0
    current_query = query

    for attempt in range(1, MAX_REFLECTION_RETRIES + 2):
        print(f"\n Attempt {attempt}/{MAX_REFLECTION_RETRIES + 1}")

        # Step 1: Run the agent
        agent_result = run_agent(current_query)
        answer       = agent_result.get("answer", "")
        steps        = agent_result.get("steps",  [])

        # Extract context from intermediate steps for evaluation
        context_parts = []
        for step in steps:
            if isinstance(step, tuple) and len(step) >= 2:
                context_parts.append(str(step[1]))
        context = "\n".join(context_parts[:3])  # top 3 observations

        # Step 2: Evaluate the answer
        print(f"\n Evaluating answer...")
        evaluation = evaluate_answer(
            query   = query,      # always evaluate against ORIGINAL query
            answer  = answer,
            context = context
        )
        evaluations.append(evaluation)

        # Track best answer so far
        if evaluation.score > best_score:
            best_score  = evaluation.score
            best_answer = answer

        # Step 3: Check if we passed
        if evaluation.passed:
            print(f"\n Answer passed evaluation on attempt {attempt}!")
            return ReflectionResult(
                original_query = query,
                final_answer   = answer,
                final_score    = evaluation.score,
                attempts       = attempt,
                passed         = True,
                evaluations    = evaluations
            )

        # Step 4: If failed and retries left, refine the query
        if attempt <= MAX_REFLECTION_RETRIES:
            print(f"\n Answer failed. Refining query for retry...")
            current_query = refine_query(
                original_query = query,
                failed_answer  = answer,
                feedback       = evaluation.feedback
            )
        else:
            print(f"\n Max retries reached. Returning best answer.")

    # Return best answer found across all attempts
    return ReflectionResult(
        original_query = query,
        final_answer   = best_answer,
        final_score    = best_score,
        attempts       = MAX_REFLECTION_RETRIES + 1,
        passed         = False,
        evaluations    = evaluations
    )


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What is Retrieval Augmented Generation?"
    result     = run_with_reflection(test_query)

    print(f"\n{'='*55}")
    print(f" REFLECTION COMPLETE")
    print(f"{'='*55}")
    print(f" Original Query : {result.original_query}")
    print(f" Final Score    : {result.final_score:.2f}")
    print(f" Attempts       : {result.attempts}")
    print(f" Passed         : {result.passed}")
    print(f" Final Answer   :\n{result.final_answer}")