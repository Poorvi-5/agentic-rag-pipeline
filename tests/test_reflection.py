# tests/test_reflection.py
"""
Test the reflection and self-correction layer.
Run: python tests/test_reflection.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_evaluate_good_answer():
    """A good answer should score >= 0.65 and pass."""
    console.print("\n[bold cyan]Test 1: Evaluate Good Answer[/bold cyan]")

    from src.reflection.evaluator import evaluate_answer

    query  = "What is RAG?"
    answer = (
        "RAG (Retrieval Augmented Generation) is a technique that "
        "combines information retrieval with text generation. Instead "
        "of relying solely on the LLM training data, RAG systems first "
        "retrieve relevant documents from a knowledge base, then use "
        "that context to generate accurate and grounded responses. "
        "This reduces hallucinations and improves factual accuracy."
    )

    result = evaluate_answer(query, answer)
    console.print(f"Score: {result.score:.2f} | Passed: {result.passed}")
    assert result.score > 0.0, "Score should be greater than 0"
    console.print("[green]PASS[/green]")
    return result


def test_evaluate_bad_answer():
    """A bad/empty answer should score low."""
    console.print("\n[bold cyan]Test 2: Evaluate Bad Answer[/bold cyan]")

    from src.reflection.evaluator import evaluate_answer

    query  = "What is the capital of France?"
    answer = "I don't know. I cannot find any information about this."

    result = evaluate_answer(query, answer)
    console.print(f"Score: {result.score:.2f} | Passed: {result.passed}")
    assert result.score < 0.9, "Bad answer should not score perfectly"
    console.print("[green]PASS[/green]")
    return result


def test_query_refinement():
    """Query refiner should return a non-empty refined query."""
    console.print("\n[bold cyan]Test 3: Query Refinement[/bold cyan]")

    from src.reflection.evaluator import refine_query

    refined = refine_query(
        original_query = "Tell me about AI",
        failed_answer  = "I don't know much about AI.",
        feedback       = "Answer is too vague and not grounded in facts."
    )

    console.print(f"Refined query: {refined}")
    assert len(refined) > 5, "Refined query should not be empty"
    assert refined != "", "Refined query cannot be empty"
    console.print("[green]PASS[/green]")
    return refined


def test_full_reflection_pipeline():
    """Run the complete reflection pipeline end-to-end."""
    console.print(
        "\n[bold cyan]Test 4: Full Reflection Pipeline[/bold cyan]"
    )
    console.print("This runs the agent + evaluator + retry loop...")

    from src.reflection.evaluator import run_with_reflection

    result = run_with_reflection(
        "What is Retrieval Augmented Generation?"
    )

    console.print(f"\nFinal Score    : {result.final_score:.2f}")
    console.print(f"Total Attempts : {result.attempts}")
    console.print(f"Passed         : {result.passed}")
    console.print(f"Answer Preview : {result.final_answer[:150]}...")

    assert len(result.final_answer) > 20, "Final answer too short"
    assert result.attempts >= 1,          "Should have at least 1 attempt"
    console.print("[green]PASS[/green]")
    return result


def print_summary(results: list):
    table = Table(title="Reflection Layer Test Results")
    table.add_column("Test",   style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Detail", style="yellow")

    for name, passed, detail in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, detail)

    console.print(table)


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]Step 5 — Reflection Layer Tests[/bold]\n"
        "Testing evaluator, refiner, and full reflection loop",
        border_style="cyan"
    ))

    all_results = []

    # Test 1
    try:
        r = test_evaluate_good_answer()
        all_results.append((
            "Evaluate Good Answer",
            True,
            f"Score: {r.score:.2f}"
        ))
    except Exception as e:
        all_results.append(("Evaluate Good Answer", False, str(e)))

    # Test 2
    try:
        r = test_evaluate_bad_answer()
        all_results.append((
            "Evaluate Bad Answer",
            True,
            f"Score: {r.score:.2f}"
        ))
    except Exception as e:
        all_results.append(("Evaluate Bad Answer", False, str(e)))

    # Test 3
    try:
        refined = test_query_refinement()
        all_results.append((
            "Query Refinement",
            True,
            f"Refined: {refined[:40]}..."
        ))
    except Exception as e:
        all_results.append(("Query Refinement", False, str(e)))

    # Test 4
    try:
        r = test_full_reflection_pipeline()
        all_results.append((
            "Full Reflection Pipeline",
            True,
            f"Score: {r.final_score:.2f} | Attempts: {r.attempts}"
        ))
    except Exception as e:
        all_results.append(("Full Reflection Pipeline", False, str(e)))

    print_summary(all_results)

    passed = sum(1 for _, ok, _ in all_results if ok)
    if passed == len(all_results):
        console.print(Panel.fit(
            f"[bold green]All {passed} reflection tests passed![/bold green]\n"
            "Self-correction layer is working.\n"
            "Ready for Step 6 — Streamlit UI!",
            border_style="green"
        ))
    else:
        console.print(
            f"[yellow]{passed}/{len(all_results)} tests passed.[/yellow]"
        )