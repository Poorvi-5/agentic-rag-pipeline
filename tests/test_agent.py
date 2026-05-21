# tests/test_agent.py
"""
Test the full ReAct agent end-to-end.
Each test exercises a different tool path.
Run: python tests/test_agent.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_vector_search_path():
    """Agent should use vector_search for knowledge base questions."""
    console.print("\n[bold cyan]Test 1: Knowledge Base Query[/bold cyan]")
    console.print("Query: 'What is RAG?'")

    from src.agent.agent_core import run_agent
    result = run_agent("What is RAG and how does it work?")

    assert len(result["answer"]) > 20, "Answer too short"
    assert result["steps"], "No tool was called"
    console.print(f"[green]PASS[/green] — Answer: {result['answer'][:150]}...")
    return result


def test_sql_path():
    """Agent should use sql_database_query for structured data."""
    console.print("\n[bold cyan]Test 2: SQL Database Query[/bold cyan]")
    console.print("Query: 'What products are in the database?'")

    from src.agent.agent_core import run_agent
    result = run_agent("List all products in the database with their prices")

    assert len(result["answer"]) > 20, "Answer too short"
    console.print(f"[green]PASS[/green] — Answer: {result['answer'][:150]}...")
    return result


def test_web_search_path():
    """Agent should use web_search for current information."""
    console.print("\n[bold cyan]Test 3: Web Search Query[/bold cyan]")
    console.print("Query: 'Latest AI news today'")

    from src.agent.agent_core import run_agent
    result = run_agent("What are the latest news in AI today?")

    assert len(result["answer"]) > 20, "Answer too short"
    console.print(f"[green]PASS[/green] — Answer: {result['answer'][:150]}...")
    return result


def test_api_path():
    """Agent should use external_api for weather queries."""
    console.print("\n[bold cyan]Test 4: External API Query[/bold cyan]")
    console.print("Query: 'Weather in Mumbai'")

    from src.agent.agent_core import run_agent
    result = run_agent("What is the current weather in Mumbai?")

    assert len(result["answer"]) > 20, "Answer too short"
    console.print(f"[green]PASS[/green] — Answer: {result['answer'][:150]}...")
    return result


def print_summary(results: list):
    """Print a summary table of all test results."""
    table = Table(title="Agent Test Results")
    table.add_column("Test",        style="cyan")
    table.add_column("Tool Used",   style="yellow")
    table.add_column("Steps",       style="magenta")
    table.add_column("Status",      style="green")

    for name, result in results:
        if result:
            tools_used = [
                str(step[0].tool)
                for step in result.get("steps", [])
                if hasattr(step[0], "tool")
            ]
            tool_str = ", ".join(tools_used) if tools_used else "unknown"
            steps    = str(len(result.get("steps", [])))
            status   = "PASS"
        else:
            tool_str = "—"
            steps    = "—"
            status   = "FAIL"

        table.add_row(name, tool_str, steps, status)

    console.print(table)


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]Step 4 — Agent Tests[/bold]\n"
        "Testing ReAct agent with all 4 tool paths",
        border_style="cyan"
    ))

    all_results = []

    for test_name, test_fn in [
        ("Knowledge Base (RAG)",  test_vector_search_path),
        ("SQL Database",          test_sql_path),
        ("Web Search",            test_web_search_path),
        ("External API",          test_api_path),
    ]:
        try:
            result = test_fn()
            all_results.append((test_name, result))
        except Exception as e:
            console.print(f"[red]FAIL — {test_name}: {e}[/red]")
            all_results.append((test_name, None))

    print_summary(all_results)

    passed = sum(1 for _, r in all_results if r)
    if passed == len(all_results):
        console.print(Panel.fit(
            f"[bold green]All {passed} agent tests passed![/bold green]\n"
            "Agent is routing queries to correct tools.\n"
            "Ready for Step 5 — Reflection Layer!",
            border_style="green"
        ))
    else:
        console.print(
            f"[yellow]{passed}/{len(all_results)} tests passed.[/yellow]"
        )