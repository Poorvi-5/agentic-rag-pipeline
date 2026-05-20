# tests/test_tools.py
"""
Test all 4 tools independently before wiring them to the agent.
Run: python tests/test_tools.py
"""

from rich.console import Console
from rich.panel import Panel

console = Console()

def test_vector_search():
    console.print("\n[bold cyan]Testing Tool 1: Vector Search[/bold cyan]")
    from src.tools.vector_search import search_documents
    result = search_documents("What is RAG and how does it work?")
    console.print(result[:400])
    assert len(result) > 10, "Vector search returned nothing"
    console.print("[green]PASS[/green]")

def test_sql_tool():
    console.print("\n[bold cyan]Testing Tool 2: SQL Query[/bold cyan]")
    from src.tools.sql_tool import run_sql_query, create_sample_database
    create_sample_database()
    result = run_sql_query("Show all products")
    console.print(result)
    assert len(result) > 10
    console.print("[green]PASS[/green]")

def test_web_search():
    console.print("\n[bold cyan]Testing Tool 3: Web Search[/bold cyan]")
    from src.tools.web_search import search_web
    result = search_web("LangChain AI framework")
    console.print(result[:400])
    assert len(result) > 10, "Web search returned nothing"
    console.print("[green]PASS[/green]")

def test_api_tool():
    console.print("\n[bold cyan]Testing Tool 4: External API[/bold cyan]")
    from src.tools.api_tool import get_weather
    result = get_weather("weather in Mumbai")
    console.print(result)
    assert "Temperature" in result or "Weather" in result
    console.print("[green]PASS[/green]")


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]Step 3 — Tool Tests[/bold]\n"
        "Testing all 4 retrieval tools independently",
        border_style="cyan"
    ))

    results = []
    for name, fn in [
        ("Vector Search", test_vector_search),
        ("SQL Tool",      test_sql_tool),
        ("Web Search",    test_web_search),
        ("API Tool",      test_api_tool),
    ]:
        try:
            fn()
            results.append((name, True))
        except Exception as e:
            console.print(f"[red]FAIL — {name}: {e}[/red]")
            results.append((name, False))

    console.print("\n" + "─" * 40)
    for name, ok in results:
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {status}  {name}")

    if all(ok for _, ok in results):
        console.print(Panel.fit(
            "[bold green]All 4 tools working![/bold green]\n"
            "Ready for Step 4 — AI Agent",
            border_style="green"
        ))