# src/tools/sql_tool.py
"""
Tool 2: SQL Database Query Tool
Queries a SQLite database for structured data.
The agent uses this when the question involves
numbers, lists, filters, or tabular data.
"""

import sqlite3
import os
from pathlib import Path
from langchain_core.tools import Tool
from src.config import DATA_DIR


# ── Database path ──────────────────────────────────────────────
DB_PATH = DATA_DIR / "sample_database.db"


def create_sample_database():
    """
    Create a sample SQLite database with demo data.
    Run this once to set up the database.
    Tables: products, employees, sales
    """
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    # Products table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            category    TEXT,
            price       REAL,
            stock       INTEGER
        )
    """)

    # Employees table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            department  TEXT,
            salary      REAL,
            city        TEXT
        )
    """)

    # Sales table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id          INTEGER PRIMARY KEY,
            product_id  INTEGER,
            quantity    INTEGER,
            total       REAL,
            sale_date   TEXT
        )
    """)

    # Insert sample products
    products = [
        (1, "LangChain Book",    "Books",       29.99,  100),
        (2, "AI Laptop",         "Electronics", 999.99,  15),
        (3, "Python Course",     "Education",   49.99,  200),
        (4, "ML Headphones",     "Electronics", 149.99,  50),
        (5, "Data Science Kit",  "Education",   79.99,   75),
        (6, "RAG Framework",     "Software",    199.99,  30),
        (7, "Vector DB License", "Software",    299.99,  20),
        (8, "AI Keyboard",       "Electronics",  89.99,  60),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO products VALUES (?,?,?,?,?)",
        products
    )

    # Insert sample employees
    employees = [
        (1, "Arjun Sharma",   "AI Research",   85000, "Mumbai"),
        (2, "Priya Patel",    "Engineering",   92000, "Bangalore"),
        (3, "Rahul Gupta",    "Data Science",  78000, "Delhi"),
        (4, "Sneha Singh",    "Engineering",   88000, "Hyderabad"),
        (5, "Vikram Kumar",   "AI Research",   95000, "Pune"),
        (6, "Anita Desai",    "Management",   110000, "Mumbai"),
        (7, "Rohan Mehta",    "Data Science",  82000, "Bangalore"),
        (8, "Kavya Reddy",    "Engineering",   90000, "Chennai"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?)",
        employees
    )

    # Insert sample sales
    sales = [
        (1, 1, 10, 299.90,  "2024-01-15"),
        (2, 2,  2, 1999.98, "2024-01-18"),
        (3, 3, 15, 749.85,  "2024-02-01"),
        (4, 4,  5, 749.95,  "2024-02-10"),
        (5, 5,  8, 639.92,  "2024-02-15"),
        (6, 6,  3, 599.97,  "2024-03-01"),
        (7, 7,  2, 599.98,  "2024-03-05"),
        (8, 8, 10, 899.90,  "2024-03-10"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO sales VALUES (?,?,?,?,?)",
        sales
    )

    conn.commit()
    conn.close()
    print(f" Sample database created at: {DB_PATH}")


def get_schema() -> str:
    """Return the database schema so the LLM knows what tables exist."""
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]

    schema_parts = []
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = cur.fetchall()
        col_defs = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        schema_parts.append(f"Table '{table}': {col_defs}")

    conn.close()
    return "\n".join(schema_parts)


def run_sql_query(natural_language_query: str) -> str:
    """
    Convert a natural language query into SQL and execute it.

    The LLM inside this tool generates SQL from the question,
    then we execute it safely against the SQLite database.

    Args:
        natural_language_query: Plain English question about the data
    Returns:
        Query results as a formatted string
    """
    if not DB_PATH.exists():
        create_sample_database()

    # Get schema so we can build correct SQL
    schema = get_schema()

    # Use Groq LLM to convert natural language → SQL
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from src.config import GROQ_API_KEY

    llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

    sql_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert SQL query writer for SQLite. "
         "Given a database schema and a question, write ONLY the SQL query. "
         "No explanation. No markdown. Just the raw SQL.\n"
         "IMPORTANT: SQLite is case-sensitive for text. "
         "Always use LOWER() for text comparisons. "
         "Example: WHERE LOWER(category) = LOWER('electronics')\n\n"
         f"Database Schema:\n{schema}"),
        ("human", "{question}")
    ])

    chain    = sql_prompt | llm
    response = chain.invoke({"question": natural_language_query})
    sql      = response.content.strip()

    # Clean up any accidental markdown
    sql = sql.replace("```sql", "").replace("```", "").strip()

    print(f"\n Generated SQL: {sql}")

    # Execute SQL safely
    try:
        conn    = sqlite3.connect(str(DB_PATH))
        cur     = conn.cursor()
        cur.execute(sql)
        rows    = cur.fetchall()
        columns = [desc[0] for desc in cur.description] \
                  if cur.description else []
        conn.close()

        if not rows:
            return "Query executed successfully but returned no results."

        # Format as a readable table
        header = " | ".join(columns)
        divider = "-" * len(header)
        row_lines = [
            " | ".join(str(v) for v in row)
            for row in rows
        ]

        return (
            f"SQL Query Results:\n"
            f"{header}\n{divider}\n"
            + "\n".join(row_lines)
            + f"\n\n({len(rows)} row(s) returned)"
        )

    except Exception as e:
        return f"SQL execution error: {str(e)}\nGenerated SQL was: {sql}"


# ── Wrap as LangChain Tool ────────────────────────────────────
sql_tool = Tool(
    name="sql_database_query",
    func=run_sql_query,
    description=(
        "Query the structured SQL database for information about "
        "products, employees, and sales data. Use this tool when "
        "the user asks about prices, quantities, salaries, lists, "
        "counts, or comparisons of structured data. "
        "Input should be a plain English question about the data."
    )
)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    # First create the DB
    create_sample_database()
    print(f"\n Schema:\n{get_schema()}\n")

    # Test query
    q = "Show me all electronics products with their prices"
    print(f" Query: {q}\n")
    result = run_sql_query(q)
    print(result)