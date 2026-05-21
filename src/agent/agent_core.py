# src/agent/agent_core.py
"""
Core ReAct Agent — the brain of the Agentic RAG pipeline.

ReAct = Reasoning + Acting
The agent thinks (Thought) → picks a tool (Action) →
reads result (Observation) → repeats if needed →
gives final answer.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.agent.prompts import AGENT_SYSTEM_PROMPT
from src.tools.vector_search import vector_search_tool
from src.tools.sql_tool import sql_tool
from src.tools.web_search import web_search_tool
from src.tools.api_tool import api_tool


# ── Build the LLM ─────────────────────────────────────────────
def get_llm():
    """Initialize the Groq LLM."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.1,        # low temp = more focused reasoning
        max_tokens=2048
    )


# ── Register all 4 tools ──────────────────────────────────────
def get_tools():
    """Return the list of all available tools."""
    return [
        vector_search_tool,
        sql_tool,
        web_search_tool,
        api_tool
    ]


# ── Build the ReAct prompt ────────────────────────────────────
def get_react_prompt():
    """
    Build the ReAct prompt template.
    LangChain's ReAct agent needs these exact variables:
    {tools}, {tool_names}, {input}, {agent_scratchpad}
    """
    template = AGENT_SYSTEM_PROMPT + """

Available tools:
{tools}

Tool names: {tool_names}

Use this EXACT format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    return PromptTemplate.from_template(template)


# ── Build the Agent ───────────────────────────────────────────
def build_agent():
    """
    Assemble the full ReAct agent with all tools.
    Returns an AgentExecutor ready to run queries.
    """
    llm    = get_llm()
    tools  = get_tools()
    prompt = get_react_prompt()

    # create_react_agent builds the reasoning chain
    agent  = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # AgentExecutor runs the agent loop
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # prints Thought/Action/Observation
        max_iterations=5,       # max tool calls per query
        max_execution_time=60,  # timeout in seconds
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return executor


# ── Main query function ───────────────────────────────────────
def run_agent(query: str) -> dict:
    """
    Run the agent on a user query.

    Args:
        query: The user's question as a string

    Returns:
        dict with keys:
          - answer:  the final answer string
          - steps:   list of (action, observation) tuples
          - query:   the original query
    """
    print(f"\n{'='*55}")
    print(f" Query: {query}")
    print(f"{'='*55}\n")

    executor = build_agent()

    try:
        result = executor.invoke({"input": query})

        answer = result.get("output", "No answer generated.")
        steps  = result.get("intermediate_steps", [])

        print(f"\n{'='*55}")
        print(f" FINAL ANSWER:\n{answer}")
        print(f"{'='*55}\n")

        return {
            "query":  query,
            "answer": answer,
            "steps":  steps
        }

    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        print(error_msg)
        return {
            "query":  query,
            "answer": error_msg,
            "steps":  []
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with 4 different query types
    test_queries = [
        "What is Retrieval Augmented Generation?",
        "Show me all products under $100",
        "What is the weather in Delhi?",
        "What are the latest developments in AI today?"
    ]

    for query in test_queries:
        result = run_agent(query)
        print(f"\n Query : {result['query']}")
        print(f" Answer: {result['answer'][:200]}...")
        print(f" Steps : {len(result['steps'])} tool call(s)")
        print("-" * 55)