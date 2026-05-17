"""
src/router/router.py
--------------------
Person A — Week 3
Router Agent + Parallel Execution + Merge/Insights Layer.
Decides whether to call SQL, RAG, or both — then merges into a PM report.
"""

import asyncio
import json
from enum import Enum
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.sql_agent.agent import create_sql_agent, run_sql_query
from src.rag_pipeline.pipeline import RAGPipeline


class RouteDecision(str, Enum):
    SQL_ONLY = "sql_only"
    RAG_ONLY = "rag_only"
    BOTH = "both"


class PMReport(BaseModel):
    summary: str
    comparison_table: list[dict]  # [{"metric": ..., "your_value": ..., "industry_avg": ..., "status": ...}]
    action_items: list[str]
    data_sources: list[str]


ROUTER_PROMPT = """You are a routing assistant for a PM data tool.
Classify the user's question into one of three categories:

- "sql_only": The question asks for internal company data (KPIs, revenue, churn, ROI, campaign performance numbers)
- "rag_only": The question asks for external/industry knowledge (benchmarks, trends, competitor data, best practices)
- "both": The question needs BOTH internal data AND industry comparison (e.g. "how do we compare to industry?")

Respond with ONLY one of: sql_only, rag_only, both
"""

INSIGHTS_PROMPT = """You are a senior data analyst preparing a report for a Product Manager.

Company data (from internal database):
{sql_result}

Industry benchmark data (from research reports):
{rag_result}

Original question: {question}

Generate a PM report in the following JSON format (return ONLY valid JSON, no markdown):
{{
  "summary": "One sentence executive summary of the key finding",
  "comparison_table": [
    {{
      "metric": "Metric name",
      "your_value": "Company's value with unit",
      "industry_avg": "Industry benchmark with unit",
      "status": "Above average / Below average / On par"
    }}
  ],
  "action_items": [
    "Specific action recommendation 1",
    "Specific action recommendation 2",
    "Specific action recommendation 3"
  ],
  "data_sources": ["Internal TheLook DB", "Industry reports"]
}}
"""


def classify_route(question: str, llm: ChatOpenAI) -> RouteDecision:
    response = llm.invoke([
        ("system", ROUTER_PROMPT),
        ("human", question),
    ])
    raw = response.content.strip().lower()
    try:
        return RouteDecision(raw)
    except ValueError:
        # Default to both if unclear
        return RouteDecision.BOTH


async def run_parallel(question: str, sql_agent, rag_pipeline: RAGPipeline) -> tuple:
    """Run SQL and RAG concurrently. Total time ≈ max(sql_time, rag_time)."""
    sql_task = asyncio.create_task(run_sql_query(question, sql_agent))
    rag_task = asyncio.create_task(rag_pipeline.arun(question))
    return await asyncio.gather(sql_task, rag_task)


def merge_and_generate_report(
    question: str,
    sql_result: Optional[str],
    rag_result: Optional[str],
    llm: ChatOpenAI,
) -> PMReport:
    prompt = INSIGHTS_PROMPT.format(
        sql_result=sql_result or "No internal data retrieved.",
        rag_result=rag_result or "No industry data retrieved.",
        question=question,
    )
    response = llm.invoke([("human", prompt)])

    try:
        data = json.loads(response.content)
        return PMReport(**data)
    except (json.JSONDecodeError, Exception):
        # Graceful fallback if JSON parsing fails
        return PMReport(
            summary=response.content[:200],
            comparison_table=[],
            action_items=["Please retry — report generation encountered a formatting issue."],
            data_sources=["Internal DB", "Industry Reports"],
        )


class RouterAgent:
    def __init__(self):
        self.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # cheaper for routing
        self.insights_llm = ChatOpenAI(model="gpt-4o", temperature=0)     # stronger for final report
        self.sql_agent = create_sql_agent()
        self.rag_pipeline = RAGPipeline().build()

    async def run(self, question: str) -> PMReport:
        # Step 1: Route
        route = classify_route(question, self.router_llm)
        print(f"[Router] Decision: {route.value}")

        sql_result = None
        rag_result = None

        # Step 2: Execute (parallel when both needed)
        if route == RouteDecision.SQL_ONLY:
            sql_data = await run_sql_query(question, self.sql_agent)
            sql_result = sql_data["result"]

        elif route == RouteDecision.RAG_ONLY:
            rag_data = await self.rag_pipeline.arun(question)
            rag_result = rag_data["result"]

        elif route == RouteDecision.BOTH:
            sql_data, rag_data = await run_parallel(question, self.sql_agent, self.rag_pipeline)
            sql_result = sql_data["result"]
            rag_result = rag_data["result"]

        # Step 3: Merge + generate PM report
        return merge_and_generate_report(question, sql_result, rag_result, self.insights_llm)
