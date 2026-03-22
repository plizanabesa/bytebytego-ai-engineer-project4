"""
Deep Research API Server

Wraps the multi-agent research pipeline behind a FastAPI endpoint.
Uses Ollama as the model backend via its OpenAI-compatible API.

Start with:  uvicorn server:app --host 0.0.0.0 --port 8001
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Deep Research API")

# Point to the Ollama server
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:3b-instruct"


class ResearchRequest(BaseModel):
    question: str


class ResearchResponse(BaseModel):
    question: str
    sub_questions: list[str]
    report: str


def plan_research(query: str) -> list[str]:
    prompt = f"""You are a research planner. Given a query, break it into 1-5 focused sub-questions.
- Simple factual queries: 1 sub-question
- Moderate topics: 3 sub-questions
- Complex topics needing multiple angles: 5 sub-questions

Query: {query}

Return ONLY the sub-questions, one per line, no numbering or bullets."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    lines = [l.strip() for l in r.choices[0].message.content.strip().split("\n") if l.strip()]
    return lines[:5]


def search_and_summarize(sub_question: str) -> dict:
    with DDGS() as ddgs:
        results = [hit["body"] for hit in ddgs.text(sub_question, max_results=3)]
    snippets = "\n".join(results)
    prompt = f"""Based on these search results, write a concise summary answering: {sub_question}

Search results:
{snippets}

Summary:"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"question": sub_question, "summary": r.choices[0].message.content.strip()}


def synthesize_report(query: str, findings: list[dict]) -> str:
    findings_text = "\n\n".join([f"### {f['question']}\n{f['summary']}" for f in findings])
    prompt = f"""You are a research synthesizer. Combine these findings into a coherent report.

Original query: {query}

Research findings:
{findings_text}

Write a well-structured report that answers the original query. Use markdown formatting."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return r.choices[0].message.content.strip()


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    sub_questions = plan_research(req.question)
    with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
        findings = list(executor.map(search_and_summarize, sub_questions))
    report = synthesize_report(req.question, findings)
    return ResearchResponse(
        question=req.question, sub_questions=sub_questions, report=report
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
