"""
Deep Research Chat UI

A Chainlit app that provides a chat interface for the deep research API.

Run:  chainlit run app.py --port 8501
"""

import chainlit as cl
import httpx

API_URL = "http://localhost:8001/research"


@cl.on_message
async def on_message(message: cl.Message):
    # Send a placeholder while researching
    msg = cl.Message(content="Researching your question...")
    await msg.send()

    # Call the deep research API
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(API_URL, json={"question": message.content})
        data = resp.json()

    # Format sub-questions and report
    sub_q_list = "\n".join(f"- {sq}" for sq in data["sub_questions"])
    msg.content = f"**Research Plan:**\n{sub_q_list}\n\n---\n\n{data['report']}"
    await msg.update()
