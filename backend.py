# backend.py

import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel

from agents import (
    FileSearchTool, WebSearchTool, CodeInterpreterTool,
    Agent, ModelSettings, TResponseInputItem, Runner, RunConfig
)

# --- Tool definitions ---
file_search = FileSearchTool(
    vector_store_ids=["vs_68e5f5b82cf881918f36b4862bebaa13"]
)
web_search_preview = WebSearchTool(
    search_context_size="medium",
    user_location={"type": "approximate"}
)
code_interpreter = CodeInterpreterTool(tool_config={
    "type": "code_interpreter",
    "container": {"type": "auto", "file_ids": []}
})

# Schema for classify agent’s output
class ClassifySchema(BaseModel):
    operating_procedure: str

# --- Agent definitions ---
query_rewrite = Agent(
    name="Query rewrite",
    instructions="Rewrite the user's question to be more specific and relevant to the knowledge base.",
    model="gpt-5",
    model_settings=ModelSettings(store=True)
)

classify = Agent(
    name="Classify",
    instructions="Determine whether the question should use the Q&A or fact-finding process.",
    model="gpt-5",
    output_type=ClassifySchema,
    model_settings=ModelSettings(store=True)
)

internal_q_a = Agent(
    name="Internal Q&A",
    instructions="Answer the user's question using the knowledge tools you have on hand (file or web search). Be concise and answer succinctly, using bullet points and summarizing the answer up front",
    model="gpt-5",
    tools=[file_search],
    model_settings=ModelSettings(store=True)
)

external_fact_finding = Agent(
    name="External fact finding",
    instructions=(
        "Use web search to identify the answer to the input query, and provide a concise response supported by evidence from reputable sources, each clearly cited.\n\n"
        "Analyze relevant data from your search results before answering. If you find conflicting information, indicate this in your supporting evidence.\n\n"
        "In your final output, always:\n"
        "- Provide a short, direct answer first.\n"
        "- Follow with bullet points summarizing supporting evidence, each with a source clearly indicated (with URL or publication name).\n"
        "- Ensure each bullet point corresponds to a different relevant source when available.\n"
        "# Steps\n\n"
        "1. Perform a web search using appropriate tools to find recent, reliable information addressing the user query.\n"
        "2. Analyze and compare the top sources for accuracy and relevance.\n"
        "3. Summarize your findings and compose a concise answer.\n"
        "4. Present supporting bullet points, each citing its original source.\n"
        "5. If significant discrepancies are present between sources, note these in your evidence.\n\n"
        "# Output Format\n\n"
        "- Short direct answer (1–3 sentences).\n"
        "- Bullet point list of 2–5 supporting facts or data, each with source attribution.\n"
    ),
    model="gpt-5",
    tools=[web_search_preview, code_interpreter],
    model_settings=ModelSettings(store=True)
)

agent = Agent(
    name="Agent",
    instructions="Ask the user to provide more detail so you can help them by either answering their question or running data analysis relevant to their query",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True)
)

# Workflow input model
class WorkflowInput(BaseModel):
    input_as_text: str
    conversation_history: List[Dict[str, str]] = []

def _msg_to_response_block(role: str, text: str) -> TResponseInputItem:
    """
    Convert a chat history message into a Responses-style input item with the
    correct content block type for this SDK.
    - user -> input_text
    - assistant -> output_text
    - system (if any) -> summary_text (or input_text if your SDK prefers)
    """
    text = text or ""
    if role == "assistant":
        content_block = {"type": "output_text", "text": text}
    elif role == "system":
        # Many SDKs accept summary_text for system context; if not, swap to input_text.
        content_block = {"type": "summary_text", "text": text}
    else:  # default to user
        content_block = {"type": "input_text", "text": text}

    return {
        "role": role if role in ("user", "assistant", "system") else "user",
        "content": [content_block],
    }

# Main workflow function
async def run_workflow(workflow_input: WorkflowInput) -> Dict[str, Any]:
    wf = workflow_input.model_dump()

    # Convert prior conversation into role-correct blocks
    conversation_history: List[TResponseInputItem] = []
    for msg in wf.get("conversation_history", []):
        conversation_history.append(
            _msg_to_response_block(msg.get("role", "user"), msg.get("content", ""))
        )

    # Append current user turn
    conversation_history.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": wf["input_as_text"]}],
        }
    )

    # ---- Query rewrite ----
    query_rewrite_result_temp = await Runner.run(
        query_rewrite,
        input=[
            *conversation_history,
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Original question: {wf['input_as_text']}"
                    }
                ],
            },
        ],
        run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
    )
    conversation_history.extend([item.to_input_item() for item in query_rewrite_result_temp.new_items])
    query_rewrite_result = {"output_text": query_rewrite_result_temp.final_output_as(str)}

    # ---- Classify ----
    classify_result_temp = await Runner.run(
        classify,
        input=[
            *conversation_history,
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Question: {query_rewrite_result['output_text']}"
                    }
                ],
            },
        ],
        run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
    )
    conversation_history.extend([item.to_input_item() for item in classify_result_temp.new_items])
    classify_result = {
        "output_text": classify_result_temp.final_output.json(),
        "output_parsed": classify_result_temp.final_output.model_dump(),
    }

    # ---- Branch ----
    op = (classify_result["output_parsed"] or {}).get("operating_procedure", "")
    if op == "q-and-a":
        internal_q_a_result_temp = await Runner.run(
            internal_q_a,
            input=[*conversation_history],
            run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
        )
        conversation_history.extend([item.to_input_item() for item in internal_q_a_result_temp.new_items])
        return {
            "final_answer": internal_q_a_result_temp.final_output_as(str),
            "path": "internal_q_a",
        }

    if op == "fact-finding":
        external_fact_finding_result_temp = await Runner.run(
            external_fact_finding,
            input=[*conversation_history],
            run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
        )
        conversation_history.extend([item.to_input_item() for item in external_fact_finding_result_temp.new_items])
        return {
            "final_answer": external_fact_finding_result_temp.final_output_as(str),
            "path": "external_fact_finding",
        }

    # Fallback agent
    agent_result_temp = await Runner.run(
        agent,
        input=[*conversation_history],
        run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
    )
    conversation_history.extend([item.to_input_item() for item in agent_result_temp.new_items])
    return {
        "final_answer": agent_result_temp.final_output_as(str),
        "path": "agent",
    }


# ---- FastAPI app ----
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agent Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class Query(BaseModel):
    input_text: str

@app.post("/query")
async def query(req: Query):
    return await run_workflow(WorkflowInput(input_as_text=req.input_text))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
