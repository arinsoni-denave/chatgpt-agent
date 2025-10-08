# app.py
import asyncio
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Please set it in .env or your environment.")
    st.stop()

# Import the backend’s run function and model
try:
    from backend import run_workflow, WorkflowInput
except Exception as e:
    st.error("Failed to import backend. Please check backend.py and the agents module.")
    st.exception(e)
    st.stop()

st.set_page_config(page_title="ChatGPT Agent", page_icon="🤖")

st.title("🤖 Agent Chat")
st.caption("Ask anything — it routes via query rewrite, classification, then answers.")

# State
if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
for msg in st.session_state.history:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.markdown(content)

# Input
prompt = st.chat_input("Type your question…")

if prompt:
    # Echo user
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant answering
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare conversation history for backend
                conversation_history = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in st.session_state.history
                ]

                # Call the agent backend with conversation history
                payload = WorkflowInput(
                    input_as_text=prompt,
                    conversation_history=conversation_history
                )

                # If run_workflow is async, asyncio.run is correct here (Streamlit runs sync).
                result = asyncio.run(run_workflow(payload))

                # Be defensive about the result shape
                if not isinstance(result, dict):
                    result = {"final_answer": str(result), "path": "unknown"}

                answer = result.get("final_answer") or "No answer returned."
                route = (result.get("path") or "unknown").replace("_", " ")

                st.markdown(f"_Route used:_ **{route}**")
                st.markdown(answer)

                st.session_state.history.append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = f"Error while running agent: {e}"
                st.error(err_msg)
                st.session_state.history.append({"role": "assistant", "content": err_msg})

st.markdown("---")
if st.button("Clear chat"):
    st.session_state.history = []
    st.rerun()
