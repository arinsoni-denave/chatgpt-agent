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

# Import the backend’s run function
try:
    from backend import run_workflow, WorkflowInput
except Exception as e:
    st.error("Failed to import backend. Please check backend.py and the agents module.")
    st.exception(e)
    st.stop()

st.set_page_config(page_title="ChatGPT Agent", page_icon="🤖")

st.title("🤖 Agent Chat")
st.caption("Ask anything — it routes via query rewrite, classification, then answers.")

if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
for msg in st.session_state.history:
    role = msg.get("role")
    content = msg.get("content", "")
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.markdown(content)

# Input prompt
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
                conversation_history = []
                for msg in st.session_state.history:
                    conversation_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # Call the agent backend with conversation history
                result = asyncio.run(run_workflow(WorkflowInput(
                    output_text=prompt,
                    conversation_history=conversation_history
                )))
                answer = result.get("final_answer", "No answer returned.")
                route = result.get("path", "unknown").replace("_", " ")

                st.markdown(f"_Route used:_ **{route}**")
                st.markdown(answer)

                st.session_state.history.append({"role": "assistant", "content": answer})

                # Optionally show internal steps
               
            except Exception as e:
                err_msg = f"Error while running agent: {e}"
                st.error(err_msg)
                st.session_state.history.append({"role": "assistant", "content": err_msg})

st.markdown("---")
if st.button("Clear chat"):
    st.session_state.history = []
    st.experimental_rerun()
