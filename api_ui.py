import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="QMS Navigator Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- App Title and Configuration ---
st.title("🤖 Conversational QMS Navigator")
st.caption("Powered by LangChain and Google Gemini, with Grounding Check")
API_URL = "http://127.0.0.1:8000/ask"
FEEDBACK_FILE_PATH = "feedback_log.csv"

# --- Helper Function for Logging Feedback ---
def log_feedback(question, answer, sources, feedback_score, feedback_text):
    """Logs user feedback to a CSV file."""
    new_feedback = {
        "timestamp": [datetime.now().isoformat()],
        "question": [question],
        "answer": [answer],
        "sources": [json.dumps(sources)],
        "score": [feedback_score],
        "comment": [feedback_text]
    }
    df = pd.DataFrame(new_feedback)
    
    ##### Check if file exists to append or write new
    if os.path.exists(FEEDBACK_FILE_PATH):
        df.to_csv(FEEDBACK_FILE_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_FILE_PATH, mode='w', header=True, index=False)
    st.toast("Thank you for your feedback!", icon="✅")


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the QMS Navigator. How can I assist you with the quality documents today?"}
    ]
if "current_response" not in st.session_state:
    st.session_state.current_response = None

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask about an SOP, work instruction, or policy..."):
    # Reset previous response before getting a new one
    st.session_state.current_response = None
    
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant's response and display it
    with st.chat_message("assistant"):
        with st.spinner("Searching, generating, and verifying the answer..."):
            try:
                response = requests.post(API_URL, json={"question": prompt})
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer", "No answer found.")
                sources = data.get("sources", [])
                
                # Store the response details in session state for feedback
                st.session_state.current_response = {
                    "question": prompt,
                    "answer": answer,
                    "sources": sources
                }
                
                # Format response with sources for display
                full_response = answer
                if sources:
                    source_list = "\n".join([f"- `{source}`" for source in sorted(sources)])
                    full_response += f"\n\n**Sources:**\n{source_list}"
                
                st.markdown(full_response)
                
            except Exception as e:
                error_message = f"**Error:** Could not process your request. Details: {e}"
                st.error(error_message)
                full_response = error_message

    # Add the assistant's response to the message history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Force a rerun to display the feedback section immediately
    st.rerun()

# --- Feedback Section ---
# This section will only appear after a response has been generated
if st.session_state.current_response:
    st.divider()
    st.subheader("Was this response helpful?")

    # Use columns for a tidy layout
    col1, col2, col3 = st.columns([1, 1, 5])
    
    with col1:
        if st.button("👍 Thumbs Up"):
            log_feedback(
                st.session_state.current_response["question"],
                st.session_state.current_response["answer"],
                st.session_state.current_response["sources"],
                1, # Score for "Good"
                "N/A"
            )
            st.session_state.current_response = None # Clear after feedback
            st.rerun()

    with col2:
        if st.button("👎 Thumbs Down"):
            log_feedback(
                st.session_state.current_response["question"],
                st.session_state.current_response["answer"],
                st.session_state.current_response["sources"],
                -1, # Score for "Bad"
                "N/A"
            )
            st.session_state.current_response = None # Clear after feedback
            st.rerun()

    # Create a form for more detailed feedback
    with st.form("feedback_form"):
        feedback_text = st.text_area("Provide more detail (optional):", key="feedback_text_area")
        submitted = st.form_submit_button("Submit Detailed Feedback")

        if submitted and feedback_text:
            log_feedback(
                st.session_state.current_response["question"],
                st.session_state.current_response["answer"],
                st.session_state.current_response["sources"],
                0, # Score for "Commented"
                feedback_text
            )
            st.session_state.current_response = None # Clear after feedback
            st.rerun()