import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import get_agent
from src.ingest import ingest_document, clear_database

st.set_page_config(page_title="Multimodal Agentic RAG", page_icon="🧠", layout="wide")

st.title("🧠 Multimodal Agentic RAG")
st.markdown("Upload your documents to build the knowledge base, then chat with your AI agent!")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Document Upload
    st.subheader("1. Ingest Data")
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "csv", "docx", "pptx", "xlsx"],
        accept_multiple_files=True
    )
    
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                temp_path = os.path.join("data", uploaded_file.name)
                os.makedirs("data", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    ingest_document(temp_path)
                st.success(f"Successfully ingested: {uploaded_file.name}")
        else:
            st.error("Please upload at least one file first.")
            
    st.divider()
    
    # Clear DB Button
    st.subheader("2. Database Management")
    if st.button("🚨 Reset Knowledge Base", use_container_width=True):
        clear_database()
        st.success("Database cleared! Upload a new document to start fresh.")
        
    st.divider()
    
    # Clear Chat Button
    st.subheader("3. Chat Options")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
@st.cache_resource
def load_agent():
    return get_agent()
    
if "agent" not in st.session_state:
    try:
        st.session_state.agent = load_agent()
    except Exception as e:
        st.session_state.agent = None
        st.error(f"Failed to initialize Agent: {str(e)}")
        st.warning("Please make sure you have added a valid GOOGLE_API_KEY to your .env file.")

# --- CHAT DISPLAY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.agent is None:
        st.error("The agent is not initialized. Please fix the API key and refresh the page.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response with streaming simulation
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Agent is searching and thinking..."):
                try:
                    # Format history for LangGraph
                    formatted_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
                    
                    # We invoke the agent
                    response = st.session_state.agent.invoke({"messages": formatted_messages})
                    
                    # Extract the final answer
                    raw_content = response["messages"][-1].content
                    if isinstance(raw_content, list):
                        full_response = "".join([part.get("text", "") for part in raw_content if isinstance(part, dict) and "text" in part])
                    else:
                        full_response = str(raw_content)
                    
                    # Display the final answer (In a real stream, we'd use agent.stream(), 
                    # but LangGraph agent tool-call streaming requires parsing intermediate steps. 
                    # For a clean Mid-Term demo, we display the result smoothly)
                    import time
                    displayed_response = ""
                    # Quick simulated streaming effect for UI polish
                    for chunk in full_response.split(" "):
                        displayed_response += chunk + " "
                        message_placeholder.markdown(displayed_response + "▌")
                        time.sleep(0.02)
                        
                    message_placeholder.markdown(full_response)
                    
                    # Save to state
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Check your terminal logs for more details.")
