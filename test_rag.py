import sys, os
from dotenv import load_dotenv

# Load env
load_dotenv()

# We don't need to re-ingest if the Streamlit app already did it, but let's be sure it's ingested
from src.ingest import ingest_document
from src.agent import get_agent

pdf_path = "data/Assignment 1.pdf"
print("Ingesting...")
ingest_document(pdf_path)

print("Starting Agent...")
agent = get_agent()
response = agent.invoke({"messages": [{"role": "user", "content": "What is this pdf about? Please summarize the main topics."}]})

print("\n--- AGENT RESPONSE ---")
print(response["messages"][-1].content)
