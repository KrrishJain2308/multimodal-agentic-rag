import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

from src.ingest import initialize_vector_store

load_dotenv()

def get_agent():
    """Build and return the Multimodal RAG Agent."""
    # 1. Initialize Vector Store Retriever
    vectorstore = initialize_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. Create the Retriever Tool
    retriever_tool = create_retriever_tool(
        retriever,
        "document_search",
        "Search for information in the uploaded documents. Always use this tool when the user asks questions about the documents or specific facts."
    )
    tools = [retriever_tool]
    
    # Using gemini-3.5-flash-lite as requested by the API for the latest features
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
    )
    
    # 4. Create the Agent using LangGraph
    system_prompt = (
        "You are a highly capable Multimodal AI assistant. You have access to tools that can search through a knowledge base. "
        "CRITICAL INSTRUCTION: Always use the `document_search` tool to find relevant context before answering questions about the uploaded documents. "
        "When you provide an answer based on the documents, you MUST cite the source (e.g., 'According to the uploaded document...'). "
        "If you do not know the answer based on the provided context, clearly state that the information is not present in the documents. "
        "If the user asks a general conversation question, you can answer directly."
    )
    
    agent = create_react_agent(llm, tools=tools, prompt=system_prompt)
    
    return agent

if __name__ == "__main__":
    # Test the agent
    agent = get_agent()
    # response = agent.invoke({"input": "What is the summary of the project?"})
    # print(response["output"])
