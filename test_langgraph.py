import os
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
agent = create_react_agent(llm, tools=[])
res = agent.invoke({"messages": [HumanMessage(content="Hello")]})
print(res["messages"][-1].content)
