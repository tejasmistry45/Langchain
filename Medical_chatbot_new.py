import os
import sqlite3
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

load_dotenv()

# Set up memory
sqlite_conn = sqlite3.connect("medical_chatbot.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Initialize LLM 
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialize Tavily tool
tavily_tool = TavilySearchResults(max_results=3)

# Tool list
tools = [tavily_tool]

# Create agent
agent = create_react_agent(model=llm, tools=tools)

# State definition
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile(checkpointer=memory)

config = RunnableConfig(configurable={"thread_id": "1"})

print("\n💬 Medical Agent: Enter your medical query (type 'exit' or 'end' to quit)")
while True:
    user_input = input("🧑‍⚕️ You: ").strip()

    if user_input.lower() in ["exit", "end"]:
        print("👋 Medical Agent: Goodbye! Stay healthy.")
        break

    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    reply = result["messages"][-1]
    if isinstance(reply, AIMessage):
        print(f"🤖 Medical Agent: {reply.content}\n")
    else:
        print("🤖 Medical Agent: [No valid AI response returned]\n")