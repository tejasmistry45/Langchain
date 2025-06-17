import sqlite3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated,TypedDict
from langgraph.graph.message import add_messages

load_dotenv()

# setup memory
sqlite_conn = sqlite3.connect("demo_chatbot.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# initialize the llm
llm = ChatGroq(model="llama-3.1-8b-instant")

# Tavily websearch
tavily_search_tool = TavilySearchResults(max_results=3)

tools = [tavily_search_tool]

# ReAct agent 
agent = create_react_agent(model=llm, tools=tools)

# define conversation state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# build a graph
graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

# compile graph
app = graph.compile(checkpointer=memory)

config = {
    "configurable": {"thread_id":"1"}
}

print("\n💬 Medical Agent: Enter your medical query (type 'exit' or 'end' to quit)")
while True:
    user_input = input("🧑‍⚕️ You: ")
    if user_input.lower() in ["exit","end"]:
        print("👋 Medical Agent: Goodbye! Stay healthy.")
        break

    # LLm decide what to do
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    replay = result["messages"][-1]
    if isinstance(replay, AIMessage):
        print(f"Medical Agent: {replay.content}\n")
    else:
        print("Medical Agent : [No Valid Ai Response Is Returned]\n")
