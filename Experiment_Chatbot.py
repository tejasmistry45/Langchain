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
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Set up memory
sqlite_conn = sqlite3.connect("medical_chatbot.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant")
# llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

# Tavily web search tool
tavily_tool = TavilySearchResults(max_results=3)

tools = [tavily_tool]

# ReAct agent (LLM decides whether to use a tool)
agent = create_react_agent(model=llm, tools=tools)

# Define conversation state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

# Compile graph
app = graph.compile(checkpointer=memory)

# Start conversation
config = RunnableConfig(configurable={"thread_id": "1"})

print("\n💬 Medical Agent: Enter your medical query (type 'exit' or 'end' to quit)")
while True:
    user_input = input("🧑‍⚕️ You: ").strip()

    if user_input.lower() in ["exit", "end"]:
        print("👋 Medical Agent: Goodbye! Stay healthy.")
        break

    # LLM decides what to do
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    reply = result["messages"][-1]
    if isinstance(reply, AIMessage):
        print(f"🤖 Medical Agent: {reply.content}\n")
    else:
        print("🤖 Medical Agent: [No valid AI response returned]\n")
