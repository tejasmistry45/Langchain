from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
import json
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

sqlite_conn = sqlite3.connect("checkpoint_tools.sqlite", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)

class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

llm = ChatGroq(model="llama-3.1-8b-instant")

llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: BasicChatBot):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])], 
    }

def tools_router(state: BasicChatBot):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    

tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatBot)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

app = graph.compile(checkpointer=memory)

config = {"configurable": {
    "thread_id": 1
}}


while True: 
    user_input = input("User: ")
    if user_input in ["exit", "end"]:
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                continue
            elif isinstance(msg, AIMessage):
                print("AI called tool:", msg.content)
            elif isinstance(msg, ToolMessage):
                try:
                    tool_data = json.loads(msg.content)
                    for entry in tool_data:
                        # print("Content:", entry.get("content"))
                        print("URL:", entry.get("url"))
                except Exception as e:
                    print("Failed to parse tool message:", e)


# while True: 
#     user_input = input("User: ")
#     if(user_input in ["exit", "end"]):
#         break
#     else: 
#         result = app.invoke({
#             "messages": [HumanMessage(content=user_input)]
#         })

#         print(result)



