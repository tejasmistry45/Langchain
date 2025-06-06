from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from datetime import datetime

load_dotenv()

sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState): 
    return {
       "messages": [llm.invoke(state["messages"])]
    }

search_tool = TavilySearchResults(max_results=3)

# 1. Web Search + Structured Answer Agent with Citations
def web_search_agent(state: BasicChatState):
    query = state["messages"][-1].content
    search_result = search_tool(query)

    prompt = [
        HumanMessage(
            content= f"""
                        You are a helpful assistant. A user Asked: "{query}".
                        Below are the web search results: \n\n{search_result}

                        please:
                        1. summarize the key points using bullet points.
                        2. include the sources or links at the end as citation.

                    """
        )
    ]

    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}

# 2. Latest News Agent

def news_agent(state: BasicChatState):
    query = state["messages"][-1].content
    search_result = search_tool.run("latest news "+ query)


    prompt = [
        HumanMessage(content=f""" 
                                use asked for latest news on: "{query}"

                                Here are the top search results:\n\n{search_result}

                                Summarize them in  3-5 bullet points with links.            
                    """)
    ]

    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}


# 3. Location-Aware Agent
def get_location():
    try:
        ip_info = requests.get("https://ipinfo.io/json").json()
        return ip_info.get("city", "unknown city")
    except:
        return "unknown"

def locartion_agent(state: BasicChatState):
    query = state["messages"][-1].content
    location =  get_location()

    search_query = f"{query}  near {location}"
    search_result = search_tool.run(search_query)

    prompt = [
        HumanMessage(content=f"""
                                User's Location is: {location}
                                They Asked: "{query}"
                                
                                Here are search results:\n\n{search_query}

                                Summarize in bullet points and provide links. 

                            """
        )
    ]
    
    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}

# 4.  Date & Time Agent
def datetime_agent(state: BasicChatState):
    now = datetime.now()
    

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)

graph.add_edge("chatbot", END)

graph.set_entry_point("chatbot")

app = graph.compile(checkpointer=memory)

config = {"configurable": {
    "thread_id": 1
}}

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        print("AI: " + result["messages"][-1].content)