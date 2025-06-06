from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools.tavily_search import TavilySearchResults
import sqlite3
import os

# Load environment variables
load_dotenv()

# Initialize SQLite for conversation memory
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Initialize language model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    include_domains=["*.edu", "*.gov", "*.org"],  # Prioritize trusted medical sources
    search_depth="basic"
)

# Define state for the medical agent
class MedicalChatState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: list

# Prompt template for structured response
RESPONSE_TEMPLATE = PromptTemplate(
    input_variables=["query", "context", "citations"],
    template="""
    Query: {query}

    Answer: Based on the available information, {context}

    If no relevant information is found, state: No relevant information was found for this query.

    Citations:
    {citations}
    """
)

def search_tavily(query):
    """Search Tavily for relevant medical information."""
    try:
        results = tavily_tool.invoke(query)
        return results
    except Exception as e:
        print(f"Error searching Tavily: {e}")
        return []

def process_search_results(results):
    """Extract relevant information and citations from Tavily search results."""
    if not results:
        return "No relevant information was found for this query.", []

    context = ""
    citations = []
    for idx, result in enumerate(results, 1):
        content = result.get("content", "No content available")
        if content != "No content available":
            context += f"{content[:500]}... "  # Limit to 500 chars for brevity
            citations.append(f"[{idx}] {result.get('title', 'Untitled')} - {result.get('url', 'No URL')}")
    
    if not context:
        context = "No relevant information was found for this query."
    
    return context, citations

def medical_chatbot(state: MedicalChatState):
    """Process user query, search Tavily, and generate structured response."""
    user_query = state["messages"][-1].content
    
    # Perform Tavily search
    search_results = search_tavily(user_query)
    
    # Process search results
    context, citations = process_search_results(search_results)
    
    # Generate structured response
    response = RESPONSE_TEMPLATE.format(
        query=user_query,
        context=context,
        citations="\n".join(citations) if citations else "No citations available."
    )
    
    # Invoke LLM to refine the response
    llm_response = llm.invoke(response).content
    
    return {
        "messages": [AIMessage(content=llm_response)],
        "search_results": search_results
    }

# Build the graph
graph = StateGraph(MedicalChatState)
graph.add_node("medical_chatbot", medical_chatbot)
graph.add_edge("medical_chatbot", END)
graph.set_entry_point("medical_chatbot")

# Compile the graph with memory
app = graph.compile(checkpointer=memory)

# Configuration for conversation thread
config = {"configurable": {"thread_id": 1}}

# Main interaction loop
print("Medical Agent: Enter your medical query (type 'exit' or 'end' to quit)")
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "end"]:
        print("Medical Agent: Goodbye!")
        break
    else:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "search_results": []
        }, config=config)
        print("Medical Agent: " + result["messages"][-1].content)