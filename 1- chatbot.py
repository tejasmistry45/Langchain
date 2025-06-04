from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pymed import PubMed
import os

# Load environment variables
load_dotenv()

# Initialize SQLite for conversation memory
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Initialize language model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialize PubMed client
pubmed = PubMed(tool="MedicalAgent", email=os.getenv("EMAIL_ADDRESS"))  # Set your email in .env

# Define state for the medical agent
class MedicalChatState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: list

# Prompt template for structured response
RESPONSE_TEMPLATE = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    Query: {query}
    
    Answer: Based on the available medical literature, {context}
    
    If no relevant information is found, state: No relevant information was found in the medical literature for this query.
    
    Citations:
    {citations}
    """
)

def search_pubmed(query, max_results=3):
    """Search PubMed for relevant articles."""
    try:
        results = pubmed.query(query, max_results=max_results)
        articles = []
        for article in results:
            articles.append({
                "title": article.title,
                "abstract": article.abstract or "No abstract available",
                "pmid": article.pubmed_id,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/"
            })
        return articles
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []

def process_search_results(articles):
    """Extract relevant information from search results."""
    if not articles:
        return "No relevant information was found in the medical literature for this query.", []
    
    context = ""
    citations = []
    for idx, article in enumerate(articles, 1):
        if article["abstract"] != "No abstract available":
            context += f"{article['abstract'][:500]}... "  # Limit to 500 chars for brevity
            citations.append(f"[{idx}] {article['title']} (PMID: {article['pmid']}) - {article['url']}")
    
    if not context:
        context = "No relevant information was found in the medical literature for this query."
    
    return context, citations

def medical_chatbot(state: MedicalChatState):
    """Process user query, search PubMed, and generate structured response."""
    user_query = state["messages"][-1].content
    
    # Perform PubMed search
    search_results = search_pubmed(user_query)
    
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