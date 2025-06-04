from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

memory = MemorySaver()

llm = ChatGroq(model="llama-3.1-8b-instant")

search = TavilySearchResults(max_results=2)

tools = [search]

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

config = {
    "configurable" : {"thread_id": "1"}
}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "end"]:
        break
    else:
        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
