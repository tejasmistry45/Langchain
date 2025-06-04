from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.chains.llm_math.base import LLMMathChain
from dotenv import load_dotenv
import numexpr
load_dotenv()

# Checkpointing memory
memory = MemorySaver()

# LLM (LLaMA3.1 from Groq)
llm = ChatGroq(model="llama-3.1-8b-instant")

# Web search tool
search = TavilySearchResults(max_results=2)

# Math tool
math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
math_tool = Tool.from_function(
    func=math_chain.run,
    name="Calculator",
    description="Useful for when you need to do math or answer math-related questions"
)

# Combine tools
tools = [search, math_tool]

# Create ReAct Agent
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Config for LangGraph thread state
config = {
    "configurable": {"thread_id": "1"}
}

# Run loop
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
