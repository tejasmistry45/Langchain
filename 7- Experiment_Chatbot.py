import sqlite3
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages
import wikipedia
import re

load_dotenv()

# SQLite memory setup
sqlite_conn = sqlite3.connect("Database/medical_chatbot.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Search tool
tavily_tool = TavilySearchResults(max_results=3)

# Tool Definitions
@tool
def get_current_datetime(dummy: str = "") -> str:
    """Returns the current date and time. Input can be anything (not used)."""
    now = datetime.now()
    return f"Today is {now.strftime('%A, %d, %B, %Y')} and The Current Time is {now.strftime('%I:%M %p')}."

@tool
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia for the query and returns a short summary."""
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"The query is ambiguous. Did you mean: {', '.join(e.options[:5])}?"
    except wikipedia.exceptions.PageError:
        return "No page found on Wikipedia for your query."

@tool
def search_web(query: str) -> str:
    """Searches the web using Tavily for current information."""
    try:
        results = tavily_tool.invoke({"query": query})
        if results:
            formatted_results = []
            for result in results[:2]:  # Limit to 2 results
                formatted_results.append(f"- {result.get('content', '')[:200]}...")
            return "\n".join(formatted_results)
        return "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

# Tools dictionary
TOOLS = {
    "get_current_datetime": get_current_datetime,
    "search_wikipedia": search_wikipedia,
    "search_web": search_web
}

class ReActState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    thought: str
    action: str
    action_input: str
    observation: str
    final_answer: str
    history: List[Dict[str, Any]]

class ReActAgent:
    def __init__(self, llm, tools, max_iterations=3):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        
    def parse_action(self, text: str) -> tuple:
        """Parse action and action_input from LLM response"""
        # Look for Action: and Action Input: patterns
        action_pattern = r"Action:\s*([^\n]+)"
        input_pattern = r"Action Input:\s*([^\n]+)"
        
        action_match = re.search(action_pattern, text, re.IGNORECASE)
        input_match = re.search(input_pattern, text, re.IGNORECASE)
        
        action = action_match.group(1).strip() if action_match else ""
        action_input = input_match.group(1).strip() if input_match else ""
        
        return action, action_input
    
    def execute_action(self, action: str, action_input: str) -> str:
        """Execute the action using available tools"""
        if action in self.tools:
            try:
                result = self.tools[action].invoke(action_input)
                return str(result)
            except Exception as e:
                return f"Error executing {action}: {str(e)}"
        else:
            available_tools = ", ".join(self.tools.keys())
            return f"Invalid action '{action}'. Available tools: {available_tools}"
    
    def should_continue(self, state: ReActState) -> bool:
        """Determine if we should continue the ReAct loop"""
        return (state["iteration"] < self.max_iterations and 
                not state["final_answer"] and 
                "Final Answer:" not in state.get("thought", ""))
    
    def react_step(self, state: ReActState) -> ReActState:
        """Single ReAct step: Thought -> Action -> Observation"""
        
        # Build context from history
        context = ""
        if state["history"]:
            for i, step in enumerate(state["history"], 1):
                context += f"\nStep {i}:\n"
                context += f"Thought: {step['thought']}\n"
                context += f"Action: {step['action']}\n"
                context += f"Action Input: {step['action_input']}\n"
                context += f"Observation: {step['observation']}\n"
        
        # Get the user question
        user_question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        # Create ReAct prompt
        react_prompt = f"""You are a medical assistant using ReAct (Reasoning and Acting) methodology.

Question: {user_question}

Previous steps:{context}

Available tools:
- get_current_datetime: Get current date and time
- search_wikipedia: Search Wikipedia for medical definitions and background info
- search_web: Search web for current medical information

You MUST follow this exact format:

Thought: [Your reasoning about what to do next]
Action: [Choose one tool from the available tools]
Action Input: [Input for the chosen tool]

If you have enough information to answer, instead use:
Thought: [Your reasoning]
Final Answer: [Your complete answer to the user's question]

Current iteration: {state["iteration"] + 1}/{self.max_iterations}
"""
        
        # Get LLM response
        response = self.llm.invoke([SystemMessage(content=react_prompt)])

        print("=" * 60)
        print("Raw Response: ", response)
        response_text = response.content
        # print("Raw LLM Response: ", response_text)
        
        print(f"\n🧠 Iteration {state['iteration'] + 1}:")
        print(f"LLM Response:\n{response_text}")
        
        # Check for Final Answer
        if "Final Answer:" in response_text:
            final_answer_pattern = r"Final Answer:\s*(.+?)(?:\n|$)"
            final_match = re.search(final_answer_pattern, response_text, re.DOTALL)
            if final_match:
                state["final_answer"] = final_match.group(1).strip()
                state["thought"] = response_text
                return state
        
        # Parse thought
        thought_pattern = r"Thought:\s*([^\n]+(?:\n(?!Action:)[^\n]+)*)"
        thought_match = re.search(thought_pattern, response_text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else response_text
        
        # Parse action and action input
        action, action_input = self.parse_action(response_text)
        
        # Execute action
        observation = ""
        if action and action_input:
            observation = self.execute_action(action, action_input)
            print(f"Action: {action}")
            print(f"Action Input: {action_input}")
            print(f"Observation: {observation}")
        else:
            observation = "Could not parse action and action input from response."
            print(f"Parsing Error: {observation}")
        
        # Update state
        step_info = {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation
        }
        
        state["history"].append(step_info)
        state["thought"] = thought
        state["action"] = action
        state["action_input"] = action_input
        state["observation"] = observation
        state["iteration"] += 1
        
        return state
    
    def finalize_answer(self, state: ReActState) -> ReActState:
        """Generate final answer if not already provided"""
        if not state["final_answer"]:
            # Build context from all history
            context = ""
            for i, step in enumerate(state["history"], 1):
                context += f"\nStep {i}:\n"
                context += f"Thought: {step['thought']}\n"
                context += f"Action: {step['action']}\n"
                context += f"Action Input: {step['action_input']}\n"
                context += f"Observation: {step['observation']}\n"
            
            user_question = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
                    break
            
            final_prompt = f"""Based on the ReAct process below, provide a final comprehensive answer to the user's question.

Question: {user_question}

ReAct Process:{context}

Provide a clear, helpful, and medically accurate final answer:"""
            
            response = self.llm.invoke([SystemMessage(content=final_prompt)])
            state["final_answer"] = response.content
        
        return state

# Create ReAct agent
react_agent = ReActAgent(llm, TOOLS, max_iterations=3)

# Graph nodes
def react_reasoning_node(state: ReActState) -> ReActState:
    """ReAct reasoning step"""
    return react_agent.react_step(state)

def should_continue_node(state: ReActState) -> str:
    """Decide whether to continue or finalize"""
    if react_agent.should_continue(state):
        return "continue"
    else:
        return "finalize"

def finalize_node(state: ReActState) -> ReActState:
    """Generate final answer"""
    return react_agent.finalize_answer(state)

# Build the graph
workflow = StateGraph(ReActState)

# Add nodes
workflow.add_node("react_step", react_reasoning_node)
workflow.add_node("finalize", finalize_node)

# Add edges
workflow.set_entry_point("react_step")
workflow.add_conditional_edges(
    "react_step",
    should_continue_node,
    {
        "continue": "react_step",
        "finalize": "finalize"
    }
)
workflow.add_edge("finalize", END)

# Compile the graph
app = workflow.compile(checkpointer=memory)

# System prompt for medical context
MEDICAL_SYSTEM_PROMPT = """
You are a helpful medical assistant that provides accurate, evidence-based medical information.
- Always prioritize patient safety and recommend consulting healthcare professionals
- Use tools to gather accurate information before responding
- Be empathetic and supportive in your responses
- Clearly distinguish between general information and specific medical advice
"""

def run_medical_react_agent(user_input: str) -> str:
    """Run the ReAct medical agent"""
    
    # Initialize state
    initial_state = {
        "messages": [
            SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ],
        "iteration": 0,
        "thought": "",
        "action": "",
        "action_input": "",
        "observation": "",
        "final_answer": "",
        "history": []
    }
    
    # Run the workflow
    config = RunnableConfig(configurable={"thread_id": f"react_{datetime.now().timestamp()}"})
    
    print(f"\n🔍 Starting ReAct process for: '{user_input}'")
    print("=" * 60)
    
    final_state = app.invoke(initial_state, config=config)
    
    print("\n" + "=" * 60)
    # print("📋 REACT PROCESS SUMMARY:")
    # print("=" * 60)
    
    # # Show the ReAct process summary
    # for i, step in enumerate(final_state["history"], 1):
    #     print(f"\n🔄 ITERATION {i}:")
    #     print(f"💭 Thought: {step['thought']}")
    #     print(f"🛠️  Action: {step['action']}")
    #     print(f"📝 Action Input: {step['action_input']}")
    #     print(f"👁️  Observation: {step['observation'][:200]}..." if len(step['observation']) > 200 else f"👁️  Observation: {step['observation']}")
    
    print(f"\n🎯 FINAL ANSWER:")
    print("=" * 60)
    return final_state["final_answer"]

# Main chat loop
if __name__ == "__main__":
    print("\n💬 ReAct Medical Agent is ready!")
    print("🔬 This agent uses ReAct methodology: Reasoning + Acting")
    print("📊 Maximum 3 iterations per question")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("⚕️ You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "end"]:
            print("🤖 Medical Agent: Goodbye! Stay healthy and consult healthcare professionals when needed.")
            break
        
        if not user_input:
            continue
            
        try:
            final_answer = run_medical_react_agent(user_input)
            print(f"🤖 Medical Agent: {final_answer}\n")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("🤖 Medical Agent: I apologize, but I encountered an error. Please try rephrasing your question.\n")