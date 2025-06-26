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
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
import wikipedia
import re
import json
from collections import Counter
import asyncio

load_dotenv()

# SQLite memory setup
sqlite_conn = sqlite3.connect("Database/medical_chatbot.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# LLM
llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
llm_deterministic = ChatGroq(model="llama3-70b-8192", temperature=0.0)  # For consistency checks

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

class CoTSCReActState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    reasoning_paths: List[Dict[str, Any]]  # Multiple CoT reasoning paths
    consistency_check: Dict[str, Any]      # Self-consistency results
    react_history: List[Dict[str, Any]]    # ReAct action history
    current_thought: str
    current_action: str
    current_action_input: str
    current_observation: str
    final_answer: str
    confidence_score: float
    reasoning_quality: str

class CoTSCReActAgent:
    def __init__(self, llm, llm_deterministic, tools, max_iterations=3, num_reasoning_paths=3):
        self.llm = llm
        self.llm_deterministic = llm_deterministic
        self.tools = tools
        self.max_iterations = max_iterations
        self.num_reasoning_paths = num_reasoning_paths
        
    def generate_cot_reasoning_paths(self, question: str, context: str = "", conversation_history: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Generate multiple Chain of Thought reasoning paths with conversation history"""
        
        cot_prompt_template = """You are a medical expert using Chain of Thought reasoning.

Previous Conversation History:
{conversation_history}

Current Question: {question}
{context}

Please provide a detailed step-by-step reasoning process to approach this medical question, taking into account the conversation history to ensure continuity and relevance.

1. First, identify the key medical concepts and symptoms mentioned in the current question and previous conversation
2. Consider differential diagnoses or relevant medical conditions, referencing prior context if applicable
3. Think about what additional information might be needed
4. Reason through the medical logic step by step
5. Consider potential complications or important considerations

Provide your reasoning in a clear, logical sequence:

Step 1: [Your first reasoning step]
Step 2: [Your second reasoning step]
Step 3: [Continue with logical steps]
...
Conclusion: [Your preliminary conclusion based on this reasoning path]

Be thorough but focused on medical accuracy and safety, ensuring alignment with the conversation history."""
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            for entry in conversation_history:
                history_text += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
        
        reasoning_paths = []
        
        for i in range(self.num_reasoning_paths):
            try:
                prompt = cot_prompt_template.format(
                    conversation_history=history_text or "No previous conversation history.",
                    question=question,
                    context=f"\nAdditional Context: {context}" if context else ""
                )
                
                response = self.llm.invoke([SystemMessage(content=prompt)])
                
                steps = self._parse_reasoning_steps(response.content)
                
                reasoning_paths.append({
                    "path_id": i + 1,
                    "raw_response": response.content,
                    "steps": steps,
                    "conclusion": self._extract_conclusion(response.content)
                })
                
                print(f"🧠 CoT Path {i + 1} Generated: {len(steps)} reasoning steps")
                
            except Exception as e:
                print(f"❌ Error generating CoT path {i + 1}: {str(e)}")
                
        return reasoning_paths
    
    def _parse_reasoning_steps(self, text: str) -> List[str]:
        """Parse reasoning steps from CoT response"""
        steps = []
        
        # Look for numbered steps
        step_pattern = r"Step\s+\d+:\s*([^Step]+?)(?=Step\s+\d+:|Conclusion:|$)"
        matches = re.findall(step_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            step = match.strip()
            if step:
                steps.append(step)
        
        # If no numbered steps found, try to extract logical segments
        if not steps:
            sentences = text.split('.')
            steps = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
        
        return steps
    
    def _extract_conclusion(self, text: str) -> str:
        """Extract conclusion from CoT response"""
        conclusion_pattern = r"Conclusion:\s*([^$]+)"
        match = re.search(conclusion_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # If no explicit conclusion, take the last meaningful sentence
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        return sentences[-1] + '.' if sentences else "No clear conclusion found."
    
    def perform_self_consistency_check(self, reasoning_paths: List[Dict[str, Any]], question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Perform self-consistency check across multiple reasoning paths with conversation history"""
        
        if len(reasoning_paths) < 2:
            return {"consistency_score": 1.0, "consensus": "Insufficient paths for consistency check"}
        
        conclusions = [path["conclusion"] for path in reasoning_paths]
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            history_text = "\nPrevious Conversation History:\n" + "\n".join(
                [f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in conversation_history]
            )
        
        consistency_prompt = f"""Analyze the consistency of these multiple reasoning approaches to the same medical question, considering the conversation history.

{history_text}

Question: {question}

Reasoning Path Conclusions:
{chr(10).join([f"Path {i+1}: {conclusion}" for i, conclusion in enumerate(conclusions)])}

Please analyze:
1. How consistent are these conclusions with each other and the conversation history?
2. What are the common themes across all paths?
3. Are there any conflicting recommendations or diagnoses?
4. What is the overall consensus, if any?
5. Rate the consistency on a scale of 0.0 to 1.0

Provide your analysis in this format:
Consistency Score: [0.0 to 1.0]
Common Themes: [List common elements]
Conflicts: [Any disagreements or conflicts]
Consensus: [Overall consensus conclusion]
Reliability: [HIGH/MEDIUM/LOW based on consistency]"""

        try:
            response = self.llm_deterministic.invoke([SystemMessage(content=consistency_prompt)])
            
            score_match = re.search(r"Consistency Score:\s*([0-9.]+)", response.content)
            consistency_score = float(score_match.group(1)) if score_match else 0.5
            
            reliability_match = re.search(r"Reliability:\s*(\w+)", response.content, re.IGNORECASE)
            reliability = reliability_match.group(1).upper() if reliability_match else "MEDIUM"
            
            return {
                "consistency_score": consistency_score,
                "analysis": response.content,
                "reliability": reliability,
                "num_paths_analyzed": len(reasoning_paths)
            }
            
        except Exception as e:
            print(f"❌ Error in consistency check: {str(e)}")
            return {"consistency_score": 0.5, "error": str(e)}
    
    def parse_action(self, text: str) -> tuple:
        """Parse action and action_input from LLM response"""
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
    
    def should_continue_react(self, state: CoTSCReActState) -> bool:
        """Determine if we should continue the ReAct loop"""
        return (state["iteration"] < self.max_iterations and 
                not state["final_answer"] and 
                "Final Answer:" not in state.get("current_thought", ""))
    
    def react_step_with_cot(self, state: CoTSCReActState) -> CoTSCReActState:
        """Enhanced ReAct step that incorporates CoT reasoning and conversation history"""
        
        react_context = ""
        if state["react_history"]:
            for i, step in enumerate(state["react_history"], 1):
                react_context += f"\nReAct Step {i}:\n"
                react_context += f"Thought: {step['thought']}\n"
                react_context += f"Action: {step['action']}\n"
                react_context += f"Action Input: {step['action_input']}\n"
                react_context += f"Observation: {step['observation']}\n"
        
        cot_context = ""
        if state["reasoning_paths"]:
            cot_context = "\nChain of Thought Analysis:\n"
            for path in state["reasoning_paths"]:
                cot_context += f"Reasoning Path {path['path_id']}: {path['conclusion']}\n"
        
        consistency_context = ""
        if state["consistency_check"]:
            consistency_context = f"\nConsistency Analysis:\n"
            consistency_context += f"Reliability: {state['consistency_check'].get('reliability', 'UNKNOWN')}\n"
            consistency_context += f"Score: {state['consistency_check'].get('consistency_score', 0.5)}\n"
        
        # Get conversation history from messages
        conversation_history = []
        user_question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                conversation_history.append({"user": msg.content, "assistant": ""})
                user_question = msg.content
            elif isinstance(msg, AIMessage) and conversation_history:
                conversation_history[-1]["assistant"] = msg.content
        
        history_text = "\nPrevious Conversation History:\n" + "\n".join(
            [f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in conversation_history[:-1]]
        ) if len(conversation_history) > 1 else "\nNo previous conversation history."
        
        enhanced_react_prompt = f"""You are a medical assistant using enhanced ReAct (Reasoning and Acting) with Chain of Thought.

{history_text}

Current Question: {user_question}

{cot_context}
{consistency_context}
Previous ReAct Steps:{react_context}

Available tools:
- get_current_datetime: Get current date and time
- search_wikipedia: Search Wikipedia for medical definitions and background info
- search_web: Search web for current medical information

Based on the Chain of Thought analysis and conversation history, decide what action to take next.

You MUST follow this exact format:

Thought: [Your reasoning about what to do next, considering the CoT analysis and conversation history]
Action: [Choose one tool from the available tools]
Action Input: [Input for the chosen tool]

If you have enough information to provide a comprehensive answer, instead use:
Thought: [Your final reasoning, integrating CoT analysis, ReAct observations, and conversation history]
Final Answer: [Your complete, medically accurate answer]

Current iteration: {state["iteration"] + 1}/{self.max_iterations}
Remember to prioritize patient safety and recommend consulting healthcare professionals when appropriate.
"""
        
        response = self.llm.invoke([SystemMessage(content=enhanced_react_prompt)])
        response_text = response.content
        
        print(f"\n🔄 Enhanced ReAct Step {state['iteration'] + 1}:")
        print(f"Response:\n{response_text}")
        
        if "Final Answer:" in response_text:
            final_answer_pattern = r"Final Answer:\s*(.+)"
            final_match = re.search(final_answer_pattern, response_text, re.DOTALL)
            if final_match:
                complete_answer = final_match.group(1).strip()
                state["final_answer"] = complete_answer
                state["current_thought"] = response_text
                state["messages"].append(AIMessage(content=complete_answer))
                print(f"✅ Final answer captured: {len(complete_answer)} characters")
                return state
        
        thought_pattern = r"Thought:\s*([^\n]+(?:\n(?!Action:)[^\n]+)*)"
        thought_match = re.search(thought_pattern, response_text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else response_text
        
        action, action_input = self.parse_action(response_text)
        
        observation = ""
        if action and action_input:
            observation = self.execute_action(action, action_input)
            print(f"Action: {action}")
            print(f"Action Input: {action_input}")
            print(f"Observation: {observation}")
        else:
            observation = "Could not parse action and action input from response."
            print(f"Parsing Error: {observation}")
        
        step_info = {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation
        }
        
        state["react_history"].append(step_info)
        state["current_thought"] = thought
        state["current_action"] = action
        state["current_action_input"] = action_input
        state["current_observation"] = observation
        state["iteration"] += 1
        
        return state
    
    def finalize_answer_with_confidence(self, state: CoTSCReActState) -> CoTSCReActState:
        """Generate final answer with confidence assessment"""
        if not state["final_answer"]:
            cot_context = ""
            for path in state["reasoning_paths"]:
                cot_context += f"CoT Path {path['path_id']}: {path['conclusion']}\n"
            
            react_context = ""
            for i, step in enumerate(state["react_history"], 1):
                react_context += f"Step {i}: {step['thought']} -> {step['action']} -> {step['observation'][:100]}...\n"
            
            consistency_info = state["consistency_check"]
            
            conversation_history = []
            user_question = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    conversation_history.append({"user": msg.content, "assistant": ""})
                    user_question = msg.content
                elif isinstance(msg, AIMessage) and conversation_history:
                    conversation_history[-1]["assistant"] = msg.content
            
            history_text = "\nPrevious Conversation History:\n" + "\n".join(
                [f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in conversation_history[:-1]]
            ) if len(conversation_history) > 1 else "\nNo previous conversation history."
            
            final_prompt = f"""Based on the comprehensive analysis below, provide a complete and detailed medical response.

{history_text}

Current Question: {user_question}

Chain of Thought Analysis:
{cot_context}

Self-Consistency Check:
Reliability: {consistency_info.get('reliability', 'UNKNOWN')}
Consistency Score: {consistency_info.get('consistency_score', 0.5)}

ReAct Investigation Results:
{react_context}

Please provide a comprehensive medical response that includes:

1. DETAILED MEDICAL ASSESSMENT: A thorough analysis of the symptoms/condition
2. RECOMMENDED TREATMENTS: Specific medications, dosages, and treatments
3. LIFESTYLE RECOMMENDATIONS: Diet, rest, and care instructions
4. WARNING SIGNS: When to seek immediate medical attention
5. FOLLOW-UP GUIDANCE: When to see a healthcare provider
6. CONFIDENCE LEVEL: Your confidence in this assessment (HIGH/MEDIUM/LOW)

Make your response complete, detailed, and medically sound. Do not truncate or summarize - provide full comprehensive guidance."""
            
            try:
                response = self.llm_deterministic.invoke([SystemMessage(content=final_prompt)])
                full_response = response.content
                
                if len(full_response) < 200:
                    print("⚠️ Response seems incomplete, regenerating...")
                    response = self.llm.invoke([SystemMessage(content=final_prompt)])
                    full_response = response.content
                
                state["final_answer"] = full_response
                state["messages"].append(AIMessage(content=full_response))
                
                print(f"✅ Generated final answer with {len(full_response)} characters")
                
                if "HIGH" in full_response.upper():
                    state["confidence_score"] = 0.9
                    state["reasoning_quality"] = "HIGH"
                elif "MEDIUM" in full_response.upper():
                    state["confidence_score"] = 0.7
                    state["reasoning_quality"] = "MEDIUM"
                else:
                    state["confidence_score"] = 0.5
                    state["reasoning_quality"] = "LOW"
                    
            except Exception as e:
                print(f"❌ Error generating final answer: {str(e)}")
                state["final_answer"] = f"I apologize, but I encountered an error while generating the final response. Please consult with a healthcare professional regarding your symptoms. Error: {str(e)}"
                state["messages"].append(AIMessage(content=state["final_answer"]))
        
        return state

# Create enhanced agent
enhanced_agent = CoTSCReActAgent(llm, llm_deterministic, TOOLS, max_iterations=3, num_reasoning_paths=3)

# Graph nodes
def cot_reasoning_node(state: CoTSCReActState) -> CoTSCReActState:
    """Generate multiple Chain of Thought reasoning paths"""
    user_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    # Build conversation history for CoT
    conversation_history = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation_history.append({"user": msg.content, "assistant": ""})
        elif isinstance(msg, AIMessage) and conversation_history:
            conversation_history[-1]["assistant"] = msg.content
    
    print("🧠 Generating Chain of Thought reasoning paths...")
    reasoning_paths = enhanced_agent.generate_cot_reasoning_paths(user_question, conversation_history=conversation_history)
    state["reasoning_paths"] = reasoning_paths
    
    return state

def consistency_check_node(state: CoTSCReActState) -> CoTSCReActState:
    """Perform self-consistency check"""
    user_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    # Build conversation history for consistency check
    conversation_history = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation_history.append({"user": msg.content, "assistant": ""})
        elif isinstance(msg, AIMessage) and conversation_history:
            conversation_history[-1]["assistant"] = msg.content
    
    print("🔍 Performing self-consistency check...")
    consistency_result = enhanced_agent.perform_self_consistency_check(state["reasoning_paths"], user_question, conversation_history=conversation_history)
    state["consistency_check"] = consistency_result
    
    print(f"✅ Consistency Score: {consistency_result.get('consistency_score', 'N/A')}")
    print(f"🎯 Reliability: {consistency_result.get('reliability', 'N/A')}")
    
    return state

def enhanced_react_node(state: CoTSCReActState) -> CoTSCReActState:
    """Enhanced ReAct step with CoT integration"""
    return enhanced_agent.react_step_with_cot(state)

def should_continue_node(state: CoTSCReActState) -> str:
    """Decide whether to continue ReAct or finalize"""
    if enhanced_agent.should_continue_react(state):
        return "continue"
    else:
        return "finalize"

def finalize_node(state: CoTSCReActState) -> CoTSCReActState:
    """Generate final answer with confidence"""
    return enhanced_agent.finalize_answer_with_confidence(state)

# Build the enhanced graph
workflow = StateGraph(CoTSCReActState)

# Add nodes
workflow.add_node("cot_reasoning", cot_reasoning_node)
workflow.add_node("consistency_analysis", consistency_check_node)
workflow.add_node("enhanced_react", enhanced_react_node)
workflow.add_node("finalize", finalize_node)

# Add edges
workflow.set_entry_point("cot_reasoning")
workflow.add_edge("cot_reasoning", "consistency_analysis")
workflow.add_edge("consistency_analysis", "enhanced_react")
workflow.add_conditional_edges(
    "enhanced_react",
    should_continue_node,
    {
        "continue": "enhanced_react",
        "finalize": "finalize"
    }
)
workflow.add_edge("finalize", END)

# Compile the enhanced graph
enhanced_app = workflow.compile(checkpointer=memory)

# Enhanced system prompt
ENHANCED_MEDICAL_SYSTEM_PROMPT = """
You are an advanced medical assistant that combines Chain of Thought reasoning with Self-Consistency checking and ReAct methodology.

Your approach:
1. Generate multiple reasoning paths for complex medical questions
2. Check consistency across different reasoning approaches
3. Use tools to gather current, accurate medical information
4. Provide evidence-based, safe medical guidance
5. Always prioritize patient safety and recommend professional consultation

Key principles:
- Multiple reasoning perspectives improve accuracy
- Self-consistency checking reduces errors
- Tool use ensures current information
- Patient safety is paramount
- Clear confidence assessment helps users understand reliability
"""

def run_enhanced_medical_agent(user_input: str, thread_id: str = None) -> str:
    """Run the enhanced CoT-SC + ReAct medical agent with conversation history"""
    
    # Generate or reuse thread_id for conversation continuity
    if not thread_id:
        thread_id = f"enhanced_{datetime.now().timestamp()}"
    
    config = RunnableConfig(configurable={"thread_id": thread_id})
    
    # Retrieve previous conversation from memory
    conversation_history = []
    try:
        checkpoint = memory.get(config)
        if checkpoint and checkpoint.get("state") and checkpoint["state"].get("messages"):
            for msg in checkpoint["state"]["messages"]:
                if isinstance(msg, HumanMessage):
                    conversation_history.append({"user": msg.content, "assistant": ""})
                elif isinstance(msg, AIMessage) and conversation_history:
                    conversation_history[-1]["assistant"] = msg.content
        print(f"[DEBUG] Retrieved {len(conversation_history)} previous interactions from memory")
    except Exception as e:
        print(f"⚠️ Error retrieving conversation history: {str(e)}")
    
    # Initialize enhanced state with full conversation history
    initial_state = {
        "messages": [
            SystemMessage(content=ENHANCED_MEDICAL_SYSTEM_PROMPT),
        ] + [HumanMessage(content=entry["user"]) for entry in conversation_history] +
          [AIMessage(content=entry["assistant"]) for entry in conversation_history if entry["assistant"]] +
          [HumanMessage(content=user_input)],
        "iteration": 0,
        "reasoning_paths": [],
        "consistency_check": {},
        "react_history": [],
        "current_thought": "",
        "current_action": "",
        "current_action_input": "",
        "current_observation": "",
        "final_answer": "",
        "confidence_score": 0.0,
        "reasoning_quality": ""
    }
    
    print(f"\n🚀 Starting Enhanced CoT-SC + ReAct Analysis")
    print(f"📋 Question: '{user_input}'")
    print(f"🗂 Conversation History: {len(conversation_history)} previous interactions")
    print("=" * 80)
    
    try:
        final_state = enhanced_app.invoke(initial_state, config=config)
        
        # Ensure the final state is saved to memory
        try:
            memory.save(final_state, config)
            print("[DEBUG] Successfully saved conversation state to memory")
        except Exception as e:
            print(f"⚠️ Error saving conversation state: {str(e)}")
        
        print("\n" + "=" * 80)
        print("📊 ANALYSIS SUMMARY:")
        print("=" * 80)
        
        print(f"🧠 Reasoning Paths Generated: {len(final_state['reasoning_paths'])}")
        print(f"🔍 Consistency Score: {final_state['consistency_check'].get('consistency_score', 'N/A')}")
        print(f"🎯 Reliability Level: {final_state['consistency_check'].get('reliability', 'N/A')}")
        print(f"🔄 ReAct Iterations: {len(final_state['react_history'])}")
        print(f"📈 Final Confidence: {final_state.get('reasoning_quality', 'N/A')}")
        
        print(f"\n🎯 ENHANCED MEDICAL RESPONSE:")
        print("=" * 80)
        
        final_answer = final_state.get("final_answer", "")
        if not final_answer:
            return "I apologize, but I was unable to generate a complete response. Please try asking your question again or consult with a healthcare professional."
        
        print(f"[DEBUG] Final answer length: {len(final_answer)} characters")
        
        return final_answer
        
    except Exception as e:
        print(f"❌ Error in workflow execution: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or consult with a healthcare professional."

# Modified main chat loop to maintain thread_id for conversation continuity
if __name__ == "__main__":
    print("\n🏥 Enhanced Medical Agent (CoT-SC + ReAct) is ready!")
    print("🧠 Features: Multiple reasoning paths, self-consistency checking, intelligent tool use, and conversation context retention")
    print("Type 'exit' to quit.\n")
    
    # Initialize thread_id for the conversation session
    thread_id = f"enhanced_{datetime.now().timestamp()}"
    
    while True:
        user_input = input("⚕️ You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "end"]:
            print("🤖 Enhanced Medical Agent: Goodbye! Stay healthy and consult healthcare professionals when needed.")
            break
        
        if not user_input:
            continue
            
        try:
            final_answer = run_enhanced_medical_agent(user_input, thread_id=thread_id)
            print(f"🤖 Enhanced Medical Agent: {final_answer}\n")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("🤖 Enhanced Medical Agent: I apologize, but I encountered an error. Please try rephrasing your question.\n")