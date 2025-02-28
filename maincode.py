import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from tavily import TavilyClient  # ✅ Tavily for web search
from huggingface_hub import InferenceClient

# ✅ Load API Keys from .env file
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

# ✅ Validate API Keys
if not tavily_api_key:
    raise ValueError("❌ Tavily API key is missing. Check your .env file.")
if not huggingface_api_key:
    raise ValueError("❌ Hugging Face API key is missing. Check your .env file.")

# ✅ Initialize Tavily Client
client = TavilyClient(api_key=tavily_api_key)

# ✅ Define Research State
@dataclass
class ResearchState:
    query: str
    research_data: list
    answer_draft: str

def research_agent(state: ResearchState):
    """Uses TavilyClient to get web search results."""
    print(f"🔍 Researching: {state.query}")
    
    try:
        response = client.search(query=state.query, num_results=5)
        research_data = [res.get("content", "No content available.") for res in response.get("results", [])]
        
        if not research_data or all(not item.strip() for item in research_data):
            print("⚠️ No useful search results found.")
            research_data = ["No relevant search results."]
        
        return ResearchState(query=state.query, research_data=research_data, answer_draft="")
    except Exception as e:
        print(f"❌ Tavily API error: {e}")
        return ResearchState(query=state.query, research_data=["Error fetching results"], answer_draft="")

# ✅ Initialize Hugging Face Client
hf_client = InferenceClient(model="tiiuae/falcon-7b-instruct", token=huggingface_api_key)

def sanitize_response(response_text):
    """Removes corrupted characters from AI output."""
    return ''.join(char for char in response_text if char.isprintable())

def answer_drafting_agent(state: ResearchState):
    """Generates an answer using Hugging Face inference API."""
    print("✍️ Generating answer using Hugging Face...")
    
    try:
        prompt = f"""
        Based on the following research results, generate a structured, insightful answer.
        Avoid excessive bullet points or repetitive information. Make the answer concise and readable.

        Research Data:
        {state.research_data}

        Format the answer with short paragraphs and use headings when necessary.
        """
        
        response = hf_client.text_generation(prompt, max_new_tokens=500)
        print("🔎 Raw Hugging Face Response:", response)

        answer_text = sanitize_response(response.strip()) if isinstance(response, str) else "Error: No answer generated."

        # Clean up and reformat the response
        if not answer_text or "\uFFFD" in answer_text:
            print("⚠️ Invalid AI response detected. Retrying with fallback response...")
            answer_text = "Error: The AI response was corrupted. Please try again."

        # Post-process to remove excessive bullet points and format properly
        cleaned_answer = '\n'.join(line for line in answer_text.split('\n') if line.strip())  # Remove empty lines
        cleaned_answer = cleaned_answer.replace("•", "-")  # Replace bullet points with dashes

        return ResearchState(query=state.query, research_data=state.research_data, answer_draft=cleaned_answer)
    except Exception as e:
        print(f"❌ Hugging Face API error: {e}")
        return ResearchState(query=state.query, research_data=state.research_data, answer_draft="Error fetching response.")

# ✅ Define LangGraph Workflow
graph = StateGraph(ResearchState)
graph.add_node("research", research_agent)
graph.add_node("draft", answer_drafting_agent)

graph.add_edge("research", "draft")
graph.add_edge("draft", END)
graph.set_entry_point("research")

executor = graph.compile()

def run_research_system(user_query):
    state = ResearchState(query=user_query, research_data=[], answer_draft="")
    print("🔎 Initial State:", state)  # Log the initial state
    
    final_state = executor.invoke(state)
    
    print("🔎 Final state:", final_state)  # Debugging print
    print("Type of final state:", type(final_state))  # Check the type of final state

    if isinstance(final_state, dict):  
        final_state = ResearchState(**final_state)  # Convert to ResearchState

    if isinstance(final_state, ResearchState):
        return final_state.answer_draft or "Error: No answer generated."
    else:
        return f"Error: Unexpected final state structure. Received type: {type(final_state)}"

if __name__ == "__main__":
    query = input("Enter your research query: ")
    answer = run_research_system(query)
    print("\n💡 Final Answer:\n", answer)
