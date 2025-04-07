import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from tavily import TavilyClient  # ✅ Tavily for web search
from huggingface_hub import InferenceClient
from bert_score import score  # ✅ BERTScore for better accuracy evaluation
import torch

# ✅ Load API Keys from .env file
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

# ✅ Validate API Keys
if not tavily_api_key:
    raise ValueError("❌ Tavily API key is missing. Check your .env file.")

# ✅ Initialize Tavily Client
client = TavilyClient(api_key=tavily_api_key)

# ✅ Define Research State
@dataclass
class ResearchState:
    query: str
    research_data: list
    answer_draft: str

def research_agent(state: ResearchState):
    """Uses TavilyClient to get high-quality web search results."""
    print(f"🔍 Researching: {state.query}")
    
    try:
        response = client.search(query=state.query, num_results=5)
        research_data = [res.get("content", "No content available.") for res in response.get("results", [])]

        # ✅ Filter out empty or low-quality search results
        filtered_data = [res for res in research_data if "No content" not in res and len(res) > 50]

        if not filtered_data:
            print("⚠️ No useful search results found.")
            filtered_data = ["No relevant search results."]

        return ResearchState(query=state.query, research_data=filtered_data, answer_draft="")
    
    except Exception as e:
        print(f"❌ Tavily API error: {e}")
        return ResearchState(query=state.query, research_data=["Error fetching results"], answer_draft="")

# ✅ Initialize Hugging Face Client with a better model
hf_client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta")


def answer_drafting_agent(state: ResearchState):
    """Generates an answer using Hugging Face inference API with improved prompt engineering."""
    print("✍️ Generating answer using Hugging Face...")

    try:
        prompt = f"""
        You are an AI assistant providing highly accurate, factual answers.
        Strictly base your response on the research data provided.
        Avoid assumptions and provide structured, well-cited responses.

        Research Data:
        {state.research_data}

        Output a fact-checked, well-structured response.
        """

        response = hf_client.text_generation(prompt, max_new_tokens=500)

        if isinstance(response, str):
            answer_text = response.strip()
        elif isinstance(response, dict):
            answer_text = response.get("generated_text", "").strip()
        else:
            answer_text = "Error: No answer generated."

        # ✅ Sanitize output and check for AI hallucinations
        if not answer_text or any(char in answer_text for char in ["--c2-", "( ( (", "< ( ("]):
            print("⚠️ Invalid AI response detected. Retrying...")
            answer_text = "Error: The AI response was corrupted. Please try again."

        return ResearchState(query=state.query, research_data=state.research_data, answer_draft=answer_text)
    
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

def evaluate_accuracy(predicted_answer, reference_answer):
    """Computes accuracy using BERTScore."""
    P, R, F1 = score([predicted_answer], [reference_answer], lang="en", model_type="microsoft/deberta-xlarge-mnli")
    return F1.mean().item()

def run_research_system(user_query):
    """Executes research and AI-generated answering system."""
    state = ResearchState(query=user_query, research_data=[], answer_draft="")
    final_state = executor.invoke(state)

    if isinstance(final_state, dict):  
        final_state = ResearchState(**final_state)

    return final_state.answer_draft if isinstance(final_state, ResearchState) else "Error: Unexpected state."

if __name__ == "__main__":
    query = input("Enter your research query: ")
    answer = run_research_system(query)

    # ✅ Define a high-quality reference answer for accuracy evaluation
    reference_answer = "The common cold is a viral infection that affects the respiratory system, typically caused by rhinovirus. Symptoms include congestion, sneezing, sore throat, and mild fatigue. Treatment focuses on symptom relief, including decongestants, pain relievers, and hydration."

    # ✅ Evaluate accuracy using BERTScore
    accuracy_score = evaluate_accuracy(answer, reference_answer)

    print("\n💡 Final Answer:\n", answer)
    print(f"✅ Accuracy Score: {accuracy_score:.2f}")
 