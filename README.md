# **AI-Powered Research & Answering System using LangGraph, LangChain, Tavily, and Hugging Face**

## **Overview**
This project builds an **AI-powered research assistant** that:
1. **Searches the web** using **Tavily API** to gather relevant information.
2. **Processes the results** and structures an insightful response.
3. **Generates a well-formatted answer** using **Hugging Face’s Falcon-7B** model.
4. **Manages workflow** with **LangGraph**, ensuring smooth execution from query to answer.
5. **Leverages LangChain** for potential future **retrieval-augmented generation (RAG)** and **LLM-powered research workflows**.

---

## **Features**
✅ **Automated Web Research**: Uses **Tavily API** for real-time, relevant search results.  
✅ **AI-Powered Answer Generation**: Utilizes **Hugging Face Falcon-7B** for response synthesis.  
✅ **LangGraph Workflow Management**: Structured **stateful execution** for consistent results.  
✅ **LangChain Integration**: Future-ready for **document retrieval**, **memory**, and **agent-based reasoning**.  
✅ **Secure API Handling**: Uses **dotenv** for managing API keys securely.  

---

## **Technologies Used**
- **Python**
- **LangGraph** (for structured research workflows)
- **LangChain** (for advanced RAG and memory-based interactions)
- **Tavily API** (for real-time web searches)
- **Hugging Face API** (for LLM-powered text generation)
- **Dotenv** (for environment variable management)

---

## **Setup Instructions**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### **2️⃣ Install Dependencies**
Ensure you have Python installed (`>=3.8`). Then, install required libraries:
```sh
pip install langgraph langchain tavily-python huggingface_hub python-dotenv requests
```

### **3️⃣ Configure API Keys**
Create a `.env` file in the root directory and add your **Tavily** and **Hugging Face** API keys:
```sh
TAVILY_API_KEY=your_tavily_api_key_here
HUGGINGFACEHUB_API_KEY=your_huggingface_api_key_here
```
⚠️ **Do not share API keys**. Add `.env` to `.gitignore` before committing.

### **4️⃣ Run the System**
Execute the script:
```sh
python main.py
```
You will be prompted to enter a query:
```
Enter your research query: <your-question-here>
```
The system will **search**, **generate an AI response**, and **display the final answer**.

---

## **How It Works**
1. **User enters a query** (e.g., *"What are the latest AI trends?"*).
2. **Tavily fetches search results** (top 5 web pages).
3. **LangGraph manages workflow execution**.
4. **Data is passed to Hugging Face Falcon-7B** for response generation.
5. **Final Answer is displayed**.

---

## **LangGraph vs. LangChain**
| Feature | LangGraph | LangChain |
|---------|----------|-----------|
| **Primary Use** | Workflow automation for AI agents | LLM-powered reasoning & memory |
| **Execution Model** | Graph-based state transitions | Chain-of-thought & agents |
| **Ideal For** | Structured research, data pipelines | Conversational AI, RAG |
| **Integration** | Can be used **with** LangChain | Can use LangGraph for structured execution |

This project **uses LangGraph** for structured workflow execution but can be extended **with LangChain** for **memory** and **retrieval-augmented generation (RAG)**.

---

## **Code Breakdown**
### **Research Workflow**
- **`research_agent(state: ResearchState)`**: Fetches data using **Tavily API**.
- **`answer_drafting_agent(state: ResearchState)`**: Generates a structured answer using **Hugging Face Falcon-7B**.
- **LangGraph manages execution**, ensuring smooth transitions.

### **Core Data Model**
```python
@dataclass
class ResearchState:
    query: str
    research_data: list
    answer_draft: str
```
- **`query`**: User input.
- **`research_data`**: Web search results.
- **`answer_draft`**: AI-generated response.

---
## **Future Enhancements**
🔹 **Integrate LangChain’s Retrieval-Augmented Generation (RAG)** for better long-term research.  
🔹 **Use NVIDIA DeepSeek-R1** for better AI-generated insights.  
🔹 **Implement a UI** using Streamlit or Flask for a user-friendly interface.  

---

## **Example Output**
![image](https://github.com/user-attachments/assets/856778dd-b8fe-4938-bbd0-058aa55396cf)
![image](https://github.com/user-attachments/assets/eaec0f6e-43af-454e-975d-c85590bc203a)

---

## **Contributing**
Pull requests are welcome!  
For major changes, please open an issue first to discuss your ideas.  

---

## **License**
MIT License - Feel free to use and modify this project.
  









