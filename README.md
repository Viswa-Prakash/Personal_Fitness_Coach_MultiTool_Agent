# Personal Fitness Coach ReAct Agent

A conversational AI agent (built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Streamlit](https://streamlit.io/)) that helps you:

- Generate custom workout plans for your fitness goals.
- Find exercise video demo links pulled from real web sources.
- Calculate daily caloric and macronutrient needs using best-practice formulas.
- Carry out any necessary math or health calculations using Python.
- Reason step-by-step, using tools as needed, always responding with a clear “Final answer:” summary.

---

##  Features

- **ReAct-style multi-tool reasoning:**  
  The AI assistant breaks your request into steps, uses the best tool for each, and synthesizes a clear, actionable result.
- **Integrated tools:**  
  - Workout plan generator (via LLM tool)
  - Web search for demo videos (DuckDuckGo, Tavily, or SerpAPI)
  - Calorie/health calculations (Python REPL or custom tool)
  - Currency conversion for international users
- **Friendly Streamlit frontend:**  
  Enter your prompt, see a single, formatted “Final answer” per request.
- **Expandable:**  
  Add more tools (supplements, recipes, equipment search, etc).

---

## 🛠️ Installation

   ```bash
1. **Clone this repo:**  
   ```bash
   git clone https://github.com/Viswa-Prakash/Personal_Fitness_Coach_MultiTool_Agent.git
   cd Personal_Fitness_Coach_MultiTool_Agent

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Set up API keys:**
   ```bash
    - OpenAI API Key
    - SERPAPI Key
    - TAVILY API Key 

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 📝 Usage

1. **Enter your prompt:**  
   Start by typing your request in the input box. The AI assistant will break it down into steps and use the best tool for each.
    Example:
    I want a 3-day fat loss plan with workout video demos, and estimate daily calories for a moderately active 30-year-old male, 80kg, 175cm.
2. **See the breakdown:**  
   The assistant will show you a step-by-step breakdown of its reasoning, using the integrated tools as needed.
3. **Get your final answer:**