from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage, AnyMessage

from langchain_community.utilities import SerpAPIWrapper
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.tools.python.tool import PythonREPLTool 

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


llm = init_chat_model("gpt-4.1", temperature=0.7)

serpapi_tool = Tool(
    name="serpapi",
    description="Search the web or YouTube for up-to-date fitness routines, exercise demos, nutrition tips, and product reviews.",
    func=SerpAPIWrapper().run,
)

repl_tool = PythonREPLTool()  # Calculates or solves any Python-based fitness/math/nutrition formula.

duckduckgo_tool = Tool(
    name="duckduckgo_search",
    description="Finds fitness articles, workout plans, video demos, and health information across the web.",
    func=DuckDuckGoSearchResults().run,
)

tavily_tool = Tool(
    name="tavily_search",
    description="Search the internet for exercise tutorials, health calculators, and gym or trainer recommendations in real time.",
    func=TavilySearch().run,
)

@tool
def workout_plan(focus: str, days: int) -> str:
    """
    Generate a workout plan for the given focus (e.g., fat loss, muscle gain) and number of days.
    """
    return f"Workout plan for {days} days focusing on {focus}:\n-Day 1: HIIT\n-Day 2: Strength\n-Day 3: Cardio"


# Create the list of properly wrapped Tool instances
tools = [serpapi_tool, workout_plan, repl_tool, duckduckgo_tool, tavily_tool]

react_prompt = """
You are an intelligent AI personal fitness and health assistant.

You have access to the following tools:
- WorkoutPlan: Generates multi-day workout plans for goals like fat loss, muscle gain, or sports performance.
- WebSearch (Tavily, DuckDuckGo, or SerpAPI): Finds exercise video demos, health articles, and nutrition information from the latest web sources or YouTube.
- CalorieCalculator: Calculates daily caloric needs, macronutrient targets, or BMI for a person.
- Python REPL: Performs any necessary calculations (e.g., caloric intake, body fat formulas, average workout time).

For each user request:
1. **Understand** what they want (plan, calculation, videos, or advice).
2. **Break down** the problem into subtasks, thinking aloud at each step ("Thought: ...").
3. **For each subtask, select the best tool** and use it:
   - Use this format for Tool Use:  
     Thought: Explain what step you're taking and why.  
     Action: (name of the tool, e.g. WorkoutPlan, DuckDuckGo, Tavily, CalorieCalculator, Python REPL)  
     Action Input: Describe the exact input for the tool.  
     Observation: Write what the tool returned.
4. If you get unclear or unhelpful results, try an alternative tool or suggest what the user could try next.
5. After completing every step, ALWAYS finish with a message that starts with **"Final answer:"**.  
   - Summarize the workout plan, calculations, links, or other findings you discovered.
   - Restate all numbers (calories, times, weights, etc.) clearly, and give a practical closing tip or recommendation.

**IMPORTANT:**  
Only output a message starting with "Final answer:" at the end, and do not use this phrase in intermediate steps.

---

**Example user request:**  
Help me create a 3-day workout plan for fat loss. Give me links to video demos for each exercise and calculate daily caloric needs for a 30-year-old male, 80kg, 175cm, moderately active.

---

Your reasoning and outputs should look like this:

Thought: I need to generate a 3-day fat loss plan.
Action: WorkoutPlan
Action Input: focus: fat loss, days: 3
Observation: [Day 1: HIIT; Day 2: Strength...]
Thought: Now I’ll find exercise demo videos for each movement.
Action: DuckDuckGoSearch
Action Input: "HIIT routine video"
Observation: [URL]
Thought: Next, I’ll calculate calorie needs for a 30-year-old male, 80kg, 175cm, moderately active.
Action: Python REPL
Action Input: # Harris-Benedict formula calculation using weight, height, age, and activity
Observation: 2500 kcal/day
...
Final answer:  
Here is your 3-day fat loss workout plan: Day 1: HIIT (see this video demo [YouTube URL]), Day 2: Strength, Day 3: Cardio. Your estimated daily calorie needs are 2,500 kcal. Drink water and sleep well for best results!

"""


class State(TypedDict):
    messages : Annotated[list[AnyMessage], add_messages]

def reasoning_node(state: State):
    # LLM with bound tools to enable tool-calling
    llm_with_tools = llm.bind_tools(tools)
    messages = [{"role": "system", "content": react_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}


tool_node = ToolNode(tools = tools)


def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "content") and "final answer:" in last_message.content.lower():
        return "end"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    if len(state["messages"]) > 20:
        return "end"
    # Otherwise, no tool_calls, not a final answer, so end gracefully
    return "end"


builder = StateGraph(State)
builder.add_node("reason", reasoning_node)
builder.add_node("action", tool_node)
builder.set_entry_point("reason")
builder.add_conditional_edges(
    "reason",
    should_continue,
    {
        "continue": "action",
        "end": END,
    }
)
builder.add_edge("action", "reason")
fitness_agent = builder.compile()
