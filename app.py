import streamlit as st
from langchain_core.messages import HumanMessage
from fitness_agent import fitness_agent

# Personal Fitness Coach Multi Tool Agent
st.set_page_config(page_title="Personal Fitness Coach Multi Tool Agent", page_icon=":running:")

st.title("Personal Fitness Coach Multi Tool Agent")

st.markdown("""
Example:I want to lose weight and build stamina, but I only have 3 days a week to work out.  
Can you create a sample 3-day workout plan for me, with video demos for each exercise?  
Also, calculate how many calories I should eat per day if I’m a 28-year-old female, 68kg, 165cm, moderately active.
""")

with st.form("query_form"):
    user_query = st.text_area("Enter your fitness plan here:", height=60)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Agent is analyzing..."):
        output = fitness_agent.invoke({"messages": [HumanMessage(content=user_query)]})
        # Show **only the last agent message** (the Final Answer)
        final_message = None
        for msg in reversed(output["messages"]):
            content = getattr(msg, "content", "")
            if "final answer" in content.lower():
                final_message = content
                break
        if not final_message:
            # fallback: just show last assistant/system message
            last = output["messages"][-1]
            final_message = getattr(last, "content", str(last))
        st.markdown("**Here’s a clear summary of your requests and answers:**\n\n" + final_message)
