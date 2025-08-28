# create an streamlit app with simple user interface and side bar to input api keys and on the UI, user can add
# their research queries for deep research agent to answer the them

import streamlit as st
from agents import DeepResearchAgent
import time
from dotenv import load_dotenv
import base64

st.set_page_config(page_title="Deep Research Agent", page_icon=":mag_right:")

st.title("Deep Research Agent")

# Sidebar for API keys
st.sidebar.header("API Keys")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")
nebius_api_key = st.sidebar.text_input("Nebius API Key", type="password")

# Main content area
st.header("Research Query")
query = st.text_area("Enter your research query here:")

if st.button("Submit"):
    with st.spinner("Running deep research..."):
        # Initialize the deep research agent
        research_agent = DeepResearchAgent()
        final_report = research_agent.run(query)

        st.success("Deep research completed!")
        st.markdown(final_report)
