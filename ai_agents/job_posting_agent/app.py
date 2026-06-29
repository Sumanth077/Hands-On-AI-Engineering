"""Streamlit app for the Job Posting Agent. Enter a company and role to generate a tailored, professional job posting using a multi-agent LangChain pipeline backed by DeepSeek V4 Flash on NVIDIA NIM."""

import json
import logging
import traceback

logging.basicConfig(level=logging.INFO, format="%(message)s")

import streamlit as st
import streamlit.components.v1 as components

from agents.pipeline import run_pipeline

st.set_page_config(
    page_title="Job Posting Agent",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Job Posting Agent")
st.markdown(
    "Enter a company and role. Three LangChain agents research the company, "
    "define requirements, and write a tailored job posting."
)

col1, col2 = st.columns(2)
with col1:
    company = st.text_input("Company Name", placeholder="e.g. Stripe")
with col2:
    role = st.text_input("Job Role", placeholder="e.g. Senior Software Engineer")

generate = st.button("Generate Job Posting", type="primary", use_container_width=True)

if generate:
    if not company.strip() or not role.strip():
        st.error("Please enter both a company name and a job role.")
    else:
        try:
            with st.status("Running agents...", expanded=True) as status:

                def on_stage(message: str) -> None:
                    status.update(label=message)

                result = run_pipeline(
                    company.strip(),
                    role.strip(),
                    on_stage=on_stage,
                )
                status.update(label="All agents finished.", state="complete")

            st.divider()
            st.subheader("Final Job Posting")

            st.text_area(
                "Job Posting",
                value=result.job_posting,
                height=420,
                label_visibility="collapsed",
            )

            posting_json = json.dumps(result.job_posting)
            components.html(
                f"""
                <button id="copy-btn" style="
                    margin-top: 0.5rem;
                    padding: 0.5rem 1.25rem;
                    background: #ff4b4b;
                    color: white;
                    border: none;
                    border-radius: 0.5rem;
                    font-size: 0.95rem;
                    cursor: pointer;
                ">Copy to Clipboard</button>
                <span id="copy-status" style="margin-left: 0.75rem; color: #0a7;"></span>
                <script>
                    const text = {posting_json};
                    document.getElementById('copy-btn').onclick = async () => {{
                        try {{
                            await navigator.clipboard.writeText(text);
                            document.getElementById('copy-status').textContent = 'Copied!';
                            setTimeout(() => {{
                                document.getElementById('copy-status').textContent = '';
                            }}, 2000);
                        }} catch (e) {{
                            document.getElementById('copy-status').textContent = 'Copy failed';
                        }}
                    }};
                </script>
                """,
                height=60,
            )

            with st.expander("Company Research", expanded=False):
                st.markdown(result.company_research)

            with st.expander("Role Requirements", expanded=False):
                st.markdown(result.role_requirements)

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            st.caption(
                "Check your terminal for `[Pipeline]` / `[Research Analyst]` logs. "
                "Confirm `NVIDIA_API_KEY` in `.env` is valid and you have internet access."
            )

else:
    st.info("Fill in the form above and click **Generate Job Posting** to start.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Research Analyst**: searches DuckDuckGo for company culture, values, and news
        2. **Job Requirements Specialist**: defines skills and qualifications for the role
        3. **Job Posting Writer**: produces a polished, tailored job posting

        Powered by **DeepSeek V4 Flash** via NVIDIA NIM.
        """
    )
    st.markdown("---")
    st.caption("Set `NVIDIA_API_KEY` in your `.env` file to run.")
    st.caption("Watch the terminal running Streamlit for step-by-step logs.")
    st.caption(
        "First NVIDIA request may take 30–60s while the model warms up "
        "(you will see polling messages in the terminal)."
    )
