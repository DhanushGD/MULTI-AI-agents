import streamlit as st
import os
from dotenv import load_dotenv
from agent import all_agents, embeddings, WikipediaAgent
from custom_tools import DocumentIngestor, build_hybrid_retriever
from crewai import Task, Crew, Process, Agent
from tempfile import NamedTemporaryFile
import shutil

# Load env
load_dotenv()

st.set_page_config(page_title="AI IT Assistant", layout="centered")

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- Step 1: Authentication ---
st.title("🔐 Authenticate with GROQ API Key")

if not st.session_state.authenticated:
    groq_key_input = st.text_input("Enter your GROQ API Key", type="password")

    if st.button("Login"):
        if groq_key_input and groq_key_input.startswith("gsk_"):
            st.session_state["authenticated"] = True
            os.environ["GROQ_API_KEY"] = groq_key_input
            st.success("Authenticated successfully!")
            st.rerun()
        else:
            st.error("Invalid API Key. Must start with `gsk_`")

# --- Step 2: PDF Upload and Query ---
if st.session_state.authenticated:
    st.title("📁 Upload PDFs and Ask a Question")

    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)
    user_query = st.text_input("Enter your technical query")

    if uploaded_files:
        tmp_dir = "uploaded_pdfs"
        os.makedirs(tmp_dir, exist_ok=True)

        for file in uploaded_files:
            with open(os.path.join(tmp_dir, file.name), "wb") as f:
                shutil.copyfileobj(file, f)

    if st.button("🔍 Submit") and user_query:
        try:
            with st.spinner("Processing your documents..."):
                # Get list of uploaded file paths
                file_paths = [os.path.join(tmp_dir, file.name) for file in uploaded_files]
                ingestor = DocumentIngestor(file_paths=file_paths)

                documents = ingestor.load_and_split()
                retriever = build_hybrid_retriever(documents, embeddings)

            with st.spinner("Running agents to generate response..."):
                crew_tasks = [
                    Task(
                        description="Ingest and chunk uploaded PDFs.",
                        agent=all_agents[0],
                        expected_output="List of document chunks ready for retrieval."
                    ),
                    Task(
                        description=f"Retrieve relevant documents for: '{user_query}'",
                        agent=all_agents[1],
                        expected_output="Top 4 relevant text chunks."
                    ),
                    Task(
                        description=f"Answer the query: '{user_query}'",
                        agent=all_agents[2],
                        expected_output="Clear, technical response answering the user's query."
                    ),
                    Task(
                        description=f"Suggest next actions for: '{user_query}'",
                        agent=all_agents[3],
                        expected_output="Follow-up actions, automation ideas, or improvement suggestions."
                    ),
                    Task(
                        description=f"Validate or test the solution for: '{user_query}'",
                        agent=all_agents[4],
                        expected_output="Validation steps, test cases, or scripts for verifying the solution."
                    ),]

                crew = Crew(
                agents=all_agents,
                tasks=crew_tasks,
                process=Process.sequential,
                memory=False,
                cache=False,
                verbose=True,
                max_rpm=100,
                share_crew=True
            )
                result = crew.kickoff(inputs={"query": user_query})
                if not result or "no relevant documents found" in result:
                    st.warning("⚠️ No relevant information found in the uploaded documents. Trying Wikipedia...")
                    try:
                        # Manually run the Wikipedia tool using agent
                        wiki_task = Task(
                            description=f"Answer this using Wikipedia: '{user_query}'",
                            agent=WikipediaAgent,
                            expected_output="Concise and informative summary from Wikipedia."
                        )
                        wiki_crew = Crew(
                            agents=[WikipediaAgent],
                            tasks=[wiki_task],
                            process=Process.sequential,
                            verbose=True
                        )
                        result = wiki_crew.kickoff(inputs={"query": user_query})

                    except Exception as e:
                        result = f"❌ Wikipedia fallback failed: {e}"

                st.success("✅ Query Processed!")
                st.markdown("### 🧠 Final Answer:")
                st.markdown(result)

        except Exception as e:
            st.error(f"Error: {e}")
