# --- agent.py ---
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from crewai import Agent
from custom_tools import WikipediaTool

wikipedia_tool = WikipediaTool()

# Load environment variables
load_dotenv()
# Set HF token (optional depending on hosting method)
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Load embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="groq/gemma2-9b-it"  # Do not prefix with 'groq/' when using langchain_groq
)

        
# Define Agents
DocIngestorAgent = Agent(
    name="DocIngestorAgent",
    role="Preprocessor",
    goal="Load and split technical documents",
    backstory="Expert in parsing internal IT runbooks and technical PDFs",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

RetrieverAgent = Agent(
    name="RetrieverAgent",
    role="Indexer",
    goal="Index and retrieve relevant documents using hybrid search",
    backstory="Knows how to combine semantic and keyword search",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

ResponderAgent = Agent(
    name="ResponderAgent",
    role="Q&A Expert",
    goal="Answer IT-related technical questions accurately",
    backstory="Helps resolve incidents by referencing documentation and best practices",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

SuggesterAgent = Agent(
    name="SuggesterAgent",
    role="Optimizer",
    goal="Suggest improvements or next actions",
    backstory="Provides optimization recommendations for IT processes",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

TesterAgent = Agent(
    name="TesterAgent",
    role="Validator",
    goal="Suggest tests or validation steps for fixes",
    backstory="Helps confirm whether the proposed solution actually resolves the issue",
    verbose=True,
    allow_delegation=False,
    llm=llm
)
WikipediaAgent = Agent(
    name="WikipediaAgent",
    role="Knowledge Source",
    goal="Answer questions using Wikipedia when not found in uploaded documents",
    backstory="Expert in searching and summarizing reliable information from Wikipedia.",
    tools=[wikipedia_tool],  # Attach the tool here
    verbose=True,
    allow_delegation=False,
    llm=llm

)
all_agents = [
    DocIngestorAgent,
    RetrieverAgent,
    ResponderAgent,
    SuggesterAgent,
    TesterAgent,
    WikipediaAgent
]
