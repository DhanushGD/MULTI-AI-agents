# --- tools/custom_tool.py ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import wikipedia

class WikipediaToolInput(BaseModel):
    """Input schema for WikipediaTool."""
    query: str = Field(..., description="The search query for Wikipedia.")

class WikipediaTool(BaseTool):
    name: str = "Wikipedia Search Tool"
    description: str = "Searches Wikipedia for a summary of a given query."
    args_schema: Type[BaseModel] = WikipediaToolInput

    def _run(self, query: str) -> str:
        try:
            summary = wikipedia.summary(query, sentences=2)
            return f"📚 Wikipedia Summary: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"⚠️ Disambiguation error: Be more specific. Options: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return "❌ No page found for that query."
        except Exception as ex:
            return f"🚨 Error fetching Wikipedia info: {str(ex)}"

class DocumentIngestor:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def load_and_split(self):
        documents = []
        for file_path in self.file_paths:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            chunks = self.splitter.split_documents(raw_docs)
            documents.extend(chunks)
        return documents

# Hybrid Search Setup Tool (RAG)
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

def build_hybrid_retriever(docs, embeddings):
    vector_store = FAISS.from_documents(docs, embeddings)
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 4

    hybrid = EnsembleRetriever(retrievers=[semantic_retriever, keyword_retriever], weights=[0.5, 0.5])
    return hybrid
